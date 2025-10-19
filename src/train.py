import os
import numpy as np
import torch
import contextlib
from tqdm.auto import tqdm as tqdm_auto

if os.getenv("PJRT_DEVICE", "CPU") == "TPU":
    is_xla = True
else:
    is_xla = False

def _base_loader_len(loader):
    # If wrapped by MpDeviceLoader, the real DataLoader is in .loader
    base = getattr(loader, "loader", loader)
    return len(base)

def set_sampler_epoch(loader, epoch):
    sampler = getattr(loader, "sampler", None)
    # for MpDeviceLoader, underlying DataLoader is in .loader
    if sampler is None and hasattr(loader, "loader"):
        sampler = getattr(loader.loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

def _probe(msg, xm):
    print(f"[rank {xm.get_local_ordinal()}] {msg}", flush=True)

def _safe_mean(xs, default=np.nan):
    return float(np.mean(xs)) if len(xs) else default

def training_step(
        model,
        loader,
        loss_fn,
        loss_weights,
        optimizer,
        device,
        prefix="",
        xm=None
):

    if is_xla:
        is_master = xm.is_master_ordinal()
        device_loader = loader
        pbar = None
        if is_master:
            from tqdm import tqdm
            base_total = _base_loader_len(loader)
            pbar = tqdm(total=base_total, desc=f"[{prefix}] Train")
    else:
        from tqdm.auto import tqdm
        print("Running on non-TPU device.")
        is_master = True
        device_loader = tqdm(loader, dynamic_ncols=True)

    model.train()

    # iterator = tqdm(loader, dynamic_ncols=True)
    cls_loss_avg = []
    loc_loss_avg = []
    total_loss_avg = []

    base_total = len(getattr(device_loader, "loader", device_loader))
    print(f"[rank {xm.get_local_ordinal()}] train local steps = {base_total}", flush=True)

    for i, batch_sample in enumerate(device_loader):

        if i == 0 and is_xla:
            try:
                rank = xm.get_local_ordinal()

            except Exception:
                rank = -1
            print(f"[rank {rank}] reached first batch — compiling...", flush=True)

        if i < 2:
            _probe(f"got batch {i}", xm)

        optimizer.zero_grad()

        image_batch = torch.stack(batch_sample[0]).to(device)
        box_targets = torch.stack(batch_sample[3]).to(device).float()
        cls_targets = torch.stack(batch_sample[4]).to(device).long()

        if i < 2:
            _probe(f"stacked batch {i}", xm)

        # ctx = torch.autocast("xla", dtype=torch.bfloat16) if is_xla else contextlib.nullcontext()
        # with ctx:
        pred_boxes, pred_labels = model(image_batch)

        loc_loss = loss_fn["loc_loss"](pred_boxes.float(), box_targets, cls_targets)
        cls_loss = loss_fn["cls_loss"](pred_labels.float(), cls_targets)
        total_loss = loss_weights["loc_wt"]*loc_loss + loss_weights["cls_wt"]*cls_loss

        if is_xla:
            xm.mark_step()
            _probe("marked step after forward", xm)

        total_loss.backward()
        if i < 2: _probe("backward done", xm)

        if is_xla:
            xm.optimizer_step(optimizer, barrier=True)
            xm.mark_step()
            if i < 2: _probe("optimizer_step + mark_step done", xm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        optimizer_lr = optimizer.param_groups[0]["lr"]

        cls_loss_avg.append(cls_loss.item())
        loc_loss_avg.append(loc_loss.item())
        total_loss_avg.append(total_loss.item())

        status = f"{prefix}[Train][{i}] Total Loss: {_safe_mean(total_loss_avg):.4f}, "
        status+= f"Loc Loss: {_safe_mean(loc_loss_avg):.4f}, Cls Loss: {_safe_mean(cls_loss_avg):.4f}, "
        status+= f"LR: {optimizer_lr:.3f}"

        if is_master:
            if is_xla:
                pbar.update(1)
                pbar.set_postfix_str(status)
            else:
                device_loader.set_description(status)

    if is_master:
        if is_xla:
            pbar.close()

    return {"loc_loss": np.mean(loc_loss_avg), "cls_loss": np.mean(cls_loss_avg), "total_loss": np.mean(total_loss_avg)}




def validation_step(
        model,
        loader,
        device,
        loss_fn,
        loss_weights,
        metric_fn,
        encoder,
        nms_threshold,
        score_threshold,
        prefix="",
        xm=None
):

    if is_xla:
        is_master = xm.is_master_ordinal()
        device_loader = loader
        pbar = None
        if is_master:
            from tqdm import tqdm
            base_total = _base_loader_len(loader)
            pbar = tqdm(total=base_total, desc=f"[{prefix}] Train")
    else:
        from tqdm.auto import tqdm
        print("Running on non-TPU device.")
        is_master = True
        device_loader = tqdm(loader, dynamic_ncols=True)

    model.eval()

    # iterator = tqdm(loader, dynamic_ncols=True)

    cls_loss_avg = []
    loc_loss_avg = []
    total_loss_avg = []

    metric_fn.reset()


    for i, batch_sample in enumerate(device_loader):

        image_batch = torch.stack(batch_sample[0]).to(device)
        box_targets = torch.stack(batch_sample[3]).to(device)
        cls_targets = torch.stack(batch_sample[4]).to(device)

        with torch.no_grad():
            # ctx = torch.autocast("xla", dtype=torch.bfloat16) if is_xla else contextlib.nullcontext()
            # with ctx:
            pred_boxes, pred_labels = model(image_batch)

        loc_loss = loss_fn["loc_loss"](pred_boxes, box_targets, cls_targets)
        cls_loss = loss_fn["cls_loss"](pred_labels, cls_targets)
        total_loss = loss_weights["loc_wt"]*loc_loss + loss_weights["cls_wt"]*cls_loss

        cls_loss_avg.append(loc_loss.item())
        loc_loss_avg.append(cls_loss.item())
        total_loss_avg.append(total_loss.item())


        # Prepare targets and predictions for evaluations.
        targets = []
        predictions = []

        for idx, (box_raw, label_raw) in enumerate(zip(batch_sample[1], batch_sample[2])):

            if is_xla:
                boxes_raw_per_image  = box_raw
                labels_raw_per_image = label_raw
            else:
                boxes_raw_per_image  = box_raw.to(device)
                labels_raw_per_image = label_raw.to(device)

            prediction_data = encoder.decode(pred_boxes[idx],
                                             pred_labels[idx],
                                             device,
                                             nms_threshold,
                                             score_threshold)

            pred_bbox   = prediction_data[:,:4]
            pred_conf   = prediction_data[:,4]
            pred_cls_id = prediction_data[:,5]

            target_dict = dict(
                boxes  = boxes_raw_per_image,
                labels = labels_raw_per_image
            )

            pred_dict  = dict(
                boxes  = pred_bbox,
                scores = pred_conf,
                labels = pred_cls_id.int()
            )

            targets.append(target_dict)
            predictions.append(pred_dict)


        metric_fn.update(predictions, targets)

        status = f"{prefix}[Test][{i}] Total Loss: {np.mean(total_loss_avg):.4f}, "
        status+= f"Loc Loss: {np.mean(loc_loss_avg):.4f}, Cls Loss: {np.mean(cls_loss_avg):.4f}, "

        if is_master:
            if is_xla:
                pbar.update(1)
                pbar.set_postfix_str(status)
            else:
                device_loader.set_description(status)


    metrics_dict = metric_fn.compute()

    map_50 = float(metrics_dict["map_50"])
    status+= f"val_mAP@50: {map_50:.3f}"

    if is_master:
        if is_xla:
            pbar.update(1)
            pbar.set_postfix_str(status)
        else:
            device_loader.set_description(status)

    if is_master:
        if is_xla:
            pbar.close()

    output = {"loc_loss": np.mean(loc_loss_avg), "cls_loss": np.mean(cls_loss_avg), "total_loss":np.mean(total_loss_avg),
              "metrics": metrics_dict}
    return output


def fit(
        model,
        epochs,
        classes,
        loader_train,
        loader_test,
        loss_fn,
        loss_weights,
        optimizer,
        lr_scheduler,
        encoder,
        metric_fn,
        nms_thresh,
        score_thresh,
        device = "cpu",
        checkpoint_dir = "",
        visualizer = None,
):
    if is_xla:
        from torch_xla.core import xla_model as xm
        from .distributed_utils import save_fsdp_model

    iterator = tqdm_auto(range(epochs), dynamic_ncols=True)

    history = {"epoch": [], "train_loss": [], "test_loss": [],
               "val_mAP": [], "val_mAP@50": []}

    for epoch in iterator:
        set_sampler_epoch(loader_train, epoch)
        set_sampler_epoch(loader_test, epoch)
        output_train = training_step(
            model=model,
            loader=loader_train,
            loss_fn=loss_fn,
            loss_weights=loss_weights,
            optimizer=optimizer,
            device=device,
            prefix=f"[{epoch}/{epochs}]",
            xm=xm if is_xla else None
        )
        output_test  = validation_step(
            model=model,
            loader=loader_test,
            device=device,
            loss_fn=loss_fn,
            loss_weights=loss_weights,
            metric_fn=metric_fn,
            encoder=encoder,
            nms_threshold=nms_thresh,
            score_threshold=score_thresh,
            prefix=f"[{epoch}/{epochs}]",
            xm=xm if is_xla else None
        )


        if visualizer:
            visualizer.update_charts(
                classes,
                None,
                output_test["metrics"],
                output_train["total_loss"],
                output_test["total_loss"],
                output_train["cls_loss"],
                output_train["loc_loss"],
                output_test["cls_loss"],
                output_test["loc_loss"],
                optimizer.param_groups[0]['lr'],
                epoch
            )

        history["epoch"].append(epoch)
        history["train_loss"].append(output_train["total_loss"])
        history["test_loss"].append(output_test["total_loss"])
        history["val_mAP"].append(output_test["metrics"]["map"].numpy())
        history["val_mAP@50"].append(output_test["metrics"]["map_50"].numpy())

        if lr_scheduler is not None:
            lr_scheduler.step()

        end_epoch_verbose(iterator, epoch, output_train, output_test)

        # Save model based on best validation mAP@50.
        best_acc = max(history["val_mAP"])
        current_acc = output_test["metrics"]["map"].numpy()

        if current_acc >= best_acc:
            if device == "TPU":
                save_fsdp_model(model, os.path.join(checkpoint_dir, model.__class__.__name__))
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(checkpoint_dir, model.__class__.__name__) + '_best.pth'
                )

    return history

def end_epoch_verbose(iterator, epoch, output_train, output_test):

    val_map_50 = float(output_test["metrics"]["map_50"])
    train_loss = output_train["total_loss"]
    test_loss  = output_test["total_loss"]

    iterator.set_description(
        f"epoch: {epoch}, val_mAP@50: {val_map_50:.3f}, train_loss: {train_loss:.3f}, test_loss: {test_loss:.3f}"
    )


