import os
import numpy as np
import torch
from tqdm.auto import tqdm

from .distributed_utils import save_fsdp_model


def training_step(
        model,
        loader,
        loss_fn,
        loss_weights,
        optimizer,
        device,
        prefix="",
):

    model.train()

    iterator = tqdm(loader, dynamic_ncols=True)

    cls_loss_avg = []
    loc_loss_avg = []
    total_loss_avg = []

    for i, batch_sample in enumerate(iterator):
        optimizer.zero_grad()
        image_batch = torch.stack(batch_sample[0]).to(device)
        box_targets = torch.stack(batch_sample[3]).to(device)
        cls_targets = torch.stack(batch_sample[4]).to(device)

        pred_boxes, pred_labels = model(image_batch)

        loc_loss = loss_fn["loc_loss"](pred_boxes, box_targets, cls_targets)
        cls_loss = loss_fn["cls_loss"](pred_labels, cls_targets)
        total_loss = loss_weights["loc_wt"]*loc_loss + loss_weights["cls_wt"]*cls_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(1.0) #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        optimizer_lr = optimizer.param_groups[0]["lr"]

        cls_loss_avg.append(loc_loss.item())
        loc_loss_avg.append(cls_loss.item())
        total_loss_avg.append(total_loss.item())

        status = f"{prefix}[Train][{i}] Total Loss: {np.mean(total_loss_avg):.4f}, "
        status+= f"Loc Loss: {np.mean(loc_loss_avg):.4f}, Cls Loss: {np.mean(cls_loss_avg):.4f}, "
        status+= f"LR: {optimizer_lr:.3f}"

        iterator.set_description(status)

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
):

    model.eval()

    iterator = tqdm(loader, dynamic_ncols=True)

    cls_loss_avg = []
    loc_loss_avg = []
    total_loss_avg = []

    metric_fn.reset()


    for i, batch_sample in enumerate(iterator):

        image_batch = torch.stack(batch_sample[0]).to(device)
        box_targets = torch.stack(batch_sample[3]).to(device)
        cls_targets = torch.stack(batch_sample[4]).to(device)

        with torch.no_grad():
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

        iterator.set_description(status)


    metrics_dict = metric_fn.compute()

    map_50 = float(metrics_dict["map_50"])
    status+= f"val_mAP@50: {map_50:.3f}"

    iterator.set_description(status)

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


    iterator = tqdm(range(epochs), dynamic_ncols=True)

    history = {"epoch": [], "train_loss": [], "test_loss": [],
               "val_mAP": [], "val_mAP@50": []}

    for epoch in iterator:
        output_train = training_step(
            model=model,
            loader=loader_train,
            loss_fn=loss_fn,
            loss_weights=loss_weights,
            optimizer=optimizer,
            device=device,
            prefix=f"[{epoch}/{epochs}]"
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
            prefix=f"[{epoch}/{epochs}]"
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


