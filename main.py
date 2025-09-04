import argparse
import pprint
import math
import os

import torch
import torch.optim as optim
from torchinfo import summary
from torchmetrics.detection.mean_ap import MeanAveragePrecision

try:
    import torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    from src.distributed_utils import apply_fsdp_with_ckpt_detector
except Exception:
    print("Failed to import torch_xla. Please ensure that torch_xla is installed.")




from config import Config
from src.data import PlateDataset, DataEncoder
from src.transform import get_augmentations
from src.utils import set_seeds, download_and_unzip_zip
from src.detector import Detector
from src.logging import setup_log_directory
from src.train_utils import build_cosine_warmup, make_param_groups
from src.train import fit
from src.loss import OHEMLoss, SmoothL1Loss, FocalLoss, IoULoss
from src.tensorboard import TensorBoardVisualizer




def build_datasets(cfg, rank, device):
    print(f"Building datasets on rank {rank} and device {device}...")
    if getattr(device, "type", str(device)) == "cpu":
        world_size = 1
    else:
        world_size = xm.xrt_world_size()

    assert cfg.BATCH_SIZE % world_size == 0
    local_batch_size = cfg.BATCH_SIZE // world_size

    ROOT_DIR = cfg.ROOT_DIR
    IMG_SIZE = cfg.IMG_SIZE
    NUM_WORKERS = cfg.NUM_WORKERS

    train_augmentations, valid_augmentations = get_augmentations(height=IMG_SIZE[0], width=IMG_SIZE[1])

    # Create custom datasets.
    train_dataset = PlateDataset(
        root_dir=ROOT_DIR,
        transform=train_augmentations,
        classes=cfg.CLASSES,
        input_size=IMG_SIZE,
        is_train=True,
        debug=cfg.DEBUG_MODE
    )

    val_dataset = PlateDataset(
        root_dir=ROOT_DIR,
        transform=valid_augmentations,
        classes=cfg.CLASSES,
        input_size=IMG_SIZE,
        is_train=False,
        debug=cfg.DEBUG_MODE
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, drop_last=True, shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=train_dataset.collate_fn,
    )

    if world_size > 1:
        train_loader = pl.MpDeviceLoader(train_loader, device)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, drop_last=True, shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=local_batch_size,
        sampler=val_sampler,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=val_dataset.collate_fn,
    )
    if world_size > 1:
        val_loader = pl.MpDeviceLoader(val_loader, device)

    return (
        train_dataset,
        train_loader,
        train_sampler,
        val_dataset,
        val_loader,
        val_sampler,
    )


def build_od_model(cfg, device):
    pi = 0.01

    model = Detector(
        backbone_name=cfg.BACKBONE_MODEL,
        num_classes=len(cfg.CLASSES),
        fpn_channels=cfg.FPN_CHANNELS,
        num_anchors=cfg.NUM_ANCHORS
    )

    prior_bias = -math.log((1 - pi) / pi)
    with torch.no_grad():
        model.cls_head.head[-1].bias.fill_(prior_bias)

    model.to(device)

    if getattr(device, "type", str(device)) == "cpu":
        print("Loaded model")
    else:
        model = apply_fsdp_with_ckpt_detector(model, use_conv_auto_wrap=True)
        xm.master_print(summary(model, input_size=(1,) + cfg.IMG_SIZE[::-1], row_settings=["var_names"]))

    return model


def main_worker(rank, cfg):

    download_and_unzip_zip(cfg.ZIP_URL, cfg.CONTAINER_DATA_DIR)

    cfg, current_version_name = setup_log_directory(cfg)

    if cfg.BACKEND:
        os.environ["PJRT_DEVICE"] = "CPU" if cfg.BACKEND.lower() == "cpu" else "TPU"

    set_seeds(rank)

    try:
        device = xm.xla_device()
        xm.master_print(f"Process {rank} using device: {device}")
        xm.master_print(f"Current version: {current_version_name} with cfg: {pprint.pformat(cfg)}")
    except Exception:
        device = torch.device("cpu")
        print(f"Process {rank} using device: {device}")
        print(f"Current version: {current_version_name} with cfg: {pprint.pformat(cfg)}")

    if cfg.XLA_CACHE:
        xr.initialize_cache(cfg.xla_cache, readonly=False)

    # build datasets
    data_encoder = DataEncoder(input_size=cfg.IMG_SIZE[:2], classes=cfg.CLASSES)
    train_dataset, train_loader, train_sampler, _, val_loader, _ = build_datasets(cfg, rank, device)
    if getattr(device, "type", str(device)) == "cpu":
        print("loaded dataset")
    else:
        xm.rendezvous("loaded dataset")
        xm.master_print(f"\n=== dataset ===\n{pprint.pformat(train_dataset)}\n")

    # build model and loss
    model = build_od_model(cfg, device)
    if getattr(device, "type", str(device)) == "cpu":
        print("loaded model")
    else:
        xm.rendezvous("loaded model")
        xm.master_print(f"\n=== model ===\n{pprint.pformat(model)}\n")

    parameters = list(model.parameters())
    if getattr(device, "type", str(device)) == "cpu":
        print("Placeholder")
    else:
        xm.master_print(f"per-TPU (sharded) parameter num: {sum(p.numel() for p in parameters)}")

    # build optimizer and scheduler
    optimizer = optim.AdamW(
        make_param_groups(model, wd=cfg.WEIGHT_DECAY),
        lr=cfg.INIT_LEARING_RATE, betas=(0.9, 0.999), eps=1e-8
    )

    lr_scheduler = build_cosine_warmup(optimizer, total_epochs=cfg.EPOCHS, warmup_epochs=1,
                                       min_lr_ratio=cfg.MIN_LR_RATIO)

    if getattr(device, "type", str(device)) == "cpu":
        print("loaded optimizer")
    else:
        xm.rendezvous("loaded optimizer")
        xm.master_print(f"\n=== optimizer ===\n{pprint.pformat(optimizer)}\n")

    loss_fn = dict(
        loc_loss=IoULoss(encoded=True, anchors=data_encoder.anchor_boxes),
        cls_loss=FocalLoss(num_classes=len(cfg.CLASSES), alpha=0.35, gamma=1.5)
    )

    loss_weights = dict(
        loc_wt=cfg.LOC_WEIGHTAGE,
        cls_wt=cfg.CLS_WEIGHTAGE,
    )

    # 5. Initialize Custom Metric Class.
    mAP_metric = MeanAveragePrecision(class_metrics=True).to(device)

    # 6. Intialize Visualizer.
    tb_visualizer = TensorBoardVisualizer(logs_dir=cfg.log_dir)

    if getattr(device, "type", str(device)) == "cpu":
        print("training begins")
    else:
        xm.rendezvous("training begins")

    training_results = fit(
        model,
        epochs=cfg.EPOCHS,
        classes=cfg.CLASSES[1:],
        loader_train=train_loader,
        loader_test=val_loader,
        loss_fn=loss_fn,
        loss_weights=loss_weights,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        encoder=data_encoder,
        metric_fn=mAP_metric,
        nms_thresh=cfg.NMS_THRESHOLD,
        score_thresh=cfg.SCORE_THRESHOLD,
        device=device,
        checkpoint_dir=cfg.checkpoint_dir,
        visualizer=tb_visualizer
    )


def run(args):
    # Force single process on CPU debug (PJRT CPU doesn’t simulate multiple devices)
    if args.BACKEND.lower() == "cpu":
        args.WORLD_SIZE = 1
        print(f"Running in CPU mode, forcing world_size to {args.WORLD_SIZE}")
    else:
        args.WORLD_SIZE = xm.xrt_world_size()
        print(f"Running in {args.BACKEND} mode with world_size: {args.WORLD_SIZE}")

    if args.WORLD_SIZE <= 1:
        main_worker(0, args)
    else:
        xmp.spawn(main_worker, args=(args,), nprocs=args.WORLD_SIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/datasets/imagenet-1k")
    parser.add_argument("--fake_data", action="store_true", dest="fake_data")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt_dir", type=str, default="/tmp/vit_fsdp")
    parser.add_argument("--resume_epoch", type=int, default=0)
    parser.add_argument("--ckpt_epoch_interval", type=int, default=10)
    parser.add_argument("--test_epoch_interval", type=int, default=10)
    parser.add_argument("--log_step_interval", type=int, default=20)

    # the default model hyperparameters is a ViT with 10 billion parameters
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--embed_dim", type=int, default=5120)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=32)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--pos_dropout", type=float, default=0.0)
    parser.add_argument("--att_dropout", type=float, default=0.0)
    parser.add_argument("--mlp_dropout", type=float, default=0.0)
    parser.add_argument("--num_classes", type=int, default=1000)

    # these default learning hyperparameters are not necessarily optimal
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--no_grad_ckpt", action="store_false", dest="grad_ckpt")
    parser.add_argument("--no_reshard_after_forward", action="store_false", dest="reshard_after_forward")
    parser.add_argument("--flatten_parameters", action="store_true", dest="flatten_parameters")
    parser.add_argument("--run_without_fsdp", action="store_true", dest="run_without_fsdp")
    parser.add_argument("--shard_on_cpu", action="store_true", dest="shard_on_cpu")

    # cfg = parser.parse_args()
    cfg = Config()
    run(cfg)
