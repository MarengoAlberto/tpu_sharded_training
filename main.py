import argparse
import pprint
import math
import torch
import torch.optim as optim
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
from torchinfo import summary
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from config import Config
from src.data import PlateDataset, DataEncoder
from src.transform import get_augmentations
from src.utils import set_seeds
from src.detector import Detector
from src.logging import setup_log_directory
from src.train_utils import build_cosine_warmup, make_param_groups
from src.train import fit
from src.loss import OHEMLoss, SmoothL1Loss, FocalLoss, IoULoss
from src.tensorboard import TensorBoardVisualizer


def fsdp_wrap(module, cfg, device):
    if cfg.run_without_fsdp:
        return module.to(device)
    # note: to implement ZeRO-3, set `cfg.reshard_after_forward` to True
    # FSDP can directly wrap a module on CPU in https://github.com/pytorch/xla/pull/3992
    # so one doesn't need to cast the module into XLA devices first.
    return FSDP(
        module if cfg.shard_on_cpu else module.to(device),
        reshard_after_forward=cfg.reshard_after_forward,
        flatten_parameters=cfg.flatten_parameters,
    )


def build_datasets(cfg, device):
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()

    assert cfg.batch_size % world_size == 0
    local_batch_size = cfg.batch_size // world_size

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
    )
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
    )
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

    model = fsdp_wrap(model, cfg, device)

    xm.master_print(summary(model, input_size=(1,) + cfg.IMG_SIZE[::-1], row_settings=["var_names"]))

    return model


def train(cfg):
    set_seeds()
    batch_size = cfg.batch_size
    num_epochs = cfg.num_epochs
    device = xm.xla_device()
    rank = xm.get_local_ordinal()

    # build datasets
    data_encoder = DataEncoder(input_size=cfg.IMG_SIZE[:2], classes=cfg.CLASSES)
    train_dataset, train_loader, train_sampler, _, val_loader, _ = build_datasets(cfg, device)
    xm.rendezvous("loaded dataset")
    xm.master_print(f"\n=== dataset ===\n{pprint.pformat(train_dataset)}\n")

    # build model and loss
    model = build_od_model(cfg, device)
    loss_fn = torch.nn.CrossEntropyLoss()
    xm.rendezvous("loaded model")
    xm.master_print(f"\n=== model ===\n{pprint.pformat(model)}\n")

    parameters = list(model.parameters())
    xm.master_print(f"per-TPU (sharded) parameter num: {sum(p.numel() for p in parameters)}")

    # build optimizer and scheduler
    optimizer = optim.AdamW(
        make_param_groups(model, wd=cfg.weight_decay),
        lr=cfg.INIT_LEARING_RATE, betas=(0.9, 0.999), eps=1e-8
    )

    lr_scheduler = build_cosine_warmup(optimizer, total_epochs=cfg.EPOCHS, warmup_epochs=1,
                                       min_lr_ratio=cfg.MIN_LR_RATIO)

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

    xm.rendezvous("loaded optimizer")
    xm.master_print(f"\n=== optimizer ===\n{pprint.pformat(optimizer)}\n")

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


def main(device_id, cfg):
    cfg, current_version_name = setup_log_directory(cfg)
    xm.master_print(f"\n=== cfg ===\n{pprint.pformat(cfg)}\n")
    train(cfg)
    xm.master_print("training completed")


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
    xmp.spawn(main, args=(cfg,))