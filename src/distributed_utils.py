import os
import torch.nn as nn
from functools import partial
import torch_xla.core.xla_model as xm
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
from torch_xla.distributed.fsdp import checkpoint_module
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy

def grad_ckpt_wrap(module: nn.Module) -> nn.Module:
    # Rematerialize activations within the module
    return checkpoint_module(module)

def fsdp_wrap(module: nn.Module, **kwargs) -> FSDP:
    return FSDP(module, **kwargs)

def apply_fsdp_with_ckpt_detector(
    model: nn.Module,
    use_conv_auto_wrap: bool = True,
    min_params_for_wrap: int = 100_000_000,
    fsdp_kwargs: dict | None = None,
) -> nn.Module:
    """
    Expected model structure: model.backbone, model.fpn, model.emb1, model.emb2, model.head1, model.head2
    You can rename these to match your actual attribute names.
    """
    fsdp_kwargs = fsdp_kwargs or {}

    # Choose an auto_wrap_policy:
    # - For conv-heavy detectors, telling XLA’s transformer_auto_wrap_policy to treat Conv2d as "transformer-like"
    #   is a handy way to auto-wrap every Conv2d-heavy block.
    if use_conv_auto_wrap:
        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={nn.Conv2d})
    else:
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=min_params_for_wrap)

    # Manually wrap the major chunks (and apply grad ckpt) — this mirrors the tutorial’s pattern
    if hasattr(model, "backbone"):
        model.backbone = fsdp_wrap(grad_ckpt_wrap(model.backbone), **fsdp_kwargs)
    if hasattr(model, "fpn"):
        model.fpn = fsdp_wrap(grad_ckpt_wrap(model.fpn), **fsdp_kwargs)
    if hasattr(model, "embedding_1"):
        model.emb1 = fsdp_wrap(grad_ckpt_wrap(model.emb1), **fsdp_kwargs)
    if hasattr(model, "embedding_2"):
        model.emb2 = fsdp_wrap(grad_ckpt_wrap(model.emb2), **fsdp_kwargs)
    if hasattr(model, "loc_head"):
        model.head1 = fsdp_wrap(grad_ckpt_wrap(model.head1), **fsdp_kwargs)
    if hasattr(model, "cls_head"):
        model.head2 = fsdp_wrap(grad_ckpt_wrap(model.head2), **fsdp_kwargs)

    # Finally, outer wrap the full model. You can also pass auto_wrap_policy so inner modules get auto-wrapped too.
    model = fsdp_wrap(
        model,
        auto_wrap_policy=auto_wrap_policy,
        **fsdp_kwargs,
    )
    return model


def save_fsdp_model(model: nn.Module, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f"model-rank{xm.get_ordinal():02d}_best.pt")
    xm.save({"model": model.state_dict()}, path, master_only=False)
