import os
import torch.nn as nn
from functools import partial


try:
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
            fsdp_kwargs: dict | None = None,
    ) -> nn.Module:
        fsdp_kwargs = fsdp_kwargs or {}

        # Manually wrap major chunks
        if hasattr(model, "backbone"):
            model.backbone = fsdp_wrap(grad_ckpt_wrap(model.backbone), **fsdp_kwargs)
        if hasattr(model, "fpn"):
            model.fpn = fsdp_wrap(grad_ckpt_wrap(model.fpn), **fsdp_kwargs)
        if hasattr(model, "embedding_1"):
            model.embedding_1 = fsdp_wrap(grad_ckpt_wrap(model.embedding_1), **fsdp_kwargs)
        if hasattr(model, "embedding_2"):
            model.embedding_2 = fsdp_wrap(grad_ckpt_wrap(model.embedding_2), **fsdp_kwargs)
        if hasattr(model, "loc_head"):
            model.loc_head = fsdp_wrap(grad_ckpt_wrap(model.loc_head), **fsdp_kwargs)
        if hasattr(model, "cls_head"):
            model.cls_head = fsdp_wrap(grad_ckpt_wrap(model.cls_head), **fsdp_kwargs)

        # Root wrap WITHOUT auto_wrap_policy to avoid double wrapping
        model = fsdp_wrap(model, **fsdp_kwargs)
        return model


    def save_fsdp_model(model: nn.Module, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f"model-rank{xm.get_ordinal():02d}_best.pt")
        xm.save({"model": model.state_dict()}, path, master_only=False)

except Exception:
    print("Failed to import torch_xla. Please ensure that torch_xla is installed.")
