# probe_spawn.py
import os, time, contextlib

# --- Optional: quieten logs a bit ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

# --- Imports (CPU fallback is fine) ---
import torch

# These imports must succeed to use TPU/XLA
try:
    import torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    from torch_xla.debug import metrics as met
    IS_XLA = True
except Exception as e:
    print(f"[parent] torch_xla not available: {e}", flush=True)
    IS_XLA = False


def _parent_banner():
    """Runs in the parent (before spawn) to show what PJRT *could* see."""
    if not IS_XLA:
        print("[parent] XLA not available; will run single-process on CPU.", flush=True)
        return 1
    # Honor TPU_NUM_DEVICES if set; else count supported devices
    with contextlib.suppress(Exception):
        devs = xm.get_xla_supported_devices()
    devs = devs if devs else []
    world = int(os.environ.get("TPU_NUM_DEVICES", "0")) or (len(devs) if devs else 1)
    print(f"[parent] PJRT_DEVICE={xr.device_type()} supports={devs} planned_world={world}", flush=True)
    return world


def _child_worker(rank: int, world: int):
    """Each spawned process lands here."""
    if not IS_XLA:
        dev = torch.device("cpu")
        print(f"[child] rank={rank}/{world} device={dev} (CPU mode)", flush=True)
        return

    # Bind to this process's XLA device (don’t try to index devices yourself)
    dev = xm.xla_device()

    # Local ordinal is the per-host device index this process is bound to
    with contextlib.suppress(Exception):
        local_ord = xm.get_local_ordinal()

    print(
        f"[child] rank={rank}/{world} local_ordinal={local_ord} default_device={dev} "
        f"PJRT_DEVICE={xr.device_type()}",
        flush=True,
    )

    # Barrier so you can see all children reached here
    xm.rendezvous("after_setup_prints")

    # --- Tiny “compile” smoke: first step triggers XLA compile once ---
    if xm.is_master_ordinal():
        print("[child] master: running a tiny XLA op to trigger compile…", flush=True)

    t0 = time.perf_counter()
    x = torch.randn(4, 4, device=dev)
    y = torch.matmul(x, x) + 1.0
    loss = y.sum()
    loss.backward()
    # Mark the step so XLA sends work to the device
    xm.mark_step()
    dt = time.perf_counter() - t0

    print(f"[child] rank={rank} first_step_compile_dt={dt:.2f}s", flush=True)

    # Optional: short XLA metrics snapshot (master only)
    with contextlib.suppress(Exception):
        if xm.is_master_ordinal():
            print(met.metrics_report(fmt="text", shorten=True), flush=True)

    # All done
    xm.rendezvous("done")


def main():
    world = _parent_banner()

    if not IS_XLA:
        _child_worker(rank=0, world=1)
        return

    # Under PJRT, you MUST use nprocs=None. Limit via TPU_NUM_DEVICES (already used above).
    xmp.spawn(_child_worker, args=(world,), nprocs=None)


if __name__ == "__main__":
    main()
