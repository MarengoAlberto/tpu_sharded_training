import os, sys, time

# Keep logs tidy
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

def child(rank: int, world: int):
    import torch
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    try:
        local_ord = xm.get_local_ordinal()
    except Exception:
        local_ord = -1

    print(f"[child] rank={rank}/{world} local_ordinal={local_ord} device={dev}", flush=True)

    # Trigger a tiny compile & a real device step
    x = torch.randn(4, 4, device=dev, requires_grad=True)
    y = (x @ x).sum()
    y.backward()
    xm.mark_step()
    print(f"[child] rank={rank} first_step_done", flush=True)

    # Make sure all device work is flushed before exit
    try:
        xm.wait_device_ops()
    except Exception:
        pass

    # No barriers, no extra threads; exit promptly
    sys.stdout.flush()
    return  # child returns; parent join will complete

def main():
    try:
        import torch_xla.runtime as xr
        import torch_xla.distributed.xla_multiprocessing as xmp
    except Exception as e:
        print(f"[parent] XLA not available: {e}", flush=True)
        sys.exit(1)

    world = int(os.environ.get("TPU_NUM_DEVICES", "0")) or 8
    print(f"[parent] PJRT_DEVICE={xr.device_type()} planned_world={world}", flush=True)

    # Important: nprocs=None, limit via TPU_NUM_DEVICES
    xmp.spawn(child, args=(world,), nprocs=None)

if __name__ == "__main__":
    main()
