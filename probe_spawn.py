# probe_spawn_v3.py
import os, time, contextlib
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

try:
    import torch_xla.runtime as xr
    XLA_OK = True
except Exception as e:
    print(f"[parent] torch_xla.runtime not available: {e}", flush=True)
    XLA_OK = False


def child_worker(rank: int, world: int):
    import torch
    import torch_xla.core.xla_model as xm
    from torch_xla.debug import metrics as met

    dev = xm.xla_device()
    with contextlib.suppress(Exception):
        local_ord = xm.get_local_ordinal()

    print(f"[child] rank={rank}/{world} local_ordinal={local_ord} default_device={dev} PJRT_DEVICE={xr.device_type()}",
          flush=True)
    xm.rendezvous("after_setup")

    # Trigger a tiny compile (now with grad)
    x = torch.randn(4, 4, device=dev, requires_grad=True)
    t0 = time.perf_counter()
    y = (x @ x) + 1.0
    y.sum().backward()
    xm.mark_step()
    dt = time.perf_counter() - t0
    print(f"[child] rank={rank} first_step_compile_dt={dt:.2f}s", flush=True)

    if xm.is_master_ordinal():
        print(met.metrics_report(fmt="text", shorten=True), flush=True)

    xm.rendezvous("done")


def main():
    if not XLA_OK:
        print("[parent] XLA not available; exiting.", flush=True)
        return

    world = int(os.environ.get("TPU_NUM_DEVICES", "0")) or 8
    print(f"[parent] PJRT_DEVICE={xr.device_type()} planned_world={world}", flush=True)

    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(child_worker, args=(world,), nprocs=None)


if __name__ == "__main__":
    main()
