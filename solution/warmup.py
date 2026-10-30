"""Compile our Triton kernels while Soup loads the model, not in the first training step.

Every measured run starts with empty compiler caches, and the first launch of each kernel compiles
it: about 5 s of the first step on a Kaggle T4. `apply()` runs before Soup loads and quantizes the
model (about 15 s), so a thread started there compiles each kernel for exactly the argument types,
constexprs and specializations the training steps use (Triton's `warmup`, no launch). The first
training step waits for the thread. The time still counts: it just overlaps with model loading.
The organizers' example kernels, which run in the same steps, are compiled the same way.
"""

import threading

_THREAD = None
VOCAB = 151936   # Qwen2.5's vocabulary (the head's rows)
HIDDEN = 1536    # Qwen2.5-1.5B's hidden size


def _compile_all():
    import torch

    from . import fused_ce, nf4, rmsnorm, swiglu, unpad

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    f16, f32 = torch.float16, torch.float32

    def t(n, dtype):
        return torch.zeros(n, dtype=dtype, device=dev)

    jobs = [
        lambda: nf4.nf4_dequant_kernel.warmup(
            t(256, torch.uint8), t(8, f32), t(16, f32), t(512, f16), 512, QBLOCK=64,
            BLOCK=nf4.DEQUANT_BLOCK, grid=(1,)),
        lambda: fused_ce.ce_row_stats_kernel.warmup(
            t(VOCAB, f16), VOCAB, 0, t(1, torch.int64), t(1, f32), t(1, f32),
            t(1, torch.int32), fused_ce.GRAD_SCALE, N_VOCAB=VOCAB, BLOCK_V=fused_ce.VOCAB_BLOCK,
            WRITE_GRAD=True, num_warps=8, grid=(1,)),
        lambda: rmsnorm.rmsnorm_fwd_kernel.warmup(
            t(HIDDEN, f32), t(HIDDEN, f32), t(HIDDEN, f32), t(HIDDEN, f16), t(1, f32), HIDDEN,
            1e-6, BLOCK=2048, HALF_OUT=True, num_warps=4, grid=(1,)),
        lambda: rmsnorm.rmsnorm_bwd_kernel.warmup(
            t(HIDDEN, f32), t(HIDDEN, f16), t(HIDDEN, f32), t(HIDDEN, f32), t(1, f32),
            t(HIDDEN, f32), HIDDEN, 1, 1, BLOCK=2048, num_warps=4, grid=(1,)),
        lambda: unpad.rows_masked_scale_kernel.warmup(
            t(HIDDEN, f32), t(HIDDEN, torch.bool), t(1, torch.int64), t(HIDDEN, f32), HIDDEN,
            1.0 / 0.95, BLOCK=unpad.ROW_BLOCK, grid=(1, 2)),
        lambda: swiglu.example_swiglu_fwd_kernel.warmup(
            t(8960, f16), t(8960, f16), t(8960, f16), 8960, BLOCK=swiglu.BLOCK, grid=(1,)),
        lambda: swiglu.example_swiglu_bwd_kernel.warmup(
            t(8960, f16), t(8960, f16), t(8960, f16), t(8960, f16), t(8960, f16), 8960,
            BLOCK=swiglu.BLOCK, grid=(1,)),
    ]
    for job in jobs:
        try:
            job()
        except Exception:  # noqa: BLE001 - a kernel not warmed up just compiles at first use
            pass


def start():
    global _THREAD
    try:
        import torch

        if not torch.cuda.is_available() or _THREAD is not None:
            return
    except Exception:  # noqa: BLE001
        return
    _THREAD = threading.Thread(target=_compile_all, name="soup-kernel-warmup", daemon=True)
    _THREAD.start()


def wait():
    if _THREAD is not None:
        _THREAD.join()
