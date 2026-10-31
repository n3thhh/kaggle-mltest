"""Compile our Triton kernels while Soup loads the model, not in the first training step.

Every measured run starts with empty compiler caches, and the first launch of each kernel compiles
it: about 5 s of the first step on a Kaggle T4. `apply()` runs before Soup loads and quantizes the
model (about 15 s), so a thread started there compiles each kernel for exactly the argument types,
constexprs and specializations the training steps use (Triton's `warmup`, no launch). The first
training step waits for the thread. The time still counts: it just overlaps with model loading.
Small launches on zero tensors are used where a launch is cheap: they also load each compiled
module and PyTorch's attention kernels before the first step.
"""

import threading

_THREAD = None
TIMES = {}  # when the warm-up started / finished and how long the first step waited
VOCAB = 151936   # Qwen2.5's vocabulary (the head's rows)
HIDDEN = 1536    # Qwen2.5-1.5B's hidden size
INTER = 8960     # its MLP's intermediate size
HEADS, KV, HEAD_DIM = 12, 2, 128


def _compile_all():
    """Every kernel the training steps use, launched once on zero tensors with the exact
    argument types of the training steps (compiled and loaded)."""
    import torch

    from . import fused_ce, head_stream, layer, offload, rmsnorm

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    f16, f32, i64 = torch.float16, torch.float32, torch.int64
    i32, i8, u8 = torch.int32, torch.int8, torch.uint8

    def t(n, dtype):
        return torch.zeros(n, dtype=dtype, device=dev)

    H, I, QKV = HIDDEN, INTER, (HEADS + 2 * KV) * HEAD_DIM
    rows = t(2, i64)
    keep = torch.ones(2 * H, dtype=torch.bool, device=dev)

    def attention():
        # loads the memory-efficient attention kernels (no RNG: dropout is 0)
        q = t(16 * HEADS * HEAD_DIM, f16).view(1, 16, HEADS, HEAD_DIM)
        cu = torch.tensor([0, 7, 16], dtype=torch.int32, device=dev)
        out, lse, seed, off, _, _ = torch.ops.aten._efficient_attention_forward(
            q, q, q, None, cu, cu, 9, 9, 0.0, 1, True, scale=HEAD_DIM ** -0.5)
        torch.ops.aten._efficient_attention_backward(
            out, q, q, q, None, out, cu, cu, 9, 9, lse, 0.0, seed, off, 1, False,
            scale=HEAD_DIM ** -0.5)

    jobs = [
        # the fused decoder layer (layer.py), with the argument types of the training steps
        lambda: layer.nf4_dequant_multi_kernel[(1,)](
            t(1024, torch.uint8), t(32, f32), t(1024, torch.uint8), t(32, f32),
            t(1024, torch.uint8), t(32, f32), t(16, f32), t(2048, f16), 1152, 192, QBLOCK=64,
            BLOCK=layer.DEQUANT_BLOCK),
        lambda: layer.add_rmsnorm_fwd_kernel[(2,)](
            t(2 * H, f32), t(2 * H, f16), t(H, f32), t(2 * H, f16), t(2, f32), keep, keep, rows,
            t(2 * H // 4, torch.uint8), t(2 * H, f16), t(2 * H, f16), H, 1e-6, 1.0, 1.0, 1, 1,
            BLOCK=2048, num_warps=4),
        lambda: layer.rmsnorm_bwd_lora_kernel[(2,)](
            t(2 * H, f16), t(2 * H, f16), t(2 * H, f16), t(2 * H // 4, torch.uint8),
            t(2 * H, f32), t(2 * H, f16), t(H, f32), t(2, f32), t(2 * H, f32), t(2 * H, f32), H,
            1.0, 1.0, 1, 1, 1, BLOCK=2048, num_warps=4),
        lambda: layer.qkv_lora_rope_kernel[(1, HEADS)](
            t(QKV, f16), t(HEADS * HEAD_DIM, f16), t(KV * HEAD_DIM, f16), t(HEAD_DIM, f32),
            t(HEAD_DIM, f32), t(HEADS * HEAD_DIM, f16), t(HEADS * HEAD_DIM, f16),
            t(HEADS * HEAD_DIM, f16), t(KV * HEAD_DIM, f16), t(KV * HEAD_DIM, f16), 2.0, 2.0,
            N_HEADS=HEADS, N_KV=KV, HEAD_DIM=HEAD_DIM),
        lambda: layer.qkv_rope_bwd_kernel[(1, HEADS + KV)](
            t(HEADS * HEAD_DIM, f16), t(HEADS * HEAD_DIM, f16), t(HEADS * HEAD_DIM, f16),
            t(HEAD_DIM, f32), t(HEAD_DIM, f32), t(QKV, f16), N_HEADS=HEADS, N_KV=KV,
            HEAD_DIM=HEAD_DIM),
        lambda: layer.gate_up_act_fwd_kernel[(1, 1)](
            t(2 * I, f16), t(I, f16), I, BLOCK=layer.ACT_BLOCK),
        lambda: layer.gate_up_act_bwd_kernel[(1, 1)](
            t(I, f16), t(2 * I, f16), t(2 * I, f16), I, BLOCK=layer.ACT_BLOCK),
        # the int8 MLP GEMMs (layer.py): weights, activations, epilogues, and cuBLASLt's int8 GEMM
        lambda: None if not layer.INT8_FORWARD else layer.nf4_int8_rows_kernel[(1,)](
            t(H // 2, u8), t(H // 64, f32), t(H // 2, u8), t(H // 64, f32), t(16, f32),
            t(H, i8), t(1, f32), I, H, QBLOCK=64, BLOCK=2048),
        lambda: layer.nf4_int8_t_kernel[(1, 1)](
            t(64 * H // 2, u8), t(64 * H // 64, f32), t(64 * H // 2, u8), t(64 * H // 64, f32),
            t(16, f32), t(128, f32), t(128 * 1024, i8), 1024, 1024, H, QBLOCK=64, BN=64, BK=128),
        lambda: None if not layer.INT8_FORWARD else layer.row_quant_kernel[(1,)](
            t(H, f16), t(H, i8), t(1, f32), H, BLOCK=2048, num_warps=4),
        lambda: layer.row_quant_kernel[(1,)](t(H, f32), t(H, i8), t(1, f32), H, BLOCK=2048,
                                             num_warps=4),
        lambda: None if not layer.INT8_FORWARD else layer.gate_up_act_fwd_i8_kernel[(1, 1)](
            t(2 * I, i32), t(1, f32), t(2 * I, f32), t(2 * I, f16), t(I, f16), I,
            BLOCK=layer.ACT_BLOCK),
        lambda: layer.gate_up_act_bwd_i8_kernel[(1,)](
            t(I, i32), t(1, f32), t(I, f32), t(2 * I, f16), t(2 * I, i8), t(1, f32), INTER=I,
            BLOCK=layer.ACT_BLOCK),
        lambda: torch._int_mm(t(32 * 64, i8).view(32, 64), t(64 * 32, i8).view(32, 64).t()),
        # the head and loss (fused_ce.py) and the final norm (rmsnorm.py)
        lambda: None if offload.ENABLED and offload.HEAD else fused_ce.ce_row_stats_kernel.warmup(
            t(VOCAB, f16), VOCAB, 0, t(1, i64), t(1, f32), t(1, f32),
            t(1, torch.int32), fused_ce.GRAD_SCALE, N_VOCAB=VOCAB, BLOCK_V=fused_ce.VOCAB_BLOCK,
            WRITE_GRAD=True, num_warps=8, grid=(1,)),
        lambda: rmsnorm.rmsnorm_fwd_kernel.warmup(
            t(H, f32), t(H, f32), t(H, f32), t(H, f16), t(1, f32), H,
            1e-6, BLOCK=2048, HALF_OUT=True, num_warps=4, grid=(1,)),
        lambda: rmsnorm.rmsnorm_bwd_kernel.warmup(
            t(H, f32), t(H, f16), t(H, f32), t(H, f32), t(1, f32),
            t(H, f32), H, 1, 1, BLOCK=2048, num_warps=4, grid=(1,)),
        attention,
        # the head streamed from host memory (head_stream.py)
        lambda: head_stream.ce_chunk_stats_kernel[(1,)](
            t(head_stream.CHUNK, f16), head_stream.CHUNK, head_stream.CHUNK, 0, t(1, i64),
            t(1, f32), t(1, f32), t(1, f32), t(1, f32), t(1, i64), t(1, f32), t(1, f32), 1, 1,
            MAX_COLS=head_stream.CHUNK, BLOCK_V=head_stream.STAT_BLOCK, WRITE_E=True,
            num_warps=8),
        lambda: head_stream.ce_finalize_kernel[(1,)](
            t(1, i64), t(1, f32), t(1, f32), t(1, f32), t(1, i64), t(1, f32), t(1, f32),
            t(1, f32), t(1, torch.int32), t(1, f32), 1, BLOCK=1024),
    ]
    def host_memory():
        # Pinned host memory, allocated here while the main thread imports libraries: the
        # blocks for the 4-bit weights (about 0.75 GB, filled during loading), the tied
        # embedding (467 MB) and the saved activations of a 1,536-token step (grown at a step
        # that needs more).
        if not offload.ENABLED:
            return
        offload.WEIGHTS.reserve(3)
        if offload.HEAD:
            head_stream.reserve((VOCAB, HIDDEN), torch.float16)
        store = offload.activation_store(dev)
        if store is not None:
            store.reserve(layer.host_bytes(1536, 28, INTER) + 4 * INTER * 1536)

    jobs.insert(0, host_memory)
    import time

    TIMES["start"] = time.perf_counter()
    _count_cache_hits()
    for i, job in enumerate(jobs):
        began = time.perf_counter()
        try:
            job()
        except Exception as exc:  # noqa: BLE001 - a kernel not warmed up compiles at first use
            TIMES.setdefault("errors", []).append(repr(exc)[:200])
        TIMES.setdefault("jobs", []).append((round(time.perf_counter() - began, 2), i))
    TIMES["done"] = time.perf_counter()


def _count_cache_hits():
    """Count Triton's compile-cache hits and misses in this process (for the log)."""
    try:
        from triton import knobs

        previous = knobs.compilation.listener
        counts = TIMES.setdefault("cache", {"hit": 0, "miss": 0})

        def listener(*args, **kwargs):
            counts["hit" if kwargs.get("cache_hit") else "miss"] += 1
            if previous is not None:
                previous(*args, **kwargs)

        knobs.compilation.listener = listener
    except Exception:  # noqa: BLE001
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
    if _THREAD is not None and "waited" not in TIMES:
        import time

        t0 = time.perf_counter()
        _THREAD.join()
        TIMES["waited"] = time.perf_counter() - t0
        import sys

        start, done = TIMES.get("start"), TIMES.get("done")
        print(f"[solution] kernel warm-up: {done - start:.2f} s in the background,"
              f" the first step waited {TIMES['waited']:.2f} s, compile cache {TIMES.get('cache')},"
              f" slowest jobs {sorted(TIMES.get('jobs', []), reverse=True)[:6]},"
              f" errors: {TIMES.get('errors', [])}"
              if start is not None and done is not None else
              f"[solution] kernel warm-up: unfinished ({TIMES})", file=sys.stderr)
