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


def _compile_all(offline=False, part=0, parts=1):
    """Every kernel the training steps use, for their exact argument types. In the process
    itself the kernels are launched once on zero tensors (compiled, or read from the compile
    cache, and loaded). `offline`: in a helper process without a GPU, compile only (into the
    shared cache directory), the share `part` of `parts`."""
    import torch

    from . import fused_ce, head_stream, layer, offload, rmsnorm

    if offline:
        dev = torch.device("cpu")
    else:
        dev = torch.device("cuda", 0)
        torch.cuda.set_device(dev)
    f16, f32, i64 = torch.float16, torch.float32, torch.int64

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
            t(2 * H, f32), t(2 * H, f32), t(H, f32), t(2 * H, f32), t(2 * H, f16), t(2, f32), H,
            1e-6, BLOCK=2048, HAS_D=False, WRITE_Y=True, num_warps=4),
        lambda: layer.add_rmsnorm_fwd_kernel[(2,)](
            t(2 * H, f32), t(2 * H, f16), t(H, f32), t(2 * H, f16), t(2 * H, f16), t(2, f32), H,
            1e-6, BLOCK=2048, HAS_D=True, WRITE_Y=False, num_warps=4),
        lambda: layer.rmsnorm_bwd_lora_kernel[(2,)](
            t(2 * H, f16), t(2 * H, f32), t(2 * H, f32), rows, rows, rows, t(2 * H, f32),
            t(2 * H, f16), t(H, f32), t(2, f32), t(2 * H, f32), t(2 * H, f32), H, 1.0, 1.0, 1, 0,
            BLOCK=2048, HAS_D=True, num_warps=4),
        lambda: layer.rmsnorm_bwd_lora_kernel[(2,)](
            t(2 * H, f16), t(2 * H, f16), t(2 * H, f16), keep, keep, rows, t(2 * H, f32),
            t(2 * H, f32), t(H, f32), t(2, f32), t(2 * H, f32), t(2 * H, f32), H, 1.0, 1.0, 1, 1,
            BLOCK=2048, HAS_D=False, num_warps=4),
        lambda: layer.lora_dropout_pair_kernel[(2, 2)](
            t(2 * H, f32), keep, keep, rows, t(2 * H, f16), t(2 * H, f16), H, 1.0, 1.0,
            BLOCK=layer.ROW_BLOCK),
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
            t(1, f32), t(1, f32), t(1, f32), t(1, f32), t(1, i64), t(1, f32), t(1, f32), 1,
            MAX_COLS=head_stream.CHUNK, BLOCK_V=head_stream.STAT_BLOCK, FIRST=True, WRITE_E=True,
            num_warps=8),
        lambda: head_stream.ce_chunk_stats_kernel[(1,)](
            t(head_stream.CHUNK, f16), head_stream.CHUNK, head_stream.CHUNK, 0, t(1, i64),
            t(1, f32), t(1, f32), t(1, f32), t(1, f32), t(1, i64), t(1, f32), t(1, f32), 1,
            MAX_COLS=head_stream.CHUNK, BLOCK_V=head_stream.STAT_BLOCK, FIRST=False,
            WRITE_E=True, num_warps=8),
        lambda: head_stream.ce_finalize_kernel[(1,)](
            t(1, i64), t(1, f32), t(1, f32), t(1, f32), t(1, i64), t(1, f32), t(1, f32),
            t(1, f32), t(1, torch.int32), t(1, f32), 1, BLOCK=1024),
        lambda: head_stream.rescale_add_kernel[(1,)](
            t(H, f32), t(H, f16), t(1, f32), HIDDEN=H, BLOCK_H=2048),
        lambda: head_stream.target_rows_kernel[(1,)](
            t(H, f16), 1, 0, t(1, i64), t(H, f16), HIDDEN=H, BLOCK_H=2048),
        lambda: head_stream.ce_grad_finalize_kernel[(1,)](
            t(H, f32), t(H, f16), t(1, f32), t(1, f32), t(1, f32), 1, 4096.0, HIDDEN=H,
            BLOCK_H=2048),
    ]
    def host_memory():
        # pinned host memory for saved activations, sized for the longest batch (4 x 512)
        from . import offload

        store = offload.activation_store(dev)
        if store is not None and store.arena is None:
            store.arena = torch.empty(layer.host_bytes(2048, 28, INTER), dtype=torch.uint8,
                                      pin_memory=True)

    jobs.append(host_memory)
    import time

    if offline:
        jobs = [job for i, job in enumerate(jobs)
                if job not in (attention, host_memory) and i % parts == part]
    else:
        TIMES["start"] = time.perf_counter()
        _wait_helpers()
        TIMES["helpers_done"] = time.perf_counter()
    for job in jobs:
        try:
            job()
        except Exception as exc:  # noqa: BLE001 - a kernel not warmed up compiles at first use
            if not offline:
                TIMES.setdefault("errors", []).append(repr(exc)[:200])
    if not offline:
        TIMES["done"] = time.perf_counter()


# --- compiling in helper processes --------------------------------------------------------------
#
# Triton's compiler runs partly in Python: compiling in a thread of the training process holds the
# GIL for seconds while that process imports libraries and loads the model. So two helper processes
# (the T4 machines have 4 CPUs) compile the kernels into Triton's cache directory of this run
# (TRITON_CACHE_DIR, inherited), for the T4's target (sm_75) without touching the GPU; the thread
# here then only reads the compiled kernels from the cache and loads them. Should the helpers fail
# or produce other cache keys, the thread compiles as before.

HELPERS = []
HELPER_PARTS = 2
_HELPER_CODE = r"""
import sys
sys.path.insert(0, sys.argv[1])
from triton.backends.compiler import GPUTarget
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction


class _Target:  # the T4, without initialising CUDA
    def get_current_device(self):
        return 0

    def get_current_stream(self, device=None):
        return 0

    def get_current_target(self):
        return GPUTarget("cuda", 75, 32)


driver.set_active(_Target())
_run = JITFunction.run


def _compile_only(self, *args, grid, warmup, **kwargs):
    return _run(self, *args, grid=grid, warmup=True, **kwargs)


JITFunction.run = _compile_only
from solution import warmup

warmup._compile_all(offline=True, part=int(sys.argv[2]), parts=int(sys.argv[3]))
"""


def _start_helpers():
    import os
    import subprocess
    import sys

    if os.environ.get("SOUP_SOLUTION_COMPILE_HELPERS", "1") != "1":
        return
    try:
        import torch

        major, minor = torch.cuda.get_device_capability(0)
        if (major, minor) != (7, 5):
            return  # the helpers compile for the T4 only
    except Exception:  # noqa: BLE001
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for part in range(HELPER_PARTS):
        try:
            HELPERS.append(subprocess.Popen(
                [sys.executable, "-c", _HELPER_CODE, root, str(part), str(HELPER_PARTS)],
                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        except Exception:  # noqa: BLE001
            pass


def _wait_helpers():
    for proc in HELPERS:
        try:
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001 - too slow: the thread compiles what is missing
            proc.kill()
    TIMES["helpers"] = [proc.returncode for proc in HELPERS]


def start():
    global _THREAD
    try:
        import torch

        if not torch.cuda.is_available() or _THREAD is not None:
            return
    except Exception:  # noqa: BLE001
        return
    _start_helpers()
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
        print(f"[solution] kernel warm-up: {done - start:.2f} s in the background (helper "
              f"processes {TIMES.get('helpers')}: {TIMES.get('helpers_done', start) - start:.2f} s),"
              f" the first step waited {TIMES['waited']:.2f} s, errors: {TIMES.get('errors', [])}"
              if start is not None and done is not None else
              f"[solution] kernel warm-up: unfinished ({TIMES})", file=sys.stderr)
