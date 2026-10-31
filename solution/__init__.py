"""Our solution: Triton kernels for Soup's Qwen2.5-1.5B QLoRA run on a Kaggle T4.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer.

- `fused_ce.py`  the output head and the loss in chunks, no [batch, seq, vocab] logits (memory)
- `nf4.py`       NF4 dequantization in one kernel, straight to fp16, for every 4-bit layer
- `rmsnorm.py`   RMSNorm forward and backward in one kernel each, with the fp16 copy those
                 layers need
- `layer.py`     a whole decoder layer as one autograd function: fused q|k|v and gate|up GEMMs,
                 LoRA inside, variable-length attention, hand-written backward pass
- `unpad.py`     the decoder on real tokens only (padding removed inside the batch)
- `embed.py`     one exact fp16 copy of the tied embedding / head instead of fp32 + fp16
- `warmup.py`    our kernels compile while the model loads, not in the first step
- `rope.py`      rotary embedding + fp16 cast + k/v expansion for attention, one kernel each way
- `mlp.py`       in long batches the MLP keeps only its fp16 input and recomputes gate/up
- `imports.py`   TRL reuses transformers' package-metadata scan instead of a second one
- `optional_imports.py` scikit-learn and wandb (unused here) not imported by transformers / TRL
- `dataset_name.py` Soup's tokenization map named at random instead of by hashing the tokenizer
- `offload.py`   4-bit weights and saved activations in pinned host memory, streamed per layer
- `head_stream.py` the tied embedding / head in pinned host memory: loss over vocabulary chunks
- `swiglu.py`    the organizers' example (fused SwiGLU), kept running; it does not count as ours
- `test_kernels.py`  every kernel against plain PyTorch (GPU): `python -m pytest solution -q`
"""

import sys
import time

from .swiglu import patch_qwen2_mlp


_T0 = time.perf_counter()  # when solution/ was imported (the harness imports it right before apply())


def since_start():
    """Seconds since solution/ was imported (for the timeline lines in the log)."""
    return time.perf_counter() - _T0


def mark(what):
    print(f"[solution] t+{since_start():6.2f} s since import  {what}", file=sys.stderr,
          flush=True)


def _try(name, patch):
    try:
        patch()
    except Exception as exc:  # noqa: BLE001 - a missing piece must not stop the run
        print(f"[solution] {name} not applied: {exc.__class__.__name__}: {exc}", file=sys.stderr)


def apply():
    mark("apply() starts")
    from .warmup import start

    _try("kernel warm-up", start)  # first: our kernels compile while everything else loads
    from .optional_imports import (import_candidate_generator_without_sklearn,
                                   import_trl_base_trainer_without_wandb)

    _try("assisted-generation helpers without scikit-learn",
         import_candidate_generator_without_sklearn)  # before anything imports generation
    from .imports import import_trl

    _try("one package-metadata scan", import_trl)  # before anything of ours imports TRL
    _try("TRL's base trainer without wandb", import_trl_base_trainer_without_wandb)
    from .dataset_name import patch_soup_text_map

    _try("Soup's tokenization map without hashing the tokenizer", patch_soup_text_map)
    from .fused_ce import patch_qwen2_causal_lm, patch_sft_trainer
    from .nf4 import patch_bnb_matmul_4bit, patch_linear4bit
    from .rmsnorm import patch_qwen2_rmsnorm
    from .unpad import patch_dropout, patch_qwen2_attention, patch_qwen2_model

    _try("example SwiGLU", patch_qwen2_mlp)
    from .mlp import patch_qwen2_mlp_recompute

    _try("MLP recomputing its activations in long batches", patch_qwen2_mlp_recompute)
    _try("NF4 dequantization", patch_bnb_matmul_4bit)
    _try("4-bit linear layers in fp16", patch_linear4bit)
    _try("RMSNorm", patch_qwen2_rmsnorm)
    _try("fused head and loss (model)", patch_qwen2_causal_lm)
    _try("fused head and loss (trainer)", patch_sft_trainer)
    _try("padding-free decoder", patch_qwen2_model)
    _try("padding-free attention", patch_qwen2_attention)
    _try("LoRA dropout with the base masks", patch_dropout)
    from .embed import patch_embedding, patch_prepare_kbit

    _try("fp16 tied embedding", patch_embedding)
    _try("tied embedding kept fp16 at load", patch_prepare_kbit)
    from . import offload

    if offload.ENABLED:  # before the model is loaded: weights go to host memory as quantized
        _try("4-bit weights in pinned host memory", offload.patch_quantize)
        _try("loading without GPU pre-allocation, embedding to host memory",
             offload.patch_loading)
    _try("checkpoint tensors converted to fp16 on the GPU", offload.patch_materialize)
    _try("timeline", _timeline)
    mark("apply() done")


def _timeline():
    """Marks in the log at the main stages of the run (load, trainer, training, exit)."""
    import atexit

    import transformers

    atexit.register(mark, "exit")
    trainer_cls = transformers.Trainer
    init, train = trainer_cls.__init__, trainer_cls.train

    def traced_init(self, *args, **kwargs):
        mark("trainer: init starts")
        init(self, *args, **kwargs)
        try:
            self.add_callback(_step_clock())
        except Exception:  # noqa: BLE001
            pass
        mark("trainer: init done")

    def traced_train(self, *args, **kwargs):
        mark("trainer: train() starts")
        try:
            import torch

            from .layer import MEMORY

            MEMORY["before training (max)"] = (torch.cuda.max_memory_allocated() / 2**20,
                                               torch.cuda.max_memory_reserved() / 2**20)
        except Exception:  # noqa: BLE001
            pass
        try:
            return train(self, *args, **kwargs)
        finally:
            mark("trainer: train() done")
            _print_step_clock()
            try:
                from .layer import MEMORY
                from .unpad import _MAX_TOKENS

                print(f"[solution] longest step {_MAX_TOKENS[0]} tokens; GPU memory per phase "
                      "(MiB allocated / reserved): " + ", ".join(
                          f"{k} {a:.0f}/{r:.0f}" for k, (a, r) in MEMORY.items()),
                      file=sys.stderr)
            except Exception:  # noqa: BLE001
                pass

    trainer_cls.__init__, trainer_cls.train = traced_init, traced_train


# --- where the wall time of a training step goes (CPU clock only: no synchronisation added) -----

CLOCK = {"marks": [], "last": None}


def clock(what):
    CLOCK["last"] = CLOCK["last"] or {}
    CLOCK["last"][what] = time.perf_counter()


def _step_clock():
    import transformers

    class StepClock(transformers.TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            now = time.perf_counter()
            if CLOCK["last"] is not None and "step_end" in CLOCK["last"]:
                CLOCK["last"]["next_begin"] = now
                CLOCK["marks"].append(CLOCK["last"])
            CLOCK["last"] = {"step_begin": now}

        def on_step_end(self, args, state, control, **kwargs):
            clock("step_end")

    return StepClock()


def _print_step_clock():
    marks = [m for m in CLOCK["marks"][1:] if all(
        k in m for k in ("step_begin", "loss_begin", "loss_end", "step_end", "next_begin"))]
    if not marks:
        return

    def avg(a, b):
        return 1000 * sum(m[b] - m[a] for m in marks) / len(marks)

    print(f"[solution] step clock over {len(marks)} steps (ms): begin->forward "
          f"{avg('step_begin', 'loss_begin'):.1f}, forward+loss {avg('loss_begin', 'loss_end'):.1f},"
          f" backward+optimizer {avg('loss_end', 'step_end'):.1f}, end->next begin "
          f"{avg('step_end', 'next_begin'):.1f}", file=sys.stderr)
