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
- `offload.py`   4-bit weights and saved activations in pinned host memory, streamed per layer
- `head_stream.py` the tied embedding / head in pinned host memory: loss over vocabulary chunks
- `swiglu.py`    the organizers' example (fused SwiGLU), kept running; it does not count as ours
- `test_kernels.py`  every kernel against plain PyTorch (GPU): `python -m pytest solution -q`
"""

import sys

from .swiglu import patch_qwen2_mlp


def _try(name, patch):
    try:
        patch()
    except Exception as exc:  # noqa: BLE001 - a missing piece must not stop the run
        print(f"[solution] {name} not applied: {exc.__class__.__name__}: {exc}", file=sys.stderr)


def apply():
    from .imports import import_trl

    _try("one package-metadata scan", import_trl)  # before anything of ours imports TRL
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
    from .warmup import start

    _try("fp16 tied embedding", patch_embedding)
    _try("tied embedding kept fp16 at load", patch_prepare_kbit)
    _try("kernel warm-up", start)  # compiles our kernels while Soup loads the model
    from . import offload

    if offload.ENABLED:  # before the model is loaded: weights go to host memory as quantized
        _try("4-bit weights in pinned host memory", offload.patch_quantize)
        _try("loading without GPU pre-allocation, embedding to host memory",
             offload.patch_loading)
