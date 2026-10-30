"""Our solution: Triton kernels for Soup's Qwen2.5-1.5B QLoRA run on a Kaggle T4.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer.

- `fused_ce.py`  the output head and the loss in chunks, no [batch, seq, vocab] logits (memory)
- `nf4.py`       NF4 dequantization in one kernel, straight to fp16, for every 4-bit layer
- `rmsnorm.py`   RMSNorm forward and backward in one kernel each, with the fp16 copy those
                 layers need
- `unpad.py`     the decoder layers on real tokens only, attention on the padded layout
- `swiglu.py`    the organizers' example (fused SwiGLU), kept running; it does not count as ours
"""

import sys

from .swiglu import patch_qwen2_mlp


def _try(name, patch):
    try:
        patch()
    except Exception as exc:  # noqa: BLE001 - a missing piece must not stop the run
        print(f"[solution] {name} not applied: {exc.__class__.__name__}: {exc}", file=sys.stderr)


def apply():
    from .fused_ce import patch_qwen2_causal_lm, patch_sft_trainer
    from .nf4 import patch_bnb_matmul_4bit, patch_linear4bit
    from .rmsnorm import patch_qwen2_rmsnorm
    from .unpad import patch_dropout, patch_qwen2_attention, patch_qwen2_model

    _try("example SwiGLU", patch_qwen2_mlp)
    _try("NF4 dequantization", patch_bnb_matmul_4bit)
    _try("4-bit linear layers in fp16", patch_linear4bit)
    _try("RMSNorm", patch_qwen2_rmsnorm)
    _try("fused head and loss (model)", patch_qwen2_causal_lm)
    _try("fused head and loss (trainer)", patch_sft_trainer)
    _try("padding-free decoder", patch_qwen2_model)
    _try("padding-free attention", patch_qwen2_attention)
    _try("LoRA dropout with the base masks", patch_dropout)
