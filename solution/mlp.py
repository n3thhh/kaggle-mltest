"""Qwen2 MLP that keeps only its fp16 input for the backward pass and recomputes the rest.

The MLP is down_proj(silu(gate_proj(x)) * up_proj(x)) with three frozen 4-bit layers. For the
backward pass the SwiGLU needs gate and up: two [tokens, 8960] fp16 tensors per layer, by far the
largest thing a training step keeps (about 1.4 GB at this run's longest batch). Here the forward
pass keeps only x (the fp16 copy our RMSNorm wrote, [tokens, 1536]) and the backward pass
recomputes gate and up with the same dequantized weights it needs anyway for the input gradient:
each weight is dequantized once per pass, as before, and the extra work is two GEMMs per layer in
the backward pass. The step is bound by the CPU launching kernels, not by the GPU, so most of that
extra GPU work fits in time the GPU was idle.

Every product is the one the unfused layers compute (same fp16 weights, same F.linear, the
example's SwiGLU kernels for silu(gate) * up and its gradient), so the values are unchanged.
"""

import torch

from . import swiglu as example
from .nf4 import _base_uses_fused_kernel_for, _nf4_state_ok, dequantize_nf4

import triton


def _swiglu_fwd(gate, up):
    out = torch.empty_like(gate)
    n = gate.numel()
    example.example_swiglu_fwd_kernel[(triton.cdiv(n, example.BLOCK),)](
        gate, up, out, n, BLOCK=example.BLOCK)
    return out


def _swiglu_bwd(grad, gate, up):
    dgate, dup = torch.empty_like(gate), torch.empty_like(up)
    n = gate.numel()
    example.example_swiglu_bwd_kernel[(triton.cdiv(n, example.BLOCK),)](
        grad, gate, up, dgate, dup, n, BLOCK=example.BLOCK)
    return dgate, dup


class RecomputedMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate_w, gate_qs, up_w, up_qs, down_w, down_qs):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        weight = dequantize_nf4(gate_w, gate_qs, x.dtype)
        gate = torch.nn.functional.linear(x2, weight)
        weight = dequantize_nf4(up_w, up_qs, x.dtype)
        up = torch.nn.functional.linear(x2, weight)
        del weight
        hidden = _swiglu_fwd(gate, up)
        del gate, up
        out = torch.nn.functional.linear(hidden, dequantize_nf4(down_w, down_qs, x.dtype))
        ctx.save_for_backward(x2, gate_w, up_w, down_w)
        ctx.qs, ctx.shape = (gate_qs, up_qs, down_qs), shape
        return out.view(*shape[:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad):
        x2, gate_w, up_w, down_w = ctx.saved_tensors
        gate_qs, up_qs, down_qs = ctx.qs
        dtype = x2.dtype
        grad = grad.reshape(-1, grad.shape[-1]).to(dtype)
        d_hidden = torch.matmul(grad, dequantize_nf4(down_w, down_qs, dtype))
        w_gate = dequantize_nf4(gate_w, gate_qs, dtype)
        gate = torch.nn.functional.linear(x2, w_gate)
        w_up = dequantize_nf4(up_w, up_qs, dtype)
        up = torch.nn.functional.linear(x2, w_up)
        d_gate, d_up = _swiglu_bwd(d_hidden, gate, up)
        del gate, up, d_hidden
        dx = torch.matmul(d_gate, w_gate)
        dx += torch.matmul(d_up, w_up)  # autograd sums the two branches' fp16 gradients the same way
        return dx.view(ctx.shape), None, None, None, None, None, None


def _usable(layer, rows, in_features, device_index):
    """A frozen bias-free NF4 layer whose base-run GEMM is dequantize + F.linear (bitsandbytes'
    own fused kernel only takes small batches; these layers see fp32 inputs in the base run)."""
    from bitsandbytes.nn import Linear4bit

    if type(layer) is not Linear4bit or layer.bias is not None or layer.weight.requires_grad:
        return False
    qs = getattr(layer.weight, "quant_state", None)
    if not _nf4_state_ok(layer.weight, qs, in_features):
        return False
    base_dtype = layer.compute_dtype if layer.compute_type_is_set else torch.float32
    return not _base_uses_fused_kernel_for(rows, in_features, qs, base_dtype or torch.float32,
                                           device_index)


def patch_qwen2_mlp_recompute():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls.forward, "_soup_recompute", False):
        return
    previous = cls.forward  # the example's fused SwiGLU forward

    def forward(self, x):
        if not x.is_cuda or not torch.is_autocast_enabled("cuda") or \
                torch.get_autocast_dtype("cuda") != torch.float16 or \
                getattr(getattr(self, "config", None), "hidden_act", "silu") != "silu" or \
                not torch.is_grad_enabled() or \
                x.dtype not in (torch.float16, torch.float32) or x.numel() == 0:
            return previous(self, x)
        rows, dev = x.numel() // x.shape[-1], x.device.index
        if not _usable(self.gate_proj, rows, x.shape[-1], dev) or \
                not _usable(self.up_proj, rows, x.shape[-1], dev) or \
                not _usable(self.down_proj, rows, self.gate_proj.out_features, dev):
            return previous(self, x)
        half = getattr(x, "_soup_half", None)
        if half is None or half.dtype != torch.float16 or half.shape != x.shape:
            half = x.to(torch.float16)
        for layer in (self.gate_proj, self.up_proj, self.down_proj):
            if not layer.compute_type_is_set:  # what Linear4bit.forward would record
                layer.set_compute_type(x)
                layer.compute_type_is_set = True
        return RecomputedMLP.apply(
            half, self.gate_proj.weight.view(-1), self.gate_proj.weight.quant_state,
            self.up_proj.weight.view(-1), self.up_proj.weight.quant_state,
            self.down_proj.weight.view(-1), self.down_proj.weight.quant_state)

    forward._soup_recompute = True
    cls.forward = forward
