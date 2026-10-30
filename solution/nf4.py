"""NF4 dequantization in one Triton kernel, and the 4-bit linear layer that uses it.

bitsandbytes (0.50.2) dequantizes a double-quantized ("nested") NF4 weight for a large batch in
three launches: an 8-bit blockwise dequant of the absmax statistics, an element-wise add of their
offset, then the 4-bit dequant itself. It does that twice per layer and step: in the forward pass
and again in the backward pass, for every one of the 196 quantized linear layers of Qwen2.5-1.5B.

The weights are frozen, so the absmax statistics never change. We reconstruct them once per weight
with bitsandbytes' own ops (bit for bit what it computes on every call), keep them as fp32 (about
1.6% of the weight count), and dequantize with a single kernel: every output element reads its byte,
takes its nibble (high nibble first, as bitsandbytes packs them), looks up the NF4 code and scales
it by its block's absmax, in fp32, then rounds once to fp16. This is the same arithmetic as
bitsandbytes' `kDequantizeBlockwise`, so the dequantized weight is identical.

Where bitsandbytes would route a small batch through its own fused GEMV/GEMM kernel, we leave it
alone (same heuristic, same function), so the numerics of the base run are kept everywhere.
"""

import torch
import triton
import triton.language as tl

DEQUANT_BLOCK = 2048


@triton.jit
def nf4_dequant_kernel(packed_ptr, absmax_ptr, code_ptr, out_ptr, n_elements,
                       QBLOCK: tl.constexpr, BLOCK: tl.constexpr):
    """out[i] = code[nibble(i)] * absmax[i // QBLOCK] for one tile of BLOCK output elements."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    byte = tl.load(packed_ptr + offs // 2, mask=mask, other=0).to(tl.int32)
    # Even elements live in the high nibble, odd ones in the low nibble.
    shift = (1 - (offs % 2)).to(tl.int32) * 4
    nibble = (byte >> shift) & 0xF
    value = tl.load(code_ptr + nibble, mask=mask, other=0.0)
    scale = tl.load(absmax_ptr + offs // QBLOCK, mask=mask, other=0.0)
    tl.store(out_ptr + offs, (value * scale).to(out_ptr.dtype.element_ty), mask=mask)


def _absmax_fp32(quant_state):
    """The per-block absmax as fp32, reconstructed once per weight with bitsandbytes' own ops."""
    cached = getattr(quant_state, "_soup_absmax_fp32", None)
    if cached is not None:
        return cached
    absmax = quant_state.absmax
    if quant_state.nested:
        import bitsandbytes.functional as bnb_f

        absmax = bnb_f.dequantize_blockwise(quant_state.absmax, quant_state.state2)
        absmax = absmax + quant_state.offset
    absmax = absmax.float().contiguous()
    code = quant_state.code.to(device=absmax.device, dtype=torch.float32).contiguous()
    quant_state._soup_absmax_fp32 = (absmax, code)
    return absmax, code


def dequantize_nf4(packed, quant_state, dtype=torch.float16):
    """The [N, K] weight of a 4-bit NF4 layer in `dtype`."""
    absmax, code = _absmax_fp32(quant_state)
    out = torch.empty(tuple(quant_state.shape), dtype=dtype, device=packed.device)
    n = out.numel()
    nf4_dequant_kernel[(triton.cdiv(n, DEQUANT_BLOCK),)](
        packed, absmax, code, out, n, QBLOCK=quant_state.blocksize, BLOCK=DEQUANT_BLOCK)
    return out


class NF4Linear(torch.autograd.Function):
    """`x @ dequant(W).T + bias` with the weight dequantized by our kernel, forward and backward.
    Like bitsandbytes' MatMul4Bit, only the packed weight is kept for the backward pass."""

    @staticmethod
    def forward(ctx, x, packed, quant_state, bias):
        weight = dequantize_nf4(packed, quant_state, x.dtype)
        out = torch.nn.functional.linear(x, weight, bias)
        ctx.quant_state = quant_state
        ctx.has_bias_grad = bias is not None and ctx.needs_input_grad[3]
        ctx.save_for_backward(packed)
        return out

    @staticmethod
    def backward(ctx, grad):
        (packed,) = ctx.saved_tensors
        grad_x = grad_bias = None
        if ctx.needs_input_grad[0]:
            weight = dequantize_nf4(packed, ctx.quant_state, grad.dtype)
            grad_x = torch.matmul(grad, weight)
        if ctx.has_bias_grad:
            grad_bias = grad.reshape(-1, grad.shape[-1]).sum(0)
        return grad_x, None, None, grad_bias


def _base_uses_fused_kernel_for(m, k, quant_state, dtype, device_index):
    """True where bitsandbytes' CUDA backend would run its own fused 4-bit GEMM for an [m, k]
    input (small m); there we keep its path so the numerics stay those of the base run."""
    try:
        from bitsandbytes.backends.cuda import ops as cuda_ops
    except Exception:  # noqa: BLE001 - no such backend: nothing to defer to
        return False
    n = quant_state.shape[0]
    try:
        if m > cuda_ops._gemm_4bit_custom_max_m or k % quant_state.blocksize != 0:
            return False
        return bool(cuda_ops._gemm_4bit_use_custom_fn(device_index, dtype, m, n, k))
    except Exception:  # noqa: BLE001 - a different bitsandbytes: be conservative
        return True


def _base_uses_fused_kernel(x, quant_state, dtype=None):
    k = x.shape[-1]
    return _base_uses_fused_kernel_for(x.numel() // k, k, quant_state, dtype or x.dtype,
                                       x.device.index)


def _nf4_state_ok(packed, quant_state, in_features):
    return (quant_state is not None and quant_state.quant_type == "nf4"
            and len(quant_state.shape) == 2 and quant_state.shape[1] == in_features
            and quant_state.blocksize % 2 == 0
            and (not quant_state.nested or quant_state.state2.blocksize == 256)
            and packed.dtype == torch.uint8 and packed.is_contiguous())


def _supported(x, packed, quant_state):
    return (x.is_cuda and x.numel() > 0 and x.dtype in (torch.float16, torch.float32)
            and _nf4_state_ok(packed, quant_state, x.shape[-1]))


def _gemm_dtype(module, x):
    """The dtype the base run's GEMM actually runs in: autocast's when it is on (F.linear inside
    bitsandbytes is autocast), the layer's compute dtype otherwise."""
    if torch.is_autocast_enabled(x.device.type):
        return torch.get_autocast_dtype(x.device.type)
    return module.compute_dtype or x.dtype


def patch_linear4bit():
    """Linear4bit.forward through NF4Linear, dequantizing straight to the GEMM's dtype.

    With an fp32 input (the residual stream is fp32 after prepare_model_for_kbit_training) the
    base layer switches its compute dtype to fp32: bitsandbytes dequantizes the weight to fp32,
    autocast then copies it (and the input) to fp16 for the GEMM, and the fp16 result is copied
    back to fp32. Here the weight is dequantized to fp16 directly (the same fp32 product rounded
    once, so the same fp16 weight), the input is cast once (or taken from the fp16 copy our
    RMSNorm already wrote), and the output stays fp16: every consumer in Qwen2 either upcasts it
    exactly (residual add, RoPE) or casts it to fp16 anyway (SDPA, the next 4-bit layer)."""
    import bitsandbytes as bnb
    from bitsandbytes.nn import modules as bnb_modules

    cls = bnb_modules.Linear4bit
    if getattr(cls.forward, "_soup_patched", False):
        return
    original = cls.forward
    fix_state = bnb_modules.fix_4bit_weight_quant_state_from_module

    def forward(self, x):
        if type(self) is not cls and type(self).forward is not forward:
            return original(self, x)
        fix_state(self)
        quant_state = self.weight.quant_state
        if not x.is_cuda or not _supported(x, self.weight, quant_state):
            return original(self, x)
        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True
        gemm_dtype = _gemm_dtype(self, x)
        base_dtype = self.compute_dtype or x.dtype
        if gemm_dtype not in (torch.float16, torch.float32) or \
                _base_uses_fused_kernel(x, quant_state, base_dtype):
            return original(self, x)
        half = getattr(x, "_soup_half", None)
        if half is not None and half.dtype == gemm_dtype and half.shape == x.shape:
            x = half
        else:
            x = x.to(gemm_dtype)
        bias = self.bias
        if bias is not None:
            if bias.dtype != base_dtype:  # as Linear4bit.forward does to its bias
                bias.data = bias.data.to(base_dtype)
            bias = bias.to(gemm_dtype)
        return NF4Linear.apply(x, self.weight.view(-1), quant_state, bias)

    forward._soup_patched = True
    cls.forward = forward


def patch_bnb_matmul_4bit():
    """Route bitsandbytes' 4-bit matmul (what Linear4bit.forward calls) through NF4Linear."""
    import bitsandbytes
    from bitsandbytes.autograd import _functions as bnb_autograd

    if getattr(bitsandbytes.matmul_4bit, "_soup_patched", False):
        return
    original = bitsandbytes.matmul_4bit

    def matmul_4bit(A, B, quant_state, out=None, bias=None):
        if out is not None or not _supported(A, B, quant_state) or \
                _base_uses_fused_kernel(A, quant_state):
            return original(A, B, quant_state, out=out, bias=bias)
        return NF4Linear.apply(A, B.view(-1), quant_state, bias)

    matmul_4bit._soup_patched = True
    bitsandbytes.matmul_4bit = matmul_4bit
    bnb_autograd.matmul_4bit = matmul_4bit
