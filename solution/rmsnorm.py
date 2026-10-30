"""Qwen2RMSNorm forward and backward, one Triton kernel each way.

`prepare_model_for_kbit_training` keeps the 57 RMSNorm weights and the residual stream in fp32, and
transformers computes the norm as a chain of separate element-wise kernels (square, mean, add,
rsqrt, two multiplies, a cast) with more of them in the backward pass. Here each row is read once
and written once in each direction; the backward pass recomputes the normalized row from the input
and the saved 1/rms instead of keeping it.

    y = weight * (x * rsqrt(mean(x^2) + eps)).to(x.dtype)
    dx = rstd * (g - x_hat * mean(g * x_hat)),  g = dy * weight,  x_hat = x * rstd

The weights are frozen in this LoRA run; a norm with a trainable weight keeps the original path.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, half_ptr, rstd_ptr, n_cols, eps, BLOCK: tl.constexpr,
                       HALF_OUT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x32 * x32, axis=0) / n_cols + eps)
    # As transformers does: normalize in fp32, round to the input's dtype, then scale.
    x_hat = (x32 * rstd).to(x.dtype).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = w * x_hat
    tl.store(y_ptr + row * n_cols + cols, y.to(y_ptr.dtype.element_ty), mask=mask)
    if HALF_OUT:  # the fp16 copy the next 4-bit layers would each cast on their own
        tl.store(half_ptr + row * n_cols + cols, y.to(half_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def rmsnorm_bwd_kernel(dy_ptr, dhalf_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, n_cols,
                       BLOCK: tl.constexpr, HAS_DHALF: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_DHALF:  # both outputs were used: their gradients add up
        dy += tl.load(dhalf_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    x_hat = x * rstd
    g = dy * w
    proj = tl.sum(g * x_hat, axis=0) / n_cols
    dx = rstd * (g - x_hat * proj)
    tl.store(dx_ptr + row * n_cols + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


class RMSNorm(torch.autograd.Function):
    """Returns the norm in transformers' dtype and, when `half_dtype` is given, the same values
    rounded to that dtype (the input of the following 4-bit layers' fp16 GEMMs)."""

    @staticmethod
    def forward(ctx, x, weight, eps, half_dtype):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        n_rows, n_cols = x2.shape
        out_dtype = torch.result_type(weight, x)
        y = torch.empty((n_rows, n_cols), dtype=out_dtype, device=x.device)
        half = None
        if half_dtype is not None and half_dtype != out_dtype:
            half = torch.empty((n_rows, n_cols), dtype=half_dtype, device=x.device)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        block = triton.next_power_of_2(n_cols)
        warps = 8 if block >= 4096 else 4
        rmsnorm_fwd_kernel[(n_rows,)](x2, weight, y, half if half is not None else y, rstd,
                                      n_cols, eps, BLOCK=block, HALF_OUT=half is not None,
                                      num_warps=warps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape, ctx.block, ctx.warps = shape, block, warps
        ctx.set_materialize_grads(False)  # an output nobody used gets no zero gradient
        y = y.view(*shape[:-1], n_cols)
        if half is None:
            return y, None
        return y, half.view(*shape[:-1], n_cols)

    @staticmethod
    def backward(ctx, dy, dhalf):
        x2, weight, rstd = ctx.saved_tensors
        n_rows, n_cols = x2.shape
        if dy is None:
            dy, dhalf = dhalf, None
        dy2 = dy.reshape(-1, n_cols).contiguous()
        dh2 = dhalf.reshape(-1, n_cols).contiguous() if dhalf is not None else dy2
        dx = torch.empty_like(x2)
        rmsnorm_bwd_kernel[(n_rows,)](dy2, dh2, x2, weight, rstd, dx, n_cols, BLOCK=ctx.block,
                                      HAS_DHALF=dhalf is not None, num_warps=ctx.warps)
        return dx.view(ctx.shape), None, None, None


def rmsnorm(x, weight, eps, half_dtype=None):
    y, half = RMSNorm.apply(x, weight, eps, half_dtype)
    if half is not None:
        y._soup_half = half  # read by our Linear4bit.forward instead of casting y again
    return y


def patch_qwen2_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    if getattr(cls.forward, "_soup_patched", False):
        return
    original = cls.forward

    def forward(self, hidden_states):
        weight = self.weight
        if not hidden_states.is_cuda or weight.requires_grad or hidden_states.numel() == 0 or \
                weight.dim() != 1 or hidden_states.shape[-1] != weight.shape[0] or \
                hidden_states.shape[-1] > 16384:
            return original(self, hidden_states)
        half_dtype = None
        if torch.is_autocast_enabled("cuda") and \
                torch.get_autocast_dtype("cuda") in (torch.float16, torch.bfloat16):
            half_dtype = torch.get_autocast_dtype("cuda")
        return rmsnorm(hidden_states, weight, float(self.variance_epsilon), half_dtype)

    forward._soup_patched = True
    cls.forward = forward
