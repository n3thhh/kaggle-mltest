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
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x32 * x32, axis=0) / n_cols + eps)
    # As transformers does: normalize in fp32, round to the input's dtype, then scale.
    x_hat = (x32 * rstd).to(x.dtype).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * n_cols + cols, (w * x_hat).to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def rmsnorm_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    x_hat = x * rstd
    g = dy * w
    proj = tl.sum(g * x_hat, axis=0) / n_cols
    dx = rstd * (g - x_hat * proj)
    tl.store(dx_ptr + row * n_cols + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        n_rows, n_cols = x2.shape
        out_dtype = torch.result_type(weight, x)
        y = torch.empty((n_rows, n_cols), dtype=out_dtype, device=x.device)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        block = triton.next_power_of_2(n_cols)
        warps = 8 if block >= 4096 else 4
        rmsnorm_fwd_kernel[(n_rows,)](x2, weight, y, rstd, n_cols, eps, BLOCK=block,
                                      num_warps=warps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape, ctx.block, ctx.warps = shape, block, warps
        return y.view(*shape[:-1], n_cols)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        n_rows, n_cols = x2.shape
        dy2 = dy.reshape(-1, n_cols).contiguous()
        dx = torch.empty_like(x2)
        rmsnorm_bwd_kernel[(n_rows,)](dy2, x2, weight, rstd, dx, n_cols, BLOCK=ctx.block,
                                      num_warps=ctx.warps)
        return dx.view(ctx.shape), None, None


def rmsnorm(x, weight, eps):
    return RMSNorm.apply(x, weight, eps)


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
        return rmsnorm(hidden_states, weight, float(self.variance_epsilon))

    forward._soup_patched = True
    cls.forward = forward
