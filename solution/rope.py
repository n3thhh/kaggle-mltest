"""Rotary embedding, the fp16 cast for attention and the GQA expansion of k and v, in one kernel.

Between the q/k/v projections and attention, transformers runs (per layer, forward) q * cos, a
rotate_half built from two slices, a negation and a concatenation, a multiply by sin and an add,
the same for k, then autocast's fp16 casts, and repeat_kv's expand-and-copy of k and v from 2 to
12 heads; the backward pass mirrors all of it. That is some forty small launches per layer, and the
training step is bound by the CPU launching kernels. Here one kernel does it each way.

    y[:half] = x[:half] * cos[:half] - x[half:] * sin[:half]
    y[half:] = x[half:] * cos[half:] + x[:half] * sin[half:]

(x * cos + rotate_half(x) * sin, computed in fp32 as transformers does, then rounded to fp16 as the
attention call does). The backward pass applies the transposed rotation and sums the gradients of
the expanded heads back onto their key/value head.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n_tokens"])
def rope_expand_fwd_kernel(q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr, qo_ptr, ko_ptr, vo_ptr,
                           n_tokens, N_HEADS: tl.constexpr, N_KV: tl.constexpr,
                           HEAD_DIM: tl.constexpr):
    """Program (token, head): rotates q[token, head] and the head's k[token, head // groups],
    and copies v; outputs are [heads, tokens, head_dim] (attention's layout, k/v expanded)."""
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    groups: tl.constexpr = N_HEADS // N_KV
    kv = head // groups
    half: tl.constexpr = HEAD_DIM // 2
    i = tl.arange(0, half)
    cos_lo = tl.load(cos_ptr + token * HEAD_DIM + i)
    cos_hi = tl.load(cos_ptr + token * HEAD_DIM + half + i)
    sin_lo = tl.load(sin_ptr + token * HEAD_DIM + i)
    sin_hi = tl.load(sin_ptr + token * HEAD_DIM + half + i)
    out = (head.to(tl.int64) * n_tokens + token) * HEAD_DIM

    src = (token * N_HEADS + head) * HEAD_DIM
    x_lo = tl.load(q_ptr + src + i).to(tl.float32)
    x_hi = tl.load(q_ptr + src + half + i).to(tl.float32)
    tl.store(qo_ptr + out + i, (x_lo * cos_lo - x_hi * sin_lo).to(qo_ptr.dtype.element_ty))
    tl.store(qo_ptr + out + half + i, (x_hi * cos_hi + x_lo * sin_hi).to(qo_ptr.dtype.element_ty))

    src = (token * N_KV + kv) * HEAD_DIM
    x_lo = tl.load(k_ptr + src + i).to(tl.float32)
    x_hi = tl.load(k_ptr + src + half + i).to(tl.float32)
    tl.store(ko_ptr + out + i, (x_lo * cos_lo - x_hi * sin_lo).to(ko_ptr.dtype.element_ty))
    tl.store(ko_ptr + out + half + i, (x_hi * cos_hi + x_lo * sin_hi).to(ko_ptr.dtype.element_ty))
    tl.store(vo_ptr + out + i, tl.load(v_ptr + src + i).to(vo_ptr.dtype.element_ty))
    tl.store(vo_ptr + out + half + i, tl.load(v_ptr + src + half + i).to(vo_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=["n_tokens"])
def rope_expand_bwd_kernel(dqo_ptr, dko_ptr, dvo_ptr, cos_ptr, sin_ptr, dq_ptr, dk_ptr, dv_ptr,
                           n_tokens, N_HEADS: tl.constexpr, N_KV: tl.constexpr,
                           HEAD_DIM: tl.constexpr):
    """Program (token, query head or key/value head): the transposed rotation of the gradient
    (dx_lo = g_lo cos_lo + g_hi sin_hi, dx_hi = g_hi cos_hi - g_lo sin_lo); for a key/value head
    the gradients of its expanded copies are summed first."""
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    groups: tl.constexpr = N_HEADS // N_KV
    half: tl.constexpr = HEAD_DIM // 2
    i = tl.arange(0, half)
    cos_lo = tl.load(cos_ptr + token * HEAD_DIM + i)
    cos_hi = tl.load(cos_ptr + token * HEAD_DIM + half + i)
    sin_lo = tl.load(sin_ptr + token * HEAD_DIM + i)
    sin_hi = tl.load(sin_ptr + token * HEAD_DIM + half + i)
    if head < N_HEADS:
        src = (head.to(tl.int64) * n_tokens + token) * HEAD_DIM
        g_lo = tl.load(dqo_ptr + src + i).to(tl.float32)
        g_hi = tl.load(dqo_ptr + src + half + i).to(tl.float32)
        dst = (token * N_HEADS + head) * HEAD_DIM
        tl.store(dq_ptr + dst + i, (g_lo * cos_lo + g_hi * sin_hi).to(dq_ptr.dtype.element_ty))
        tl.store(dq_ptr + dst + half + i,
                 (g_hi * cos_hi - g_lo * sin_lo).to(dq_ptr.dtype.element_ty))
    else:
        kv = head - N_HEADS
        gk_lo = tl.zeros([half], dtype=tl.float32)
        gk_hi = tl.zeros([half], dtype=tl.float32)
        gv_lo = tl.zeros([half], dtype=tl.float32)
        gv_hi = tl.zeros([half], dtype=tl.float32)
        for r in tl.static_range(groups):
            src = ((kv * groups + r).to(tl.int64) * n_tokens + token) * HEAD_DIM
            gk_lo += tl.load(dko_ptr + src + i).to(tl.float32)
            gk_hi += tl.load(dko_ptr + src + half + i).to(tl.float32)
            gv_lo += tl.load(dvo_ptr + src + i).to(tl.float32)
            gv_hi += tl.load(dvo_ptr + src + half + i).to(tl.float32)
        dst = (token * N_KV + kv) * HEAD_DIM
        tl.store(dk_ptr + dst + i, (gk_lo * cos_lo + gk_hi * sin_hi).to(dk_ptr.dtype.element_ty))
        tl.store(dk_ptr + dst + half + i,
                 (gk_hi * cos_hi - gk_lo * sin_lo).to(dk_ptr.dtype.element_ty))
        tl.store(dv_ptr + dst + i, gv_lo.to(dv_ptr.dtype.element_ty))
        tl.store(dv_ptr + dst + half + i, gv_hi.to(dv_ptr.dtype.element_ty))


class RopeExpand(torch.autograd.Function):
    """q [N, H, D], k/v [N, KV, D], cos/sin [N, D] fp32 -> q, k, v as [H, N, D] in `dtype`,
    q and k rotated, k and v expanded to the H query heads."""

    @staticmethod
    def forward(ctx, q, k, v, cos, sin, dtype):
        n, heads, dim = q.shape
        kv_heads = k.shape[1]
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        cos, sin = cos.contiguous(), sin.contiguous()
        qo = torch.empty((heads, n, dim), dtype=dtype, device=q.device)
        ko, vo = torch.empty_like(qo), torch.empty_like(qo)
        rope_expand_fwd_kernel[(n, heads)](q, k, v, cos, sin, qo, ko, vo, n, N_HEADS=heads,
                                           N_KV=kv_heads, HEAD_DIM=dim)
        ctx.save_for_backward(cos, sin)
        ctx.meta = (n, heads, kv_heads, dim, q.dtype, k.dtype, v.dtype)
        return qo, ko, vo

    @staticmethod
    def backward(ctx, dqo, dko, dvo):
        cos, sin = ctx.saved_tensors
        n, heads, kv_heads, dim, q_dtype, k_dtype, v_dtype = ctx.meta
        device = cos.device
        zeros = None
        if dqo is None or dko is None or dvo is None:
            zeros = torch.zeros((heads, n, dim), dtype=torch.float16, device=device)
        dqo = (dqo if dqo is not None else zeros).contiguous()
        dko = (dko if dko is not None else zeros).contiguous()
        dvo = (dvo if dvo is not None else zeros).contiguous()
        dq = torch.empty((n, heads, dim), dtype=q_dtype, device=device)
        dk = torch.empty((n, kv_heads, dim), dtype=k_dtype, device=device)
        dv = torch.empty((n, kv_heads, dim), dtype=v_dtype, device=device)
        rope_expand_bwd_kernel[(n, heads + kv_heads)](dqo, dko, dvo, cos, sin, dq, dk, dv, n,
                                                      N_HEADS=heads, N_KV=kv_heads, HEAD_DIM=dim)
        return dq, dk, dv, None, None, None
