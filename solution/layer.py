"""A whole Qwen2 decoder layer (attention block + MLP block) as one autograd function.

In the base run one layer is about a hundred small PyTorch calls in each direction: seven 4-bit
linear layers through bitsandbytes, peft's LoRA wrappers on q and v with their dropout, the
rotary embedding, repeat_kv, attention, two RMSNorms, SwiGLU, residual adds and the autocast
casts between all of them, each with its autograd bookkeeping. On a T4 the training step is bound
by the CPU issuing these launches, not by the GPU. Here the layer's forward pass is about 25
launches and its backward pass about 35, written out by hand:

  forward   h1 = x + o_proj(attn(rope(qkv(norm1(x)) + lora)))      h2 = h1 + down(swiglu(gate_up(norm2(h1))))
  backward  the same in reverse, with the gradients of LoRA's A and B returned to autograd

* q, k and v are one GEMM over the three dequantized weights stacked (bias added in the GEMM);
  gate and up are one GEMM; their input gradients are one GEMM each in the backward pass.
* LoRA on q and v runs inside: the dropout masks are drawn by the same calls, in the same order,
  on the same padded shape as in the base run (unpad.py), and the backward pass redraws them by
  replaying the saved RNG state instead of keeping them. The LoRA output is added to the base
  projection in fp32 inside the rotary kernel, as the base run adds the two in fp32.
* Attention runs on all rows at once through PyTorch's memory-efficient kernel in its variable-
  length form (cu_seqlens = the rows' boundaries), causal within each row: a real token attends to
  exactly the keys the base run's padding mask allows. A self-check at first use compares it with
  plain SDPA and switches to an autograd fallback if anything differs.
* The backward pass keeps per token and layer: the layer input x (fp32), o_proj's output, the
  rotated q, k, v, attention's output and its log-sum-exp, and optionally gate/up (mlp.py decides
  per batch: long batches recompute them). Everything else is recomputed by cheap kernels.

The arithmetic follows the base run's: every GEMM is the fp16 GEMM autocast runs there, sums
that the base run does in fp32 are done in fp32, and values the base run rounds to fp16 are
rounded to fp16 at the same points.
"""

import torch
import triton
import triton.language as tl

from . import unpad
from .nf4 import _absmax_fp32

DEQUANT_BLOCK = 2048
_ALLOW_CPU = False  # tests only (TRITON_INTERPRET=1)


# ================================================================================ kernels

@triton.jit
def nf4_dequant_multi_kernel(p0, a0, p1, a1, p2, a2, code_ptr, out_ptr, nb0, nb1,
                             QBLOCK: tl.constexpr, BLOCK: tl.constexpr):
    """Up to three NF4 weights dequantized into one buffer, one after the other (q, k, v or gate,
    up): block `pid` of the output belongs to weight 0, 1 or 2 by its index. Each element: its
    nibble (high first, as bitsandbytes packs), the NF4 code, times its block's absmax, rounded
    once to the output dtype - bitsandbytes' kDequantizeBlockwise, bit for bit."""
    pid = tl.program_id(0)
    if pid < nb0:
        packed = p0
        absmax = a0
        local = pid
    elif pid < nb0 + nb1:
        packed = p1
        absmax = a1
        local = pid - nb0
    else:
        packed = p2
        absmax = a2
        local = pid - nb0 - nb1
    offs = local.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    byte = tl.load(packed + offs // 2).to(tl.int32)
    nibble = (byte >> ((1 - (offs % 2)).to(tl.int32) * 4)) & 0xF
    value = tl.load(code_ptr + nibble) * tl.load(absmax + offs // QBLOCK)
    tl.store(out_ptr + pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK),
             value.to(out_ptr.dtype.element_ty))


@triton.jit(do_not_specialize=["has_d", "write_y"])
def add_rmsnorm_fwd_kernel(x_ptr, d_ptr, w_ptr, y_ptr, y16_ptr, rstd_ptr, n_cols, eps, has_d,
                           write_y, BLOCK: tl.constexpr):
    """h = x (+ d), then RMSNorm: y = w * h / rms(h) in fp32 (written if WRITE_Y), its copy in
    the GEMM dtype, and 1/rms. d is the fp16 output of o_proj (h = the residual stream after
    attention, summed in fp32 as the base run does)."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    h = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    if has_d != 0:
        h += tl.load(d_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(h * h, axis=0) / n_cols + eps)
    y = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32) * (h * rstd)
    if write_y != 0:
        tl.store(y_ptr + row * n_cols + cols, y, mask=mask)
    tl.store(y16_ptr + row * n_cols + cols, y.to(y16_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit(do_not_specialize=["has_lora", "has_dy", "has_d"])
def rmsnorm_bwd_lora_kernel(dy_ptr, dq_ptr, dv_ptr, keepq_ptr, keepv_ptr, rows_ptr, x_ptr, d_ptr,
                            w_ptr, rstd_ptr, dres_ptr, dx_ptr, n_cols, scale_q, scale_v, has_dy,
                            has_lora, has_d, BLOCK: tl.constexpr):
    """dx of RMSNorm for one row, plus the residual's gradient. The gradient reaching the norm's
    output is the base GEMMs' (fp16) plus, before attention, LoRA's two dropout branches
    (fp16 gradient * the row's padded-batch dropout mask * 1/(1-p)), summed in fp32. The norm's
    input is x (+ d), recomputed instead of stored."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    g = tl.zeros([BLOCK], dtype=tl.float32)
    if has_dy != 0:
        g += tl.load(dy_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    if has_lora != 0:
        src = tl.load(rows_ptr + row).to(tl.int64)
        kq = tl.load(keepq_ptr + src * n_cols + cols, mask=mask, other=0).to(tl.float32)
        kv = tl.load(keepv_ptr + src * n_cols + cols, mask=mask, other=0).to(tl.float32)
        g += tl.load(dq_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32) * kq * scale_q
        g += tl.load(dv_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32) * kv * scale_v
    h = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    if has_d != 0:
        h += tl.load(d_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    h_hat = h * rstd
    gw = g * tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    proj = tl.sum(gw * h_hat, axis=0) / n_cols
    dx = rstd * (gw - h_hat * proj)
    dx += tl.load(dres_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(dx_ptr + row * n_cols + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


@triton.jit
def lora_dropout_pair_kernel(y_ptr, keepq_ptr, keepv_ptr, rows_ptr, oq_ptr, ov_ptr, n_cols,
                             scale_q, scale_v, BLOCK: tl.constexpr):
    """LoRA's dropout for q and v on the real tokens: y * mask * 1/(1-p) in fp32 with each
    token's row of the masks drawn over the padded batch, rounded once to the GEMM dtype."""
    row = tl.program_id(0).to(tl.int64)
    src = tl.load(rows_ptr + row).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    y = tl.load(y_ptr + row * n_cols + cols, mask=mask, other=0.0)
    kq = tl.load(keepq_ptr + src * n_cols + cols, mask=mask, other=0).to(tl.float32)
    kv = tl.load(keepv_ptr + src * n_cols + cols, mask=mask, other=0).to(tl.float32)
    tl.store(oq_ptr + row * n_cols + cols, (y * kq * scale_q).to(oq_ptr.dtype.element_ty), mask=mask)
    tl.store(ov_ptr + row * n_cols + cols, (y * kv * scale_v).to(ov_ptr.dtype.element_ty), mask=mask)


@triton.jit
def qkv_lora_rope_kernel(qkv_ptr, lq_ptr, lv_ptr, cos_ptr, sin_ptr, qe_ptr, ke_ptr, ve_ptr,
                         kc_ptr, vc_ptr, scale_q, scale_v, N_HEADS: tl.constexpr,
                         N_KV: tl.constexpr, HEAD_DIM: tl.constexpr):
    """Program (token, query head). q = q_base + s_q * lora_q and v = v_base + s_v * lora_v in
    fp32 (as the base run adds the fp16 LoRA output to the base projection's fp32 output), the
    rotary embedding on q and k in fp32, then rounded to the attention dtype: q, and k and v
    expanded to every query head (attention's [tokens, heads, dim] layout), plus the
    unexpanded k and v that the backward pass keeps."""
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    groups: tl.constexpr = N_HEADS // N_KV
    half: tl.constexpr = HEAD_DIM // 2
    row: tl.constexpr = (N_HEADS + 2 * N_KV) * HEAD_DIM
    kv = head // groups
    i = tl.arange(0, half)
    cos_lo = tl.load(cos_ptr + token * HEAD_DIM + i)
    cos_hi = tl.load(cos_ptr + token * HEAD_DIM + half + i)
    sin_lo = tl.load(sin_ptr + token * HEAD_DIM + i)
    sin_hi = tl.load(sin_ptr + token * HEAD_DIM + half + i)

    q_src = token * row + head * HEAD_DIM
    l_src = (token * N_HEADS + head) * HEAD_DIM
    lo = tl.load(qkv_ptr + q_src + i).to(tl.float32) + \
        tl.load(lq_ptr + l_src + i).to(tl.float32) * scale_q
    hi = tl.load(qkv_ptr + q_src + half + i).to(tl.float32) + \
        tl.load(lq_ptr + l_src + half + i).to(tl.float32) * scale_q
    dst = (token * N_HEADS + head) * HEAD_DIM
    tl.store(qe_ptr + dst + i, (lo * cos_lo - hi * sin_lo).to(qe_ptr.dtype.element_ty))
    tl.store(qe_ptr + dst + half + i, (hi * cos_hi + lo * sin_hi).to(qe_ptr.dtype.element_ty))

    k_src = token * row + (N_HEADS + kv) * HEAD_DIM
    lo = tl.load(qkv_ptr + k_src + i).to(tl.float32)
    hi = tl.load(qkv_ptr + k_src + half + i).to(tl.float32)
    k_lo = (lo * cos_lo - hi * sin_lo).to(ke_ptr.dtype.element_ty)
    k_hi = (hi * cos_hi + lo * sin_hi).to(ke_ptr.dtype.element_ty)
    tl.store(ke_ptr + dst + i, k_lo)
    tl.store(ke_ptr + dst + half + i, k_hi)

    v_src = token * row + (N_HEADS + N_KV + kv) * HEAD_DIM
    lv_src = (token * N_KV + kv) * HEAD_DIM
    v_lo = (tl.load(qkv_ptr + v_src + i).to(tl.float32) +
            tl.load(lv_ptr + lv_src + i).to(tl.float32) * scale_v).to(ve_ptr.dtype.element_ty)
    v_hi = (tl.load(qkv_ptr + v_src + half + i).to(tl.float32) +
            tl.load(lv_ptr + lv_src + half + i).to(tl.float32) * scale_v).to(ve_ptr.dtype.element_ty)
    tl.store(ve_ptr + dst + i, v_lo)
    tl.store(ve_ptr + dst + half + i, v_hi)
    if head % groups == 0:
        tl.store(kc_ptr + lv_src + i, k_lo)
        tl.store(kc_ptr + lv_src + half + i, k_hi)
        tl.store(vc_ptr + lv_src + i, v_lo)
        tl.store(vc_ptr + lv_src + half + i, v_hi)


@triton.jit
def qkv_rope_bwd_kernel(dqe_ptr, dke_ptr, dve_ptr, cos_ptr, sin_ptr, dqkv_ptr,
                        N_HEADS: tl.constexpr, N_KV: tl.constexpr, HEAD_DIM: tl.constexpr):
    """Program (token, query head or key/value head): the transposed rotation of attention's
    gradients, in fp32, written to the [tokens, q|k|v] layout of the fused projection. A key/value
    head first sums the gradients of its expanded copies."""
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    groups: tl.constexpr = N_HEADS // N_KV
    half: tl.constexpr = HEAD_DIM // 2
    row: tl.constexpr = (N_HEADS + 2 * N_KV) * HEAD_DIM
    i = tl.arange(0, half)
    cos_lo = tl.load(cos_ptr + token * HEAD_DIM + i)
    cos_hi = tl.load(cos_ptr + token * HEAD_DIM + half + i)
    sin_lo = tl.load(sin_ptr + token * HEAD_DIM + i)
    sin_hi = tl.load(sin_ptr + token * HEAD_DIM + half + i)
    if head < N_HEADS:
        src = (token * N_HEADS + head) * HEAD_DIM
        g_lo = tl.load(dqe_ptr + src + i).to(tl.float32)
        g_hi = tl.load(dqe_ptr + src + half + i).to(tl.float32)
        dst = token * row + head * HEAD_DIM
        tl.store(dqkv_ptr + dst + i, (g_lo * cos_lo + g_hi * sin_hi).to(dqkv_ptr.dtype.element_ty))
        tl.store(dqkv_ptr + dst + half + i,
                 (g_hi * cos_hi - g_lo * sin_lo).to(dqkv_ptr.dtype.element_ty))
    else:
        kv = head - N_HEADS
        gk_lo = tl.zeros([half], dtype=tl.float32)
        gk_hi = tl.zeros([half], dtype=tl.float32)
        gv_lo = tl.zeros([half], dtype=tl.float32)
        gv_hi = tl.zeros([half], dtype=tl.float32)
        for r in tl.static_range(groups):
            src = (token * N_HEADS + kv * groups + r) * HEAD_DIM
            gk_lo += tl.load(dke_ptr + src + i).to(tl.float32)
            gk_hi += tl.load(dke_ptr + src + half + i).to(tl.float32)
            gv_lo += tl.load(dve_ptr + src + i).to(tl.float32)
            gv_hi += tl.load(dve_ptr + src + half + i).to(tl.float32)
        dst = token * row + (N_HEADS + kv) * HEAD_DIM
        tl.store(dqkv_ptr + dst + i, (gk_lo * cos_lo + gk_hi * sin_hi).to(dqkv_ptr.dtype.element_ty))
        tl.store(dqkv_ptr + dst + half + i,
                 (gk_hi * cos_hi - gk_lo * sin_lo).to(dqkv_ptr.dtype.element_ty))
        dst = token * row + (N_HEADS + N_KV + kv) * HEAD_DIM
        tl.store(dqkv_ptr + dst + i, gv_lo.to(dqkv_ptr.dtype.element_ty))
        tl.store(dqkv_ptr + dst + half + i, gv_hi.to(dqkv_ptr.dtype.element_ty))


@triton.jit
def gate_up_act_fwd_kernel(gu_ptr, h_ptr, inter, BLOCK: tl.constexpr):
    """One row block of the fused gate|up projection [tokens, 2*inter] -> silu(gate) * up in fp32
    (the base run applies SiLU and the product to the fp32 outputs of the two projections),
    rounded once for the down projection's GEMM."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < inter
    base = row * 2 * inter
    gate = tl.load(gu_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gu_ptr + base + inter + cols, mask=mask, other=0.0).to(tl.float32)
    act = gate / (1.0 + tl.exp(-gate))
    tl.store(h_ptr + row * inter + cols, (act * up).to(h_ptr.dtype.element_ty), mask=mask)


@triton.jit
def gate_up_act_bwd_kernel(dh_ptr, gu_ptr, dgu_ptr, inter, BLOCK: tl.constexpr):
    """Gradients of gate and up from the gradient of silu(gate) * up, in fp32, written to the
    fused [tokens, gate|up] layout the input-gradient GEMM reads."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < inter
    base = row * 2 * inter
    dh = tl.load(dh_ptr + row * inter + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gu_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gu_ptr + base + inter + cols, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-gate))
    d_up = dh * gate * sig
    d_gate = dh * up * sig * (1.0 + gate * (1.0 - sig))
    tl.store(dgu_ptr + base + cols, d_gate.to(dgu_ptr.dtype.element_ty), mask=mask)
    tl.store(dgu_ptr + base + inter + cols, d_up.to(dgu_ptr.dtype.element_ty), mask=mask)


ACT_BLOCK = 1024
ROW_BLOCK = 1024


# ================================================================================ helpers

_CODE = {}


def _code(quant_state):
    """The NF4 code table as fp32 on the weight's device (the same 16 values for every weight)."""
    code = quant_state.code
    key = (code.device, code.data_ptr())
    if key not in _CODE:
        _CODE[key] = code.to(torch.float32).contiguous()
    return _CODE[key]


def _dequant(layers, dtype, streamed=None):
    """The stacked weights of up to three NF4 layers ([sum of out_features, in_features]).
    `streamed`: [(packed, absmax_fp32)] on the GPU for these layers (offload.py), else each
    layer's own resident packed bytes and absmax."""
    states = [layer.weight.quant_state for layer in layers]
    rows = sum(int(s.shape[0]) for s in states)
    cols = int(states[0].shape[1])
    device = streamed[0][0].device if streamed else layers[0].weight.device
    out = torch.empty((rows, cols), dtype=dtype, device=device)
    if streamed is None:
        sources = []
        for layer, qs in zip(layers, states):
            if getattr(qs, "_soup_host", None) is not None:
                from .offload import fetch
                from .nf4 import _absmax_fp32

                sources.append((fetch(qs, device), _absmax_fp32(qs)[0]))
            else:
                from .nf4 import _absmax_fp32

                sources.append((layer.weight.view(-1), _absmax_fp32(qs)[0]))
    else:
        sources = list(streamed)
    numels = [int(s.shape[0]) * cols for s in states]
    if all(n % DEQUANT_BLOCK == 0 for n in numels):
        blocks = [n // DEQUANT_BLOCK for n in numels] + [0, 0]
        while len(sources) < 3:
            sources.append(sources[0])
        nf4_dequant_multi_kernel[(sum(blocks[:len(layers)]),)](
            sources[0][0], sources[0][1], sources[1][0], sources[1][1], sources[2][0],
            sources[2][1], _code(states[0]), out, blocks[0], blocks[1],
            QBLOCK=states[0].blocksize, BLOCK=DEQUANT_BLOCK)
    else:  # not seen with Qwen2.5-1.5B: one weight at a time, with masked tails
        from .nf4 import nf4_dequant_kernel

        start = 0
        for (packed, absmax), qs in zip(sources, states):
            stop = start + int(qs.shape[0])
            n = (stop - start) * cols
            nf4_dequant_kernel[(triton.cdiv(n, DEQUANT_BLOCK),)](
                packed, absmax, _code(qs), out[start:stop], n, QBLOCK=qs.blocksize,
                BLOCK=DEQUANT_BLOCK)
            start = stop
    return out


_CAST = {}


def _cast(tensor, dtype):
    """`tensor` in `dtype`, recast only when the tensor changes (the optimizer updates LoRA's
    weights in place once per step; biases never change)."""
    if tensor.dtype == dtype:
        return tensor
    key = (id(tensor), dtype)
    hit = _CAST.get(key)
    if hit is not None and hit[0] is tensor and hit[1] == tensor._version:
        return hit[2]
    value = tensor.detach().to(dtype)
    _CAST[key] = (tensor, tensor._version, value)
    return value


def _bias(layers, dtype):
    key = ("bias",) + tuple(id(layer) for layer in layers) + (dtype,)
    versions = tuple(layer.bias._version for layer in layers)
    hit = _CAST.get(key)
    if hit is not None and hit[1] == versions:
        return hit[2]
    value = torch.cat([layer.bias.detach().to(dtype) for layer in layers])
    _CAST[key] = (None, versions, value)
    return value


def _rng_get(device):
    return torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()


def _rng_set(device, state):
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    else:
        torch.set_rng_state(state)


def _masks(u, y, p_q, p_v):
    """The two dropout masks the base run draws, in its order (q's LoRA first, then v's)."""
    return unpad._padded_dropout_keep(u, y, p_q), unpad._padded_dropout_keep(u, y, p_v)


_DUMMIES = {}


def _dummy(dtype, device):
    """A one-element tensor for a pointer argument a kernel does not read (one compiled variant
    of the kernel serves every call)."""
    key = (dtype, device)
    if key not in _DUMMIES:
        _DUMMIES[key] = torch.zeros(1, dtype=dtype, device=device)
    return _DUMMIES[key]


def _norm(x, d, weight, eps, dtype, want_y):
    n, cols = x.shape
    y = torch.empty((n, cols), dtype=torch.float32, device=x.device) if want_y else None
    y16 = torch.empty((n, cols), dtype=dtype, device=x.device)
    rstd = torch.empty(n, dtype=torch.float32, device=x.device)
    add_rmsnorm_fwd_kernel[(n,)](x, d if d is not None else _dummy(dtype, x.device), weight,
                                 y if want_y else _dummy(torch.float32, x.device), y16, rstd,
                                 cols, eps, int(d is not None), int(want_y),
                                 BLOCK=triton.next_power_of_2(cols),
                                 num_warps=8 if cols >= 4096 else 4)
    return y, y16, rstd


def _norm_bwd(dy, dq, dv, keep_q, keep_v, rows, x, d, weight, rstd, dres, scale_q, scale_v):
    n, cols = x.shape
    dx = torch.empty_like(x, dtype=torch.float32)
    lora = dq is not None
    half = dy.dtype if dy is not None else (dq.dtype if lora else torch.float16)
    f16 = _dummy(half, x.device)
    flags = _dummy(torch.bool, x.device)
    rmsnorm_bwd_lora_kernel[(n,)](
        dy if dy is not None else f16, dq if lora else f16, dv if lora else f16,
        keep_q if lora else flags, keep_v if lora else flags, rows, x,
        d if d is not None else f16, weight, rstd, dres, dx, cols, scale_q, scale_v,
        int(dy is not None), int(lora), int(d is not None), BLOCK=triton.next_power_of_2(cols),
        num_warps=8 if cols >= 4096 else 4)
    return dx


# ---------------------------------------------------------------- attention core

class _Attention:
    """Causal attention within each row of the unpadded sequence.

    varlen: PyTorch's memory-efficient kernel with cu_seqlens (one launch each way for all rows).
    fallback: scaled_dot_product_attention row by row, its backward pass by autograd."""

    mode = None  # decided by self_check() at first use on CUDA

    @staticmethod
    def forward(q, k, v, u, scale):
        if _Attention.mode == "varlen":
            out, lse, seed, offset, _, _ = torch.ops.aten._efficient_attention_forward(
                q[None], k[None], v[None], None, u.cu_seqlens, u.cu_seqlens, u.max_seqlen,
                u.max_seqlen, 0.0, 1, True, scale=scale)
            return out[0], (lse, seed, offset)
        outs = []
        for start, stop in u.segments:
            outs.append(torch.nn.functional.scaled_dot_product_attention(
                q[None, start:stop].transpose(1, 2), k[None, start:stop].transpose(1, 2),
                v[None, start:stop].transpose(1, 2), is_causal=stop - start > 1,
                scale=scale)[0].transpose(0, 1))
        return torch.cat(outs, 0), None

    @staticmethod
    def backward(dout, q, k, v, out, aux, u, scale):
        if _Attention.mode == "varlen":
            lse, seed, offset = aux
            dq, dk, dv, _ = torch.ops.aten._efficient_attention_backward(
                dout[None], q[None], k[None], v[None], None, out[None], u.cu_seqlens,
                u.cu_seqlens, u.max_seqlen, u.max_seqlen, lse, 0.0, seed, offset, 1, False,
                scale=scale)
            return dq[0], dk[0], dv[0]
        q, k, v = (t.detach().requires_grad_(True) for t in (q, k, v))
        with torch.enable_grad():
            again, _ = _Attention.forward_fallback(q, k, v, u, scale)
        torch.autograd.backward(again, dout)
        return q.grad, k.grad, v.grad

    @staticmethod
    def forward_fallback(q, k, v, u, scale):
        mode, _Attention.mode = _Attention.mode, "fallback"
        try:
            return _Attention.forward(q, k, v, u, scale)
        finally:
            _Attention.mode = mode


def _self_check(device, heads, dim, dtype):
    """Compare the variable-length kernel with plain SDPA (forward and backward) on random data
    from a private generator; the default RNG is left untouched."""
    if device.type != "cuda":
        _Attention.mode = "fallback"
        return
    state = torch.cuda.get_rng_state(device)
    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(1234)
        lengths = [37, 1, 64, 130]
        n = sum(lengths)

        class U:
            pass

        u = U()
        bounds = [0]
        for length in lengths:
            bounds.append(bounds[-1] + length)
        u.segments = list(zip(bounds[:-1], bounds[1:]))
        u.cu_seqlens = torch.tensor(bounds, dtype=torch.int32, device=device)
        u.max_seqlen = max(lengths)
        q, k, v, g = (torch.randn((n, heads, dim), generator=gen, device=device).to(dtype)
                      for _ in range(4))
        scale = dim ** -0.5
        _Attention.mode = "varlen"
        out, aux = _Attention.forward(q, k, v, u, scale)
        grads = _Attention.backward(g, q, k, v, out, aux, u, scale)
        _Attention.mode = "fallback"
        ref, _ = _Attention.forward(q, k, v, u, scale)
        ref_grads = _Attention.backward(g, q, k, v, ref, None, u, scale)
        ok = torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2) and all(
            torch.allclose(a.float(), b.float(), atol=5e-2, rtol=5e-2)
            for a, b in zip(grads, ref_grads))
        same_rng = torch.equal(torch.cuda.get_rng_state(device), state)
        _Attention.mode = "varlen" if ok and same_rng else "fallback"
    except Exception as exc:  # noqa: BLE001 - any problem: the plain path
        import sys

        print(f"[solution] variable-length attention unavailable ({exc.__class__.__name__}); "
              "using SDPA per row", file=sys.stderr)
        _Attention.mode = "fallback"
    finally:
        torch.cuda.set_rng_state(state, device)


# ================================================================================ the layer

MEMORY = {}  # phase -> (largest allocated, largest reserved) in MiB, sampled (for the log)


def sample_memory(phase, device):
    if device.type != "cuda":
        return
    allocated = torch.cuda.memory_allocated(device) / 2**20
    reserved = torch.cuda.memory_reserved(device) / 2**20
    old = MEMORY.get(phase, (0.0, 0.0))
    MEMORY[phase] = (max(old[0], allocated), max(old[1], reserved))


class LayerSpec:
    """What a layer's autograd function needs besides tensors."""

    __slots__ = ("layer", "u", "cos", "sin", "dtype", "recompute_mlp", "index", "weights",
                 "store", "last")


class FusedDecoderLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, aq_w, bq_w, av_w, bv_w, spec):
        layer, u, dtype = spec.layer, spec.u, spec.dtype
        attn, mlp = layer.self_attn, layer.mlp
        qp, kp, vp, op = attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj
        n, hidden = x.shape
        heads, kv_heads, dim = attn.config.num_attention_heads, attn.config.num_key_value_heads, \
            attn.head_dim
        p_q, p_v = qp.lora_dropout["default"].p, vp.lora_dropout["default"].p
        s_q, s_v = float(qp.scaling["default"]), float(vp.scaling["default"])

        ws = spec.weights.acquire(spec.index, spec.index + 1) if spec.weights else None
        w_qkv, w_o, w_gu, w_d = (ws[0:3], ws[3:4], ws[4:6], ws[6:7]) if ws else (None,) * 4

        # attention block
        y, y16, rstd1 = _norm(x, None, layer.input_layernorm.weight,
                              float(layer.input_layernorm.variance_epsilon), dtype, True)
        rng = _rng_get(x.device)
        keep_q, keep_v = _masks(u, y, p_q, p_v)
        drop_q = torch.empty((n, hidden), dtype=dtype, device=x.device)
        drop_v = torch.empty_like(drop_q)
        lora_dropout_pair_kernel[(n, triton.cdiv(hidden, ROW_BLOCK))](
            y, keep_q, keep_v, u.rows, drop_q, drop_v, hidden, 1.0 / (1.0 - p_q),
            1.0 / (1.0 - p_v), BLOCK=ROW_BLOCK)
        del y, keep_q, keep_v
        base = (qp.base_layer, kp, vp.base_layer)
        w = _dequant(base, dtype, w_qkv)
        qkv = torch.addmm(_bias(base, dtype), y16, w.t())
        del w, y16
        a_q = torch.mm(drop_q, _cast(aq_w, dtype).t())
        l_q = torch.mm(a_q, _cast(bq_w, dtype).t())
        a_v = torch.mm(drop_v, _cast(av_w, dtype).t())
        l_v = torch.mm(a_v, _cast(bv_w, dtype).t())
        del drop_q, drop_v
        qe = torch.empty((n, heads, dim), dtype=dtype, device=x.device)
        ke, ve = torch.empty_like(qe), torch.empty_like(qe)
        kc = torch.empty((n, kv_heads, dim), dtype=dtype, device=x.device)
        vc = torch.empty_like(kc)
        qkv_lora_rope_kernel[(n, heads)](qkv, l_q, l_v, spec.cos, spec.sin, qe, ke, ve, kc, vc,
                                         s_q, s_v, N_HEADS=heads, N_KV=kv_heads, HEAD_DIM=dim)
        del qkv, l_q, l_v
        out, aux = _Attention.forward(qe, ke, ve, u, attn.scaling)
        del qe, ke, ve, kc, vc
        # What the backward pass needs goes to host memory as soon as it exists, so the copies
        # run while the rest of the layer computes and the GPU copies are released early.
        offload = spec.store is not None and not spec.last  # the last layer's are needed first
        handle = spec.store.save([x, out]) if offload else None
        o16 = torch.mm(out.reshape(n, heads * dim), _dequant((op,), dtype, w_o).t())

        # MLP block
        _, y2, rstd2 = _norm(x, o16, layer.post_attention_layernorm.weight,
                             float(layer.post_attention_layernorm.variance_epsilon), dtype, False)
        gu = torch.mm(y2, _dequant((mlp.gate_proj, mlp.up_proj), dtype, w_gu).t())
        del y2
        inter = gu.shape[1] // 2
        act = torch.empty((n, inter), dtype=dtype, device=x.device)
        gate_up_act_fwd_kernel[(n, triton.cdiv(inter, ACT_BLOCK))](gu, act, inter, BLOCK=ACT_BLOCK)
        # gate|up (35 KB per token) is not kept on the GPU in "recompute" layers: with the
        # activation store it goes to host memory with the rest (the copy engines are idle
        # enough), otherwise the backward pass recomputes it (one more large GEMM).
        gu_host = spec.recompute_mlp and offload and mlp_to_host(spec.index)
        if gu_host:
            spec.store.save([gu], handle)
        if spec.recompute_mlp or gu_host:
            del gu
            gu = None
        d16 = torch.mm(act, _dequant((mlp.down_proj,), dtype, w_d).t())
        sample_memory("layer forward", x.device)
        del act
        h2 = x + o16  # the residual stream after attention, in fp32 as in the base run
        h2 += d16

        ctx.spec, ctx.rng = spec, rng
        ctx.consts = (p_q, p_v, s_q, s_v, heads, kv_heads, dim, inter)
        ctx.aux = aux
        # Kept for the backward pass: x (fp32) and attention's output; q, k, v and o16 are
        # recomputed there from them (one q|k|v and one o GEMM, a few % of the layer's work) so
        # that 9 KB per token cross PCIe each way instead of 16 KB. gate|up: on the GPU, in
        # host memory, or recomputed.
        ctx.gu_host = gu_host
        ctx.handle = handle
        if offload:
            spec.u.handles[spec.index] = handle
            big = [rstd2, rstd2]  # placeholders: the tensors travel to host memory
        else:
            big = [x, out]
        big.append(gu if gu is not None else rstd2)
        ctx.save_for_backward(*big, a_q, a_v, rstd1, rstd2, aq_w, bq_w, av_w, bv_w)
        return h2

    @staticmethod
    def backward(ctx, dh2):
        (x, out, gu, a_q, a_v, rstd1, rstd2, aq_w, bq_w, av_w, bv_w) = ctx.saved_tensors
        spec, u, dtype = ctx.spec, ctx.spec.u, ctx.spec.dtype
        if spec.store is not None:
            if ctx.handle is not None:
                spec.store.prefetch(ctx.handle)
            if spec.index > 0:  # the next layer of the backward pass, while this one computes
                spec.store.prefetch(u.handles.get(spec.index - 1))
            if ctx.handle is not None:
                loaded = spec.store.load(ctx.handle)
                x, out = loaded[0], loaded[1]
                if ctx.gu_host:
                    gu = loaded[2]
                del loaded
                ctx.handle = None
        recompute = spec.recompute_mlp and not ctx.gu_host
        ws = spec.weights.acquire(spec.index, spec.index - 1) if spec.weights else None
        w_qkv, w_o, w_gu, w_d = (ws[0:3], ws[3:4], ws[4:6], ws[6:7]) if ws else (None,) * 4
        layer = spec.layer
        attn, mlp = layer.self_attn, layer.mlp
        qp, kp, vp, op = attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj
        p_q, p_v, s_q, s_v, heads, kv_heads, dim, inter = ctx.consts
        n, hidden = x.shape
        norm1, norm2 = layer.input_layernorm, layer.post_attention_layernorm
        dh2 = dh2.contiguous()

        # attention's output projection again (the MLP norm needs x + o16)
        wo = _dequant((op,), dtype, w_o)
        o16 = torch.mm(out.reshape(n, heads * dim), wo.t())

        # MLP block
        w_gu_full = None
        if recompute:
            _, y2, _ = _norm(x, o16, norm2.weight, float(norm2.variance_epsilon), dtype, False)
            w_gu_full = _dequant((mlp.gate_proj, mlp.up_proj), dtype, w_gu)
            gu = torch.mm(y2, w_gu_full.t())
            del y2
        d_act = torch.mm(dh2.to(dtype), _dequant((mlp.down_proj,), dtype, w_d))
        # the gradient of gate|up overwrites gate|up (each element is read, then written)
        gate_up_act_bwd_kernel[(n, triton.cdiv(inter, ACT_BLOCK))](d_act, gu, gu, inter,
                                                                   BLOCK=ACT_BLOCK)
        d_gu = gu
        del d_act, gu
        # dy2 = d_gate @ W_gate + d_up @ W_up: one weight dequantized at a time
        if w_gu_full is not None:
            w_gate, w_up = w_gu_full[:inter], w_gu_full[inter:]
            dy2 = torch.mm(d_gu[:, :inter], w_gate)
            dy2.addmm_(d_gu[:, inter:], w_up)
            del w_gate, w_up
        else:
            w_half = _dequant((mlp.gate_proj,), dtype, w_gu[0:1] if w_gu else None)
            dy2 = torch.mm(d_gu[:, :inter], w_half)
            del w_half
            w_half = _dequant((mlp.up_proj,), dtype, w_gu[1:2] if w_gu else None)
            dy2.addmm_(d_gu[:, inter:], w_half)
            del w_half
        sample_memory("layer backward", x.device)
        del d_gu, w_gu_full
        dh1 = _norm_bwd(dy2, None, None, None, None, u.rows, x, o16, norm2.weight, rstd2, dh2,
                        1.0, 1.0)
        del dy2, o16

        # q, k, v again: the norm, the dropout masks (redrawn from the saved RNG state), the
        # q|k|v GEMM, the LoRA outputs from the saved A outputs, the rotary embedding
        y, y16, _ = _norm(x, None, norm1.weight, float(norm1.variance_epsilon), dtype, True)
        now = _rng_get(x.device)
        _rng_set(x.device, ctx.rng)
        keep_q, keep_v = _masks(u, y, p_q, p_v)
        _rng_set(x.device, now)
        drop_q = torch.empty((n, hidden), dtype=dtype, device=x.device)
        drop_v = torch.empty_like(drop_q)
        lora_dropout_pair_kernel[(n, triton.cdiv(hidden, ROW_BLOCK))](
            y, keep_q, keep_v, u.rows, drop_q, drop_v, hidden, 1.0 / (1.0 - p_q),
            1.0 / (1.0 - p_v), BLOCK=ROW_BLOCK)
        del y
        base = (qp.base_layer, kp, vp.base_layer)
        w_attn = _dequant(base, dtype, w_qkv)
        qkv = torch.addmm(_bias(base, dtype), y16, w_attn.t())
        del y16
        aq16, bq16 = _cast(aq_w, dtype), _cast(bq_w, dtype)
        av16, bv16 = _cast(av_w, dtype), _cast(bv_w, dtype)
        l_q = torch.mm(a_q, bq16.t())
        l_v = torch.mm(a_v, bv16.t())
        qe = torch.empty((n, heads, dim), dtype=dtype, device=x.device)
        ke, ve = torch.empty_like(qe), torch.empty_like(qe)
        kc = torch.empty((n, kv_heads, dim), dtype=dtype, device=x.device)
        vc = torch.empty_like(kc)
        qkv_lora_rope_kernel[(n, heads)](qkv, l_q, l_v, spec.cos, spec.sin, qe, ke, ve, kc, vc,
                                         s_q, s_v, N_HEADS=heads, N_KV=kv_heads, HEAD_DIM=dim)
        del qkv, l_q, l_v, kc, vc

        # attention block
        d_out = torch.mm(dh1.to(dtype), wo).view(n, heads, dim)
        del wo
        dqe, dke, dve = _Attention.backward(d_out, qe, ke, ve, out, ctx.aux, u, attn.scaling)
        sample_memory("layer backward (attention)", x.device)
        del d_out, qe, ke, ve, out
        qkv_cols = (heads + 2 * kv_heads) * dim
        d_qkv = torch.empty((n, qkv_cols), dtype=dtype, device=x.device)
        qkv_rope_bwd_kernel[(n, heads + kv_heads)](dqe, dke, dve, spec.cos, spec.sin, d_qkv,
                                                   N_HEADS=heads, N_KV=kv_heads, HEAD_DIM=dim)
        del dqe, dke, dve

        # LoRA gradients
        dl_q = d_qkv[:, : heads * dim] * s_q
        dl_v = d_qkv[:, (heads + kv_heads) * dim:] * s_v
        grad_bq = torch.mm(dl_q.t(), a_q).float()
        grad_bv = torch.mm(dl_v.t(), a_v).float()
        da_q = torch.mm(dl_q, bq16)
        da_v = torch.mm(dl_v, bv16)
        del dl_q, dl_v
        grad_aq = torch.mm(da_q.t(), drop_q).float()
        grad_av = torch.mm(da_v.t(), drop_v).float()
        del drop_q, drop_v

        dx = None
        if ctx.needs_input_grad[0]:
            dd_q = torch.mm(da_q, aq16)
            dd_v = torch.mm(da_v, av16)
            dy = torch.mm(d_qkv, w_attn)
            dx = _norm_bwd(dy, dd_q, dd_v, keep_q, keep_v, u.rows, x, None, norm1.weight, rstd1,
                           dh1, 1.0 / (1.0 - p_q), 1.0 / (1.0 - p_v))
        return dx, grad_aq, grad_bq, grad_av, grad_bv, None


# ================================================================================ wiring

# The MLP's gate|up output (2 * intermediate fp16 per token) is the largest thing a layer keeps
# for its backward pass. A step keeps gate|up in as many layers as fit in this budget and
# recomputes it (one extra GEMM per layer) in the rest. Everything else a layer keeps is about
# 16 KB per token. The peak memory of a step is set by its longest batches, so short batches
# recompute nothing.
ACT_BUDGET_MB = 1300.0
OFFLOAD_ACT_BUDGET_MB = 0.0
KEPT_BYTES_PER_TOKEN_LAYER = 9 * 1024 + 128
# gate|up goes to host memory in every layer (measured on the T4: 6 % faster steps than
# recomputing it in one layer out of three, at the same memory peak). N > 1: recomputed in one
# layer out of N; 0: always recomputed.
MLP_HOST_PERIOD = 1


def mlp_to_host(index):
    if MLP_HOST_PERIOD <= 0:
        return False
    return MLP_HOST_PERIOD == 1 or index % MLP_HOST_PERIOD != MLP_HOST_PERIOD - 1


def host_bytes(n_tokens, n_layers, inter):
    """Pinned host memory the saved tensors of one step take (every layer but the last: x and
    attention's output; gate|up where mlp_to_host), with the 512-byte alignment of each."""
    total = 2**20
    for index in range(n_layers - 1):
        total += KEPT_BYTES_PER_TOKEN_LAYER * n_tokens + 3 * 512
        if mlp_to_host(index):
            total += 4 * inter * n_tokens
    return total


def mlp_layers_to_recompute(n_tokens, n_layers, inter, offloaded=False):
    """How many layers (the first ones) recompute gate|up in this step. With the other saved
    tensors in host memory (offload.py) only gate|up count against the budget, and the default
    budget is 0: every layer recomputes, and a step keeps almost nothing on the GPU."""
    budget = OFFLOAD_ACT_BUDGET_MB if offloaded else ACT_BUDGET_MB
    per_layer = n_tokens * 4 * inter / 2**20
    kept = 0 if offloaded else n_tokens * KEPT_BYTES_PER_TOKEN_LAYER * n_layers / 2**20
    room = budget - kept
    keep = int(max(room, 0.0) // per_layer) if per_layer > 0 else n_layers
    return max(0, n_layers - keep)


_ELIGIBLE = {}


def eligible_cached(layer, on_cuda):
    key = id(layer)
    hit = _ELIGIBLE.get(key)
    if hit is None or hit[0] is not layer:
        hit = _ELIGIBLE[key] = (layer, eligible(layer))
    return hit[1] and (on_cuda or _ALLOW_CPU)


def eligible(layer):
    """The fused path covers this run's layer: q and v with one plain LoRA adapter on frozen
    NF4 weights, k, o, gate, up, down frozen NF4 without LoRA, SiLU, SDPA attention, no
    attention dropout."""
    try:
        from bitsandbytes.nn import Linear4bit
        from peft.tuners.lora.layer import Linear as LoraLinear
    except Exception:  # noqa: BLE001
        return False
    lora_types = [LoraLinear]
    try:
        from peft.tuners.lora.bnb import Linear4bit as LoraLinear4bit

        lora_types.append(LoraLinear4bit)
    except Exception:  # noqa: BLE001 - peft without bitsandbytes support
        pass
    attn, mlp = layer.self_attn, layer.mlp
    for module in (attn.q_proj, attn.v_proj):
        if type(module) not in lora_types or module.disable_adapters or module.merged or \
                list(module.active_adapters) != ["default"] or \
                getattr(module, "lora_variant", {}) or module.use_dora.get("default", False) or \
                type(module.lora_dropout["default"]) is not torch.nn.Dropout or \
                not 0.0 < module.lora_dropout["default"].p < 1.0 or \
                module.lora_dropout["default"].inplace or \
                module.lora_A["default"].bias is not None or \
                module.lora_B["default"].bias is not None or \
                type(module.base_layer) is not Linear4bit:
            return False
    plain = (attn.k_proj, attn.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    if any(type(m) is not Linear4bit for m in plain):
        return False
    base = (attn.q_proj.base_layer, attn.k_proj, attn.v_proj.base_layer, attn.o_proj,
            mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    for m in base:
        qs = getattr(m.weight, "quant_state", None)
        if qs is None or qs.quant_type != "nf4" or m.weight.requires_grad or \
                (qs.nested and qs.state2.blocksize != 256):
            return False
    if any(m.bias is None for m in base[:3]) or any(m.bias is not None for m in base[3:]):
        return False
    if getattr(mlp.config, "hidden_act", "silu") != "silu" or attn.attention_dropout or \
            attn.config._attn_implementation != "sdpa" or attn.head_dim % 2 or \
            layer.input_layernorm.weight.requires_grad or \
            layer.post_attention_layernorm.weight.requires_grad:
        return False
    return True


def run_layer(layer, hidden, u, cos, sin, recompute_mlp, index=0, weights=None, store=None,
              last=False):
    """hidden [N, H] fp32 -> the layer's output, through FusedDecoderLayer."""
    if torch.is_autocast_enabled(hidden.device.type):
        dtype = torch.get_autocast_dtype(hidden.device.type)
    else:
        dtype = hidden.dtype
    if _Attention.mode is None:
        _self_check(hidden.device, layer.self_attn.config.num_attention_heads,
                    layer.self_attn.head_dim, dtype)
    # A layer is set up by bitsandbytes on its first call; do what Linear4bit.forward would.
    attn, mlp = layer.self_attn, layer.mlp
    for module in (attn.q_proj.base_layer, attn.k_proj, attn.v_proj.base_layer):
        if not module.compute_type_is_set:
            module.set_compute_type(hidden)
            module.compute_type_is_set = True
        if module.bias.dtype != hidden.dtype:
            module.bias.data = module.bias.data.to(hidden.dtype)
    spec = LayerSpec()
    spec.layer, spec.u, spec.cos, spec.sin = layer, u, cos, sin
    spec.dtype, spec.recompute_mlp = dtype, recompute_mlp
    spec.index, spec.weights, spec.store, spec.last = index, weights, store, last
    qp, vp = attn.q_proj, attn.v_proj
    with torch.autocast(hidden.device.type, enabled=False):
        return FusedDecoderLayer.apply(hidden, qp.lora_A["default"].weight,
                                       qp.lora_B["default"].weight, vp.lora_A["default"].weight,
                                       vp.lora_B["default"].weight, spec)


# ================================================================================ self-check

# What the fused path runs with, settled once by verify_first_layer():
# attention "varlen" (one memory-efficient call with cu_seqlens) or "fallback" (SDPA row by row),
# saved activations in host memory or not, weights streamed from host memory or fetched per use.
STATE = {"verified": False, "use_fused": True, "attn": None, "store": True, "stream": True}
VERIFY_TOKENS = 384  # the self-check runs on the first tokens of the first batch (a few MB)


def _rel(a, b):
    return (a.float() - b.float()).abs().max().item() / (b.float().abs().max().item() + 1e-12)


def _sub_batch(u, m):
    """The first `m` tokens of the unpadded batch as a batch of its own: whole rows, then a prefix
    of the row cut at `m` (causal attention makes a prefix self-contained)."""
    sub = unpad._Unpadded(u.rows[:m], u.batch, u.seq, m, None)
    segments, bounds = [], [0]
    for start, stop in u.segments:
        if start >= m:
            break
        segments.append((start, min(stop, m)))
        bounds.append(min(stop, m))
    sub.segments = segments
    sub.cu_seqlens = torch.tensor(bounds, dtype=torch.int32).to(u.rows.device)
    sub.max_seqlen = max(stop - start for start, stop in segments)
    sub.handles = {}
    return sub


class _Fp32Linear(torch.autograd.Function):
    """y [N, in] fp32 @ w16 [out, in]^T in fp32, keeping only the fp16 weight for backward."""

    @staticmethod
    def forward(ctx, y, w16):
        ctx.save_for_backward(w16)
        return y @ w16.float().t()

    @staticmethod
    def backward(ctx, grad):
        (w16,) = ctx.saved_tensors
        return grad @ w16.float(), None


def _reference_fp32(layer, x, u, cos, sin, keep_q, keep_v):
    """The layer in plain PyTorch, every activation in fp32 (the weights as the fp16 values both
    runs multiply with): the yardstick for the fused path and for the per-module path."""
    F = torch.nn.functional
    attn, mlp = layer.self_attn, layer.mlp
    qp, kp, vp, op = attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj
    heads, kv_heads, dim = attn.config.num_attention_heads, attn.config.num_key_value_heads, \
        attn.head_dim
    n = x.shape[0]

    def weight(module):  # bitsandbytes' own dequantization (no kernel of ours to compile)
        import bitsandbytes.functional as bnb_f

        qs = module.weight.quant_state
        from .offload import host_ready

        host_ready()
        packed = qs._soup_host.to(x.device) if getattr(qs, "_soup_host", None) is not None \
            else module.weight.data
        return bnb_f.dequantize_4bit(packed.reshape(-1, 1), quant_state=qs).to(torch.float16)

    def norm(h, module):
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + module.variance_epsilon)
        return module.weight.float() * h

    def rope(t):
        half = t.shape[-1] // 2
        rot = torch.cat((-t[..., half:], t[..., :half]), -1)
        return t * cos[:, None] + rot * sin[:, None]

    def lora(module, y, keep):
        p = module.lora_dropout["default"].p
        drop = y * keep.index_select(0, u.rows).to(y.dtype) / (1.0 - p)
        a, b = module.lora_A["default"].weight.float(), module.lora_B["default"].weight.float()
        return (drop @ a.t()) @ b.t() * float(module.scaling["default"])

    y = norm(x, layer.input_layernorm)
    q = _Fp32Linear.apply(y, weight(qp.base_layer)) + qp.base_layer.bias.float() + \
        lora(qp, y, keep_q)
    k = _Fp32Linear.apply(y, weight(kp)) + kp.bias.float()
    v = _Fp32Linear.apply(y, weight(vp.base_layer)) + vp.base_layer.bias.float() + \
        lora(vp, y, keep_v)
    q, k = rope(q.view(n, heads, dim)), rope(k.view(n, kv_heads, dim))
    k = k.repeat_interleave(heads // kv_heads, 1)
    v = v.view(n, kv_heads, dim).repeat_interleave(heads // kv_heads, 1)
    outs = [F.scaled_dot_product_attention(
        q[None, a:b].transpose(1, 2), k[None, a:b].transpose(1, 2), v[None, a:b].transpose(1, 2),
        is_causal=b - a > 1, scale=attn.scaling)[0].transpose(0, 1) for a, b in u.segments]
    h1 = x + _Fp32Linear.apply(torch.cat(outs, 0).reshape(n, heads * dim), weight(op))
    y2 = norm(h1, layer.post_attention_layernorm)
    act = F.silu(_Fp32Linear.apply(y2, weight(mlp.gate_proj))) * \
        _Fp32Linear.apply(y2, weight(mlp.up_proj))
    return h1 + _Fp32Linear.apply(act, weight(mlp.down_proj))


def _errors(out, grads, truth, truth_grads, x, grad):
    names = ("output", "input grad", "lora_A q", "lora_B q", "lora_A v", "lora_B v")
    values = [_rel(out - x, truth - x), _rel(grads[0] - grad, truth_grads[0] - grad)]
    values += [_rel(a, b) if b.abs().max().item() > 0 else (a.abs().max().item() > 0) * 1.0
               for a, b in zip(grads[1:], truth_grads[1:])]
    return dict(zip(names, values))


def verify_first_layer(layer, x, u, cos, sin, weights, store):
    """Once, at the first training step, on the first tokens of its batch: layer 0 through the
    fused path against the same layer computed in fp32 (forward output, input gradient, the
    four LoRA gradients). The first layers of Qwen2.5 have very large q/k biases, which makes
    the q and k gradients of any fp16 attention (the base run's too) noticeably inexact, so the
    limits are set from what the base run's own per-module path shows on the T4.

    Configurations are tried from the fastest/smallest down: variable-length attention or SDPA
    row by row, saved activations in host memory or on the GPU, weights streamed or fetched per
    use. The first one that passes is kept (STATE); if none does, the run keeps the per-module
    path. The CUDA RNG state is restored, so the step draws what it would have drawn."""
    import sys

    import time

    STATE["verified"] = True
    if x.device.type != "cuda":
        STATE["attn"] = _Attention.mode or "fallback"
        return True
    started = time.perf_counter()
    device = x.device
    state = torch.cuda.get_rng_state(device)
    saved_ctx = unpad._ctx()
    log = []
    try:
        if _Attention.mode is None:
            _self_check(device, layer.self_attn.config.num_attention_heads,
                        layer.self_attn.head_dim, torch.float16)
        varlen_ok = _Attention.mode == "varlen"
        m = min(x.shape[0], VERIFY_TOKENS)
        sub = _sub_batch(u, m)
        cos, sin = cos[:m].contiguous(), sin[:m].contiguous()
        gen = torch.Generator(device=device)
        gen.manual_seed(4321)
        xin = x[:m].detach().clone().requires_grad_(True)
        grad = torch.randn(xin.shape, generator=gen, device=device) * 1e-2
        attn = layer.self_attn
        params = [attn.q_proj.lora_A["default"].weight, attn.q_proj.lora_B["default"].weight,
                  attn.v_proj.lora_A["default"].weight, attn.v_proj.lora_B["default"].weight]
        p_q = attn.q_proj.lora_dropout["default"].p
        p_v = attn.v_proj.lora_dropout["default"].p

        # the yardstick: fp32, with the masks the step draws
        torch.cuda.set_rng_state(state, device)
        probe = torch.empty((1, xin.shape[1]), dtype=torch.float32, device=device)
        keep_q, keep_v = _masks(sub, probe, p_q, p_v)
        with torch.autocast("cuda", enabled=False):
            truth = _reference_fp32(layer, xin, sub, cos, sin, keep_q, keep_v)
            truth_grads = torch.autograd.grad(truth, [xin] + params, grad)
        truth = truth.detach()
        del keep_q, keep_v

        # Measured on the Kaggle T4 (first batch of the task): the per-module fp16 path, i.e.
        # the base run's numerics, is this far from fp32: output 3.6e-3, input grad 1.1e-1,
        # lora_B q 4.7e-2, lora_B v 9.4e-3 (the q/k biases of layer 0 are very large). The fused
        # path must stay within about twice that; a wrong kernel is off by far more (a 30 %
        # error in the q gradient shows as 3e-1).
        allowed = {"output": 1e-2, "input grad": 2.5e-1, "lora_A q": 1.2e-1, "lora_B q": 1.2e-1,
                   "lora_A v": 3e-2, "lora_B v": 3e-2}

        configs = []
        for mode in (["varlen"] if varlen_ok else []) + ["fallback"]:
            for keep_host in ([True, False] if store is not None else [False]):
                configs.append((mode, keep_host, weights is not None))
        if weights is not None:
            configs.append(("fallback", False, False))
        chosen = None
        for mode, keep_host, stream in configs:
            name = f"{mode} attention, activations {'host' if keep_host else 'GPU'}, " \
                   f"weights {'streamed' if stream else 'fetched'}"
            try:
                _Attention.mode = mode
                sub.handles = {}
                if keep_host:
                    store.begin_step(host_bytes(m, 2, layer.mlp.config.intermediate_size))
                torch.cuda.set_rng_state(state, device)
                out = run_layer(layer, xin, sub, cos, sin, True, index=0,
                                weights=weights if stream else None,
                                store=store if keep_host else None, last=False)
                grads = torch.autograd.grad(out, [xin] + params, grad)
                errs = _errors(out.detach(), grads, truth, truth_grads, xin, grad)
                del out, grads
                bad = [k for k in errs if not errs[k] <= allowed[k]]
                log.append(f"fused ({name}) " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items())
                           + (f" -> differs: {', '.join(bad)}" if bad else " -> ok"))
                if not bad:
                    chosen = (mode, keep_host, stream)
                    break
            except Exception as exc:  # noqa: BLE001
                log.append(f"fused ({name}) failed: {exc!r}")
        if chosen is None:
            STATE["use_fused"] = False
            _Attention.mode = "varlen" if varlen_ok else "fallback"
            log.append("no fused configuration matched: per-module path")
        else:
            STATE["attn"], STATE["store"], STATE["stream"] = chosen
            _Attention.mode = chosen[0]
            log.append(f"fused path: {chosen[0]} attention, activations "
                       f"{'in host memory' if chosen[1] else 'on the GPU'}, weights "
                       f"{'streamed' if chosen[2] else 'fetched per use'}")
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        log.append(f"self-check failed ({exc!r}): per-module path")
        STATE["use_fused"] = False
    finally:
        unpad._LOCAL.ctx = saved_ctx
        torch.cuda.set_rng_state(state, device)
        torch.cuda.empty_cache()  # the check's fp32 copies must not stay reserved
    log.append(f"took {time.perf_counter() - started:.2f} s")
    for line in log:
        print(f"[solution] fused layer self-check: {line}", file=sys.stderr)
    return STATE["use_fused"]
