"""Padding-free decoder layers for Qwen2 in our training steps.

TRL pads every batch of 4 rows to its longest row. A row averages about 220 tokens and a batch's
longest is often 400 to 512, so a large share of every linear layer, norm, MLP and residual add
works on padding. The rules allow removing pad tokens inside a batch (not mixing batches).

Here the decoder runs on the real tokens only, as one [1, N, hidden] sequence. Each token keeps the
position id and rotary embedding it has in the padded batch. Attention is the one place where rows
must not see each other: there q, k and v are scattered back to the padded [B, T] layout and the
same attention function runs with the same mask as in the base run, so every real token attends
to exactly the keys it attends to in the base; the output is gathered back to the real tokens.
After the final norm the hidden states are scattered back to [B, T, hidden] for the head.

LoRA dropout keeps the base run's random masks: the mask is drawn by the same dropout call on a
tensor of the padded shape (the same RNG draw, element for element) and the real tokens' rows of it
are applied with a Triton kernel of ours. Pad positions never reach the loss, the metrics or the
LoRA gradients (their labels are -100 and their attention mask is 0).
"""

import threading

import torch
import triton
import triton.language as tl

_LOCAL = threading.local()
LAST_TOKENS = None  # (real, padded) tokens of the last batch, for the debug log
_ALLOW_CPU = False  # tests only: run on CPU tensors (TRITON_INTERPRET=1)


def _ctx():
    return getattr(_LOCAL, "ctx", None)


class _Unpadded:
    def __init__(self, rows, batch, seq, n_real, segments):
        self.rows, self.batch, self.seq, self.n = rows, batch, seq, n_real
        # [(start, stop)] of each row's tokens in the unpadded order when every row's real tokens
        # are one contiguous run (right or left padding); None otherwise.
        self.segments = segments


# --- LoRA dropout with the base run's masks ----------------------------------------------------

@triton.jit
def rows_masked_scale_kernel(x_ptr, keep_ptr, rows_ptr, out_ptr, n_cols, scale,
                             BLOCK: tl.constexpr):
    """out[i] = x[i] * keep[rows[i]] * scale, one real-token row per program: that row's slice of
    a dropout mask drawn over the padded batch, applied to the unpadded activations. The backward
    pass is the same operation on the gradient."""
    row = tl.program_id(0).to(tl.int64)
    src = tl.load(rows_ptr + row).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    keep = tl.load(keep_ptr + src * n_cols + cols, mask=mask, other=0)
    out = x * keep.to(x.dtype) * scale
    tl.store(out_ptr + row * n_cols + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


ROW_BLOCK = 1024


class _RowDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, keep, rows, scale):
        n_cols = x.shape[-1]
        x2 = x.reshape(-1, n_cols).contiguous()
        out = torch.empty_like(x2)
        grid = (x2.shape[0], triton.cdiv(n_cols, ROW_BLOCK))
        rows_masked_scale_kernel[grid](x2, keep, rows, out, n_cols, scale, BLOCK=ROW_BLOCK)
        ctx.save_for_backward(keep, rows)
        ctx.scale, ctx.shape = scale, x.shape
        return out.view(x.shape)

    @staticmethod
    def backward(ctx, grad):
        keep, rows = ctx.saved_tensors
        n_cols = grad.shape[-1]
        g2 = grad.reshape(-1, n_cols).contiguous()
        out = torch.empty_like(g2)
        grid = (g2.shape[0], triton.cdiv(n_cols, ROW_BLOCK))
        rows_masked_scale_kernel[grid](g2, keep, rows, out, n_cols, ctx.scale, BLOCK=ROW_BLOCK)
        return out.view(ctx.shape), None, None, None


_ONES = {}


def _ones(numel, dtype, device):
    """A reusable all-ones buffer (only its size matters to the random draw)."""
    key = (dtype, device)
    buf = _ONES.get(key)
    if buf is None or buf.numel() < numel:
        buf = torch.ones(max(numel, 1), dtype=dtype, device=device)
        _ONES[key] = buf
    return buf[:numel]


def _padded_dropout_keep(ctx, x, p):
    """The boolean keep mask the base run's dropout draws on the padded [B, T, hidden] input.
    On CUDA, F.dropout is at::native_dropout; calling it on a tensor of the same size and dtype
    draws the same random numbers, and it hands back the mask as bool directly."""
    numel = ctx.batch * ctx.seq * x.shape[-1]
    if x.is_cuda:
        _, keep = torch.native_dropout(_ones(numel, x.dtype, x.device), p, True)
    else:  # CPU (tests): F.dropout takes the bernoulli path there
        probe = torch.ones(numel, dtype=x.dtype, device=x.device)
        keep = torch.nn.functional.dropout(probe, p, True) != 0
    return keep.view(ctx.batch * ctx.seq, -1)


def patch_dropout():
    cls = torch.nn.Dropout
    if getattr(cls.forward, "_soup_patched", False):
        return
    original = cls.forward

    def forward(self, x):
        ctx = _ctx()
        if ctx is None or not self.training or self.inplace or not 0.0 < self.p < 1.0 or \
                x.dim() != 3 or x.shape[0] != 1 or x.shape[1] != ctx.n or \
                not (x.is_cuda or _ALLOW_CPU):
            return original(self, x)
        keep = _padded_dropout_keep(ctx, x, self.p)
        return _RowDropout.apply(x, keep, ctx.rows, 1.0 / (1.0 - self.p))

    forward._soup_patched = True
    cls.forward = forward


# --- the decoder on real tokens -----------------------------------------------------------------

def _scatter(values, rows, batch, seq):
    """[N, ...] -> [B, T, ...] with zeros at pad positions (differentiable)."""
    full = values.new_zeros((batch * seq, *values.shape[1:]))
    return full.index_copy(0, rows, values).view(batch, seq, *values.shape[1:])


def patch_qwen2_attention():
    from transformers.models.qwen2 import modeling_qwen2 as mq

    cls = mq.Qwen2Attention
    if getattr(cls.forward, "_soup_patched", False):
        return
    original = cls.forward

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                **kwargs):
        ctx = _ctx()
        if ctx is None or past_key_values is not None or hidden_states.dim() != 3 or \
                hidden_states.shape[0] != 1 or hidden_states.shape[1] != ctx.n:
            return original(self, hidden_states, position_embeddings, attention_mask,
                            past_key_values=past_key_values, **kwargs)
        n, d = ctx.n, self.head_dim
        query = self.q_proj(hidden_states).view(1, n, -1, d).transpose(1, 2)
        key = self.k_proj(hidden_states).view(1, n, -1, d).transpose(1, 2)
        value = self.v_proj(hidden_states).view(1, n, -1, d).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = mq.apply_rotary_pos_emb(query, key, cos, sin)

        interface = mq.ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation,
                                                             mq.eager_attention_forward)
        dropout = 0.0 if not self.training else self.attention_dropout
        if ctx.segments is not None and not self.attention_dropout:
            # Each row on its own, causal, with no mask: a real token sees the same keys as under
            # the base run's padding mask (its row's real tokens up to itself), and no work is
            # spent on padding. k and v are expanded to all query heads (repeat_kv, as the base
            # run's masked SDPA call does) so PyTorch keeps its memory-efficient kernel; SDPA
            # with enable_gqa would fall back to the slow fp32 math kernel on a T4.
            groups = self.num_key_value_groups
            dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else \
                query.dtype
            q = query.to(dtype)
            k = mq.repeat_kv(key.to(dtype), groups)
            v = mq.repeat_kv(value.to(dtype), groups)
            outs = []
            for start, stop in ctx.segments:
                out = torch.nn.functional.scaled_dot_product_attention(
                    q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop],
                    is_causal=stop - start > 1, scale=self.scaling)
                outs.append(out)
            out = torch.cat(outs, dim=2).transpose(1, 2).reshape(1, n, -1)
            return self.o_proj(out), None

        def padded(states):  # [1, heads, N, d] -> [B, heads, T, d]
            return _scatter(states[0].transpose(0, 1), ctx.rows, ctx.batch, ctx.seq).transpose(1, 2)

        out, weights = interface(self, padded(query), padded(key), padded(value), attention_mask,
                                 dropout=dropout,
                                 scaling=self.scaling, sliding_window=self.sliding_window,
                                 **kwargs)
        out = out.reshape(ctx.batch * ctx.seq, -1).index_select(0, ctx.rows).unsqueeze(0)
        return self.o_proj(out), weights

    forward._soup_patched = True
    cls.forward = forward


def _segments(attention_mask, config):
    """Each row's run of real tokens in the unpadded order, if every row has one contiguous run
    and every layer is plain causal full attention; None otherwise (then attention runs padded)."""
    if getattr(config, "_attn_implementation", None) != "sdpa" or \
            any(t != "full_attention" for t in getattr(config, "layer_types", None) or []) or \
            getattr(config, "use_sliding_window", False):
        return None
    mask = attention_mask.detach().to("cpu", torch.bool)
    segments, start = [], 0
    for row in mask:
        idx = row.nonzero().flatten()
        count = idx.numel()
        if count == 0 or int(idx[-1]) - int(idx[0]) + 1 != count:
            return None
        segments.append((start, start + count))
        start += count
    return segments


def patch_qwen2_model():
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2 as mq

    cls = mq.Qwen2Model
    if getattr(cls.forward, "_soup_patched", False):
        return
    original = cls.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        from .fused_ce import _capture

        if _capture() is None or not self.training or input_ids is None or \
                inputs_embeds is not None or past_key_values is not None or use_cache or \
                position_ids is not None or attention_mask is None or \
                attention_mask.dim() != 2 or not (input_ids.is_cuda or _ALLOW_CPU) or \
                _ctx() is not None or \
                any(getattr(layer, "gradient_checkpointing", False) for layer in self.layers):
            # (a recomputed layer would run outside this context; this run does not checkpoint)
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        batch, seq = input_ids.shape
        rows = attention_mask.reshape(-1).nonzero().squeeze(1)
        n_real = rows.numel()  # one sync, as TRL's own .item() calls on the mask
        global LAST_TOKENS
        LAST_TOKENS = (n_real, batch * seq)
        if n_real == batch * seq or n_real == 0:
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)

        # Everything the base forward builds on the padded batch, built the same way.
        inputs_embeds = self.embed_tokens(input_ids)
        position_ids = torch.arange(seq, device=inputs_embeds.device).unsqueeze(0)
        mask_kwargs = {"config": self.config, "inputs_embeds": inputs_embeds,
                       "attention_mask": attention_mask, "past_key_values": None,
                       "position_ids": position_ids}
        masks = {"full_attention": mq.create_causal_mask(**mask_kwargs)}
        if self.has_sliding_layers:
            masks["sliding_attention"] = mq.create_sliding_window_causal_mask(**mask_kwargs)
        cos, sin = self.rotary_emb(inputs_embeds, position_ids)  # [1, T, d]

        # ... then the real tokens only.
        token_pos = rows % seq
        hidden = inputs_embeds.reshape(batch * seq, -1).index_select(0, rows).unsqueeze(0)
        position_embeddings = (cos[0].index_select(0, token_pos).unsqueeze(0),
                               sin[0].index_select(0, token_pos).unsqueeze(0))
        _LOCAL.ctx = _Unpadded(rows, batch, seq, n_real, _segments(attention_mask, self.config))
        try:
            for i, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                hidden = layer(hidden, attention_mask=masks[self.config.layer_types[i]],
                               position_embeddings=position_embeddings,
                               position_ids=position_ids, past_key_values=None,
                               use_cache=use_cache, **kwargs)
        finally:
            _LOCAL.ctx = None
        hidden = self.norm(hidden)
        return BaseModelOutputWithPast(last_hidden_state=_scatter(hidden[0], rows, batch, seq),
                                       past_key_values=None)

    forward._soup_patched = True
    cls.forward = forward
