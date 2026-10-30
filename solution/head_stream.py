"""The tied embedding / output head in pinned host memory: the loss over vocabulary chunks.

Qwen2.5-1.5B's tied embedding is 151,936 x 1,536 fp16 = 467 MB, the largest thing left on the GPU
once the 4-bit weights live in host memory (offload.py). Here it lives in pinned host memory too:

* the input embedding gathers the batch's rows on the CPU and sends those (a few MB);
* the head streams the weight through two GPU slots of 8,192 vocabulary rows, on a copy stream,
  one chunk ahead of compute, in one pass. Each chunk: every token times the chunk, folded into
  running per-token statistics (max, sum of exp, sum of exp * logit, argmax, the target's logit)
  by `ce_chunk_stats_kernel`, which then, for the labelled tokens, overwrites the logits with
  exp(logit - running max), the target's column zeroed; that times the chunk is added to an fp32
  accumulator rescaled by exp(old max - new max) (online softmax, as flash attention does for its
  output). `target_rows_kernel` picks each token's target row of the weight. At the end,
  `ce_finalize_kernel` gives log-sum-exp, entropy, cross-entropy and accuracy, and
  `ce_grad_finalize_kernel` the gradient sum_{j != t} p_j w_j - (1 - p_t) w_t. Keeping the target
  out of the accumulated sum avoids the cancellation of sum_j p_j w_j - w_t when p_t is near 1.

The weight crosses PCIe once per step (0.47 GB) on a copy engine while the GEMMs run.
"""

import torch
import triton
import triton.language as tl

CHUNK = 8192
STAT_BLOCK = 2048


@triton.jit(do_not_specialize=["n_cols", "col0", "n_grad"])
def ce_chunk_stats_kernel(logits_ptr, stride_row, n_cols, col0, target_ptr, m_ptr, s_ptr, d_ptr,
                          best_ptr, bidx_ptr, tlogit_ptr, alpha_ptr, n_grad,
                          MAX_COLS: tl.constexpr, BLOCK_V: tl.constexpr, FIRST: tl.constexpr,
                          WRITE_E: tl.constexpr):
    """Fold one vocabulary chunk's logits of one token into its running statistics. With
    WRITE_E, for the first n_grad (labelled) tokens, the logits are then overwritten with
    exp(logit - running max) (the target's column set to 0) for the gradient GEMM, and
    alpha = exp(old max - new max) is stored to rescale the gradient accumulated so far."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * stride_row
    target = tl.load(target_ptr + row)
    if FIRST:
        run_max = tl.full([], float("-inf"), tl.float32)
        run_sum = tl.full([], 0.0, tl.float32)
        run_dot = tl.full([], 0.0, tl.float32)
        best = tl.full([], float("-inf"), tl.float32)
        best_idx = tl.full([], 0, tl.int64)
    else:
        run_max = tl.load(m_ptr + row)
        run_sum = tl.load(s_ptr + row)
        run_dot = tl.load(d_ptr + row)
        best = tl.load(best_ptr + row)
        best_idx = tl.load(bidx_ptr + row)
    old_max = run_max
    for start in range(0, MAX_COLS, BLOCK_V):  # the last chunk is shorter: masked
        cols = start + tl.arange(0, BLOCK_V)
        in_row = cols < n_cols
        x = tl.load(row_ptr + cols, mask=in_row, other=float("-inf")).to(tl.float32)
        block_max = tl.max(x, axis=0)
        block_arg = tl.argmax(x, axis=0, tie_break_left=True)
        new_max = tl.maximum(run_max, block_max)
        rescale = tl.exp(run_max - new_max)
        e = tl.exp(x - new_max)
        run_sum = run_sum * rescale + tl.sum(e, axis=0)
        run_dot = run_dot * rescale + tl.sum(e * tl.where(in_row, x, 0.0), axis=0)
        better = block_max > best  # strict: the first maximum wins, as torch.argmax
        best_idx = tl.where(better, col0 + start + block_arg, best_idx)
        best = tl.where(better, block_max, best)
        run_max = new_max
    tl.store(m_ptr + row, run_max)
    tl.store(s_ptr + row, run_sum)
    tl.store(d_ptr + row, run_dot)
    tl.store(best_ptr + row, best)
    tl.store(bidx_ptr + row, best_idx)
    here = (target >= col0) & (target < col0 + n_cols)
    offset = tl.where(here, target - col0, 0)
    value = tl.load(row_ptr + offset).to(tl.float32)
    tl.store(tlogit_ptr + row, value, mask=here)
    if WRITE_E:
        if row < n_grad:
            tl.store(alpha_ptr + row, tl.exp(old_max - run_max))
            for start in range(0, MAX_COLS, BLOCK_V):
                cols = start + tl.arange(0, BLOCK_V)
                in_row = cols < n_cols
                x = tl.load(row_ptr + cols, mask=in_row, other=float("-inf")).to(tl.float32)
                e = tl.where(col0 + cols == target, 0.0, tl.exp(x - run_max))
                tl.store(row_ptr + cols, e.to(logits_ptr.dtype.element_ty), mask=in_row)


@triton.jit(do_not_specialize=["n_cols", "col0"])
def target_rows_kernel(w_ptr, n_cols, col0, target_ptr, out_ptr, HIDDEN: tl.constexpr,
                       BLOCK_H: tl.constexpr):
    """out[row] = w[target - col0] for the tokens whose target is in this vocabulary chunk."""
    row = tl.program_id(0).to(tl.int64)
    target = tl.load(target_ptr + row)
    if (target >= col0) & (target < col0 + n_cols):
        src = w_ptr + (target - col0) * HIDDEN
        for start in range(0, HIDDEN, BLOCK_H):
            cols = start + tl.arange(0, BLOCK_H)
            mask = cols < HIDDEN
            tl.store(out_ptr + row * HIDDEN + cols, tl.load(src + cols, mask=mask), mask=mask)


@triton.jit(do_not_specialize=["n_rows"])
def ce_grad_finalize_kernel(acc_ptr, wt_ptr, s_ptr, lse_ptr, tlogit_ptr, n_rows, grad_scale,
                            HIDDEN: tl.constexpr, BLOCK_H: tl.constexpr):
    """grad = grad_scale * (sum_{j != t} p_j w_j - (1 - p_t) w_t), from the accumulated
    sum_{j != t} exp(l_j - max) w_j (acc), the softmax denominator and the target's row."""
    row = tl.program_id(0).to(tl.int64)
    inv_s = 1.0 / tl.load(s_ptr + row)
    p_t = tl.exp(tl.load(tlogit_ptr + row) - tl.load(lse_ptr + row))
    for start in range(0, HIDDEN, BLOCK_H):
        cols = start + tl.arange(0, BLOCK_H)
        mask = cols < HIDDEN
        acc = tl.load(acc_ptr + row * HIDDEN + cols, mask=mask, other=0.0)
        wt = tl.load(wt_ptr + row * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(acc_ptr + row * HIDDEN + cols,
                 (acc * inv_s - (1.0 - p_t) * wt) * grad_scale, mask=mask)


@triton.jit
def rescale_add_kernel(acc_ptr, part_ptr, alpha_ptr, HIDDEN: tl.constexpr,
                       BLOCK_H: tl.constexpr):
    """acc[row] = acc[row] * alpha[row] + part[row] (fp32 accumulator, fp16 chunk product)."""
    row = tl.program_id(0).to(tl.int64)
    alpha = tl.load(alpha_ptr + row)
    for start in range(0, HIDDEN, BLOCK_H):
        cols = start + tl.arange(0, BLOCK_H)
        mask = cols < HIDDEN
        acc = tl.load(acc_ptr + row * HIDDEN + cols, mask=mask, other=0.0)
        part = tl.load(part_ptr + row * HIDDEN + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(acc_ptr + row * HIDDEN + cols, acc * alpha + part, mask=mask)


@triton.jit(do_not_specialize=["n_rows"])
def ce_finalize_kernel(target_ptr, m_ptr, s_ptr, d_ptr, bidx_ptr, tlogit_ptr, entropy_ptr,
                       loss_ptr, correct_ptr, lse_ptr, n_rows, BLOCK: tl.constexpr):
    """Per token: log-sum-exp, entropy (lse - sum(p * x)), cross-entropy at the target, and
    whether the argmax is the target."""
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = rows < n_rows
    target = tl.load(target_ptr + rows, mask=mask, other=-100)
    valid = target != -100
    m = tl.load(m_ptr + rows, mask=mask, other=0.0)
    s = tl.load(s_ptr + rows, mask=mask, other=1.0)
    d = tl.load(d_ptr + rows, mask=mask, other=0.0)
    lse = m + tl.log(s)
    tl.store(lse_ptr + rows, lse, mask=mask)
    tl.store(entropy_ptr + rows, lse - d / s, mask=mask)
    t = tl.load(tlogit_ptr + rows, mask=mask, other=0.0)
    tl.store(loss_ptr + rows, tl.where(valid, lse - t, 0.0), mask=mask)
    best = tl.load(bidx_ptr + rows, mask=mask, other=0)
    tl.store(correct_ptr + rows, ((best == target) & valid).to(tl.int32), mask=mask)


class HeadStream:
    """Vocabulary chunks of the host-resident head weight, two GPU slots, a copy stream."""

    def __init__(self, host, device, chunk=CHUNK):
        self.host, self.device = host, device
        vocab, hidden = host.shape
        self.chunks = [(c0, min(c0 + chunk, vocab)) for c0 in range(0, vocab, chunk)]
        self.cuda = device.type == "cuda"
        if self.cuda:
            self.stream = torch.cuda.Stream(device)
            self.slots = [{"idx": None, "buf": torch.empty((chunk, hidden), dtype=host.dtype,
                                                            device=device),
                           "ready": torch.cuda.Event()} for _ in range(2)]

    def _load(self, slot, j):
        c0, c1 = self.chunks[j]
        free = torch.cuda.Event()
        free.record(torch.cuda.current_stream(self.device))
        self.stream.wait_event(free)
        with torch.cuda.stream(self.stream):
            slot["buf"][: c1 - c0].copy_(self.host[c0:c1], non_blocking=True)
        slot["ready"].record(self.stream)
        slot["idx"] = j

    def get(self, j, prefetch):
        c0, c1 = self.chunks[j]
        if not self.cuda:
            return self.host[c0:c1]
        here = next((s for s in self.slots if s["idx"] == j), None)
        if here is None:
            spare = [s for s in self.slots if s["idx"] != prefetch]
            here = spare[0] if spare else self.slots[0]
            self._load(here, j)
        torch.cuda.current_stream(self.device).wait_event(here["ready"])
        if prefetch is not None and 0 <= prefetch < len(self.chunks):
            other = self.slots[1] if here is self.slots[0] else self.slots[0]
            if other["idx"] != prefetch:
                self._load(other, prefetch)
        return here["buf"][: c1 - c0]


_STREAMS = {}


def head_stream(host, device):
    key = (host.data_ptr(), str(device))
    if key not in _STREAMS:
        _STREAMS[key] = HeadStream(host, device)
    return _STREAMS[key]


def rows_through_streamed_head(hidden, stream, targets, with_grad, n_grad, grad_scale):
    """hidden [n, H] (rows with labels first, `n_grad` of them), the head in host memory ->
    entropy, token loss, correct (per row) and grad_scale * d(sum of losses)/d(hidden) of the
    first n_grad rows, in fp32. One pass over the vocabulary: each chunk crosses PCIe once."""
    n, hidden_size = hidden.shape
    device = hidden.device
    f32 = torch.float32
    m, s, d, best = (torch.empty(n, dtype=f32, device=device) for _ in range(4))
    bidx = torch.empty(n, dtype=torch.int64, device=device)
    tlogit = torch.zeros(n, dtype=f32, device=device)
    entropy, token_loss, lse = (torch.empty(n, dtype=f32, device=device) for _ in range(3))
    correct = torch.empty(n, dtype=torch.int32, device=device)
    grad = None
    k = n_grad if with_grad else 0
    if with_grad:
        grad = torch.zeros((n_grad, hidden_size), dtype=f32, device=device)
    if n == 0:
        return entropy, token_loss, correct, grad
    if k > 0:
        alpha = torch.empty(k, dtype=f32, device=device)
        wt = torch.zeros((k, hidden_size), dtype=hidden.dtype, device=device)
        part16 = torch.empty((k, hidden_size), dtype=hidden.dtype, device=device)
    else:
        alpha = m  # unused
    count = len(stream.chunks)
    width = max(c1 - c0 for c0, c1 in stream.chunks)
    width = triton.cdiv(width, STAT_BLOCK) * STAT_BLOCK
    logits = torch.empty((n, width), dtype=hidden.dtype, device=device)
    block_h = triton.next_power_of_2(hidden_size)
    for j in range(count):
        c0, c1 = stream.chunks[j]
        w = stream.get(j, j + 1 if j + 1 < count else 0)  # then the next step's first
        part = logits[:, : c1 - c0]
        torch.mm(hidden, w.t(), out=part)
        ce_chunk_stats_kernel[(n,)](part, part.stride(0), c1 - c0, c0, targets, m, s, d, best,
                                    bidx, tlogit, alpha, k, MAX_COLS=logits.shape[1],
                                    BLOCK_V=STAT_BLOCK, FIRST=j == 0, WRITE_E=k > 0,
                                    num_warps=8)
        if k > 0:
            torch.mm(part[:k], w, out=part16)
            rescale_add_kernel[(k,)](grad, part16, alpha, HIDDEN=hidden_size, BLOCK_H=block_h)
            target_rows_kernel[(k,)](w, c1 - c0, c0, targets, wt, HIDDEN=hidden_size,
                                     BLOCK_H=block_h)
    from .layer import sample_memory

    sample_memory("head", device)
    ce_finalize_kernel[(triton.cdiv(n, 1024),)](targets, m, s, d, bidx, tlogit, entropy,
                                                token_loss, correct, lse, n, BLOCK=1024)
    if k > 0:
        ce_grad_finalize_kernel[(k,)](grad, wt, s, lse, tlogit, k, grad_scale,
                                      HIDDEN=hidden_size, BLOCK_H=block_h)
    return entropy, token_loss, correct, grad


# --- the embedding side ------------------------------------------------------------------------

def offload_tied_embedding(embedding, device):
    """Keep the (fp16, tied) embedding weight in pinned host memory, leaving a zero-stride
    placeholder of its shape on `device`."""
    weight = embedding.weight
    host = torch.empty(weight.shape, dtype=weight.dtype, pin_memory=True)
    host.copy_(weight.data)
    weight._soup_host = host
    weight.data = torch.zeros(1, dtype=weight.dtype, device=device).as_strided(
        weight.shape, (0,) * weight.dim())
    return host


def host_of(weight):
    return getattr(weight, "_soup_host", None)


def embed_from_host(weight, ids):
    """F.embedding for a weight kept in host memory: the rows are gathered on the CPU."""
    host = weight._soup_host
    flat = ids.reshape(-1).to("cpu")
    rows = host.index_select(0, flat)
    return rows.to(weight.device, non_blocking=False).view(*ids.shape, host.shape[1])


class materialized:
    """The weight back on its device for a block of code outside our paths (rare)."""

    def __init__(self, weight):
        self.weight = weight

    def __enter__(self):
        host = host_of(self.weight)
        self.placeholder = None
        if host is not None:
            self.placeholder = self.weight.data
            self.weight.data = host.to(self.placeholder.device)
        return self.weight

    def __exit__(self, *exc):
        if self.placeholder is not None:
            self.weight.data = self.placeholder
        return False


VERIFIED = {"done": False}


def _plain_head(h, w, targets, k, grad_scale):
    """The same quantities in plain fp32 PyTorch (the yardstick of verify())."""
    logits = h.float() @ w.float().t()
    lse = torch.logsumexp(logits, -1)
    p = torch.softmax(logits, -1)
    entropy = lse - (p * logits).sum(-1)
    valid = targets != -100
    t = targets.clamp(min=0)
    loss = torch.where(valid, lse - logits.gather(1, t[:, None])[:, 0], 0.0)
    correct = ((logits.argmax(-1) == targets) & valid).to(torch.int32)
    g = p[:k].clone()
    g[torch.arange(k, device=h.device), t[:k]] -= 1.0
    return entropy, loss, correct, (g * grad_scale) @ w.float()


def verify(weight, host, device, hidden, grad_scale):
    """Once, at the first step: the streamed head against plain fp32 PyTorch on the first 8,192
    vocabulary rows (two chunks of 4,096, so the slot logic runs) and up to 64 of the step's
    tokens. Any difference, or any error, and the weight goes back to the GPU for good."""
    import sys

    VERIFIED["done"] = True
    try:
        sub = min(8192, host.shape[0])
        gen = torch.Generator(device=device)
        gen.manual_seed(99)
        h = hidden[: min(64, hidden.shape[0])].contiguous()
        targets = torch.randint(0, sub, (h.shape[0],), generator=gen, device=device)
        targets[h.shape[0] // 2:] = -100
        k = h.shape[0] // 2
        resident = host[:sub].to(device)
        ref = _plain_head(h, resident, targets, k, grad_scale)
        got = rows_through_streamed_head(h, HeadStream(host[:sub], device, chunk=sub // 2),
                                         targets, True, k, grad_scale)
        del resident
        errors = []
        for name, a, b in zip(("entropy", "loss", "grad"), (got[0], got[1], got[3]),
                              (ref[0], ref[1], ref[3])):
            rel = (a.float() - b.float()).abs().max().item() / \
                (b.float().abs().max().item() + 1e-6)
            if not rel <= 2e-2:
                errors.append(f"{name} {rel:.1e}")
        if not torch.equal(got[2], ref[2]) and \
                (got[2] != ref[2]).float().mean().item() > 0.05:
            errors.append("argmax")
        if not errors:
            return True
        print(f"[solution] streamed head differs ({', '.join(errors)}): head back on the GPU",
              file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"[solution] streamed head check failed ({exc!r}): head back on the GPU",
              file=sys.stderr)
    weight.data = host.to(device)
    weight._soup_host = None
    return False
