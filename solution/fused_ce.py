"""The output head and the causal-LM loss without the [batch, seq, vocab] logits.

Base Soup builds the logits of all 4 x 512 positions over Qwen's 151,936-token vocabulary
(fp16 from the autocast matmul, then an fp32 copy from accelerate's output conversion, another in
the loss's upcast, the softmax and its gradient: several 1.2 GB tensors at once), and TRL and Soup
then read them again for the entropy, the token accuracy and the per-row losses of rewind.jsonl.

Here the head runs on chunks of positions. For each chunk cuBLAS computes the fp16 logits (the same
matmul autocast runs in the base), and one Triton kernel per row makes a single online pass over
the vocabulary for the max, the log-sum-exp, the entropy, the argmax and the label's logit, and a
second pass that writes the gradient of the loss, softmax - one_hot, back over the logits in place.
A second cuBLAS matmul turns that into the gradient of the chunk's hidden states, so the backward
pass of the head is done during the forward pass and no logits outlive their chunk (the idea of
Liger Kernel's and Unsloth's fused linear cross-entropy; the code here is ours).

Only positions that are not padding go through the head at all: their entropy is the only thing
the metrics read from padded positions, and TRL masks it out.

The loss is the same sum of per-token cross-entropies over the same fp16 logits, divided by the
same denominator; entropy, accuracy and the per-row losses follow TRL's and Soup's formulas.
"""

import threading
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100
ROWS_PER_CHUNK = 256  # 256 x 151,936 fp16 logits: 78 MB live at a time
VOCAB_BLOCK = 2048
# The gradient over the logits is stored scaled by this power of two, in the logits' dtype; the
# true gradient is recovered in fp32 in the backward pass. It keeps small probabilities above the
# fp16 subnormal range (the base gets the same from the loss scaler) without risking overflow.
GRAD_SCALE = 4096.0
STEP_ATTR = "_soup_chunked_step"  # where Soup's rewind recorder reads a step without logits
_ALLOW_CPU = False  # tests only: run the fused path on CPU tensors (TRITON_INTERPRET=1)


@triton.jit(do_not_specialize=["first_row"])
def ce_row_stats_kernel(logits_ptr, stride_row, first_row, target_ptr, entropy_ptr, loss_ptr,
                        correct_ptr, grad_scale, N_VOCAB: tl.constexpr, BLOCK_V: tl.constexpr,
                        WRITE_GRAD: tl.constexpr):
    """One row of logits: entropy, cross-entropy at the target, whether the argmax is the target,
    and (WRITE_GRAD) grad_scale * (softmax - one_hot(target)) written over the logits."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + row * stride_row  # the chunk's logits buffer
    row = row + first_row                    # the row in the per-row outputs
    target = tl.load(target_ptr + row)
    valid = target != -100

    run_max = tl.full([], float("-inf"), tl.float32)
    run_sum = tl.full([], 0.0, tl.float32)      # sum of exp(x - run_max)
    run_dot = tl.full([], 0.0, tl.float32)      # sum of exp(x - run_max) * x
    best = tl.full([], float("-inf"), tl.float32)
    best_idx = tl.full([], 0, tl.int32)
    for start in range(0, N_VOCAB, BLOCK_V):
        cols = start + tl.arange(0, BLOCK_V)
        in_row = cols < N_VOCAB
        x = tl.load(row_ptr + cols, mask=in_row, other=float("-inf")).to(tl.float32)
        block_max = tl.max(x, axis=0)
        block_arg = tl.argmax(x, axis=0, tie_break_left=True)
        new_max = tl.maximum(run_max, block_max)
        rescale = tl.exp(run_max - new_max)
        e = tl.exp(x - new_max)
        run_sum = run_sum * rescale + tl.sum(e, axis=0)
        run_dot = run_dot * rescale + tl.sum(e * tl.where(in_row, x, 0.0), axis=0)
        better = block_max > best  # strict: the first maximum wins, as torch.argmax
        best_idx = tl.where(better, start + block_arg, best_idx)
        best = tl.where(better, block_max, best)
        run_max = new_max

    lse = run_max + tl.log(run_sum)
    entropy = lse - run_dot / run_sum
    safe_target = tl.where(valid, target, 0)
    target_logit = tl.load(row_ptr + safe_target).to(tl.float32)
    tl.store(entropy_ptr + row, entropy)
    tl.store(loss_ptr + row, tl.where(valid, lse - target_logit, 0.0))
    tl.store(correct_ptr + row, ((best_idx == target) & valid).to(tl.int32))

    if WRITE_GRAD:
        scale = tl.where(valid, grad_scale, 0.0)
        for start in range(0, N_VOCAB, BLOCK_V):
            cols = start + tl.arange(0, BLOCK_V)
            in_row = cols < N_VOCAB
            x = tl.load(row_ptr + cols, mask=in_row, other=0.0).to(tl.float32)
            grad = tl.exp(x - lse) - (cols == target).to(tl.float32)
            tl.store(row_ptr + cols, (grad * scale).to(logits_ptr.dtype.element_ty), mask=in_row)


_WEIGHT_CACHE = {}


def _head_weight(weight, dtype):
    """The frozen head weight in the compute dtype; the fp16 copy autocast would make on every
    call, made once."""
    if weight.dtype == dtype:
        return weight
    key = (weight.data_ptr(), weight._version, tuple(weight.shape), dtype)
    cached = _WEIGHT_CACHE.get("head")
    if cached is None or cached[0] != key:
        _WEIGHT_CACHE.clear()
        _WEIGHT_CACHE["head"] = (key, weight.detach().to(dtype).contiguous())
    return _WEIGHT_CACHE["head"][1]


def _rows_through_head(hidden, weight16, targets, with_grad, n_grad):
    """Stats for the selected rows, chunk by chunk, and the scaled hidden-state gradient of the
    first `n_grad` rows (the labelled ones, sorted first; the others have zero gradient)."""
    n_rows, n_vocab = hidden.shape[0], weight16.shape[0]
    device = hidden.device
    entropy = torch.empty(n_rows, dtype=torch.float32, device=device)
    token_loss = torch.empty(n_rows, dtype=torch.float32, device=device)
    correct = torch.empty(n_rows, dtype=torch.int32, device=device)
    grad = torch.empty((n_grad, hidden.shape[1]), dtype=hidden.dtype, device=device) \
        if with_grad else None
    if n_rows == 0:
        return entropy, token_loss, correct, grad
    chunk = min(ROWS_PER_CHUNK, n_rows)
    n_chunks = triton.cdiv(n_rows, chunk)
    chunk = triton.cdiv(n_rows, n_chunks)  # even chunks
    logits = torch.empty((chunk, n_vocab), dtype=weight16.dtype, device=device)
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        part = logits[: stop - start]
        torch.mm(hidden[start:stop], weight16.t(), out=part)
        write_grad = with_grad and start < n_grad
        ce_row_stats_kernel[(stop - start,)](
            part, part.stride(0), start, targets, entropy, token_loss, correct, GRAD_SCALE,
            N_VOCAB=n_vocab, BLOCK_V=VOCAB_BLOCK, WRITE_GRAD=write_grad, num_warps=8)
        if write_grad:
            end = min(stop, n_grad)
            torch.mm(part[: end - start], weight16, out=grad[start:end])
    return entropy, token_loss, correct, grad


class FusedLinearCrossEntropy(torch.autograd.Function):
    """sum over labelled positions of cross_entropy(hidden @ weight.T) / denominator, with the
    gradient on `hidden` computed in the forward pass. The weight must be frozen. `rows` lists
    the labelled positions first (`n_grad` of them): only they get a gradient."""

    @staticmethod
    def forward(ctx, hidden, weight, targets, rows, n_grad, denominator, compute_dtype,
                stream=None):
        shape = hidden.shape
        flat = hidden.detach().reshape(-1, shape[-1])
        picked = flat.index_select(0, rows).to(compute_dtype).contiguous()
        with_grad = ctx.needs_input_grad[0]
        picked_targets = targets.index_select(0, rows).contiguous()
        if stream is not None:  # the head weight lives in host memory (head_stream.py)
            from .head_stream import rows_through_streamed_head

            entropy, token_loss, correct, grad = rows_through_streamed_head(
                picked, stream, picked_targets, with_grad, n_grad, GRAD_SCALE)
        else:
            weight16 = _head_weight(weight.detach(), compute_dtype)
            entropy, token_loss, correct, grad = _rows_through_head(
                picked, weight16, picked_targets, with_grad, n_grad)
        loss = token_loss.sum() / denominator
        ctx.shape, ctx.dtype = shape, hidden.dtype
        if with_grad:
            ctx.save_for_backward(grad, rows[:n_grad], denominator)
        return loss, entropy, token_loss, correct

    @staticmethod
    def backward(ctx, grad_loss, *_unused):
        grad, rows, denominator = ctx.saved_tensors
        factor = (grad_loss.float() / (denominator.float() * GRAD_SCALE))
        grad_hidden = torch.zeros((ctx.shape.numel() // ctx.shape[-1], ctx.shape[-1]),
                                  dtype=ctx.dtype, device=grad.device)
        grad_hidden.index_copy_(0, rows, (grad.float() * factor).to(ctx.dtype))
        return grad_hidden.view(ctx.shape), None, None, None, None, None, None, None


def fused_linear_cross_entropy(hidden, weight, labels, attention_mask=None,
                               num_items_in_batch=None):
    """The causal-LM loss of `hidden @ weight.T` against `labels` (as transformers'
    ForCausalLMLoss computes it), and the per-position terms TRL's and Soup's metrics need."""
    batch, seq = hidden.shape[:2]
    device = hidden.device
    targets = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
    targets = targets.to(device).reshape(-1).contiguous()
    if attention_mask is not None:
        rows = attention_mask.to(device).reshape(-1).nonzero().squeeze(1)
    else:
        rows = torch.arange(batch * seq, device=device)
    labelled = (targets != IGNORE_INDEX).sum()
    # Labelled positions first: only they need the gradient pass and its GEMM.
    has_label = targets.index_select(0, rows) != IGNORE_INDEX
    with_label, without = rows[has_label], rows[~has_label]
    n_grad = with_label.numel()
    rows = torch.cat([with_label, without])
    if num_items_in_batch is None:
        denominator = labelled.float()
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(device).float()
    else:
        denominator = torch.tensor(float(num_items_in_batch), device=device)
    if torch.is_autocast_enabled(device.type):
        compute_dtype = torch.get_autocast_dtype(device.type)
    else:
        compute_dtype = torch.result_type(hidden, weight)
    from . import head_stream as hs

    host = hs.host_of(weight)
    if host is not None and not hs.VERIFIED["done"] and device.type == "cuda":
        picked = hidden.detach().reshape(-1, hidden.shape[-1]).index_select(0, rows)
        hs.verify(weight, host, device, picked.to(torch.float16), GRAD_SCALE)
        host = hs.host_of(weight)
    stream = hs.head_stream(host, device) if host is not None else None
    loss, entropy, token_loss, correct = FusedLinearCrossEntropy.apply(
        hidden, weight, targets, rows, n_grad, denominator, compute_dtype, stream)

    def spread(values):
        full = torch.zeros(batch * seq, dtype=values.dtype, device=device)
        full.index_copy_(0, rows, values)
        return full.view(batch, seq)

    step = SimpleNamespace(token_entropy=spread(entropy), token_loss=spread(token_loss),
                           correct=correct.sum(), labelled=labelled)
    return loss, step


# --- wiring into transformers' Qwen2 and TRL's SFTTrainer -------------------------------------

_LOCAL = threading.local()


def _capture():
    return getattr(_LOCAL, "capture", None)


def patch_qwen2_causal_lm():
    """With labels, inside our trainer step, Qwen2ForCausalLM returns the loss and no logits."""
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2ForCausalLM
    if getattr(cls.forward, "_soup_patched", False):
        return
    original = cls.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        capture = _capture()
        head = self.lm_head
        if capture is None or labels is None or not self.training or \
                not isinstance(logits_to_keep, int) or logits_to_keep != 0 or \
                "shift_labels" in kwargs or type(head) is not torch.nn.Linear or \
                head.bias is not None or head.weight.requires_grad or \
                getattr(self.config, "final_logit_softcapping", None) is not None:
            from .head_stream import materialized

            with materialized(head.weight):
                return original(self, input_ids=input_ids, attention_mask=attention_mask,
                                position_ids=position_ids, past_key_values=past_key_values,
                                inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                                logits_to_keep=logits_to_keep, **kwargs)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        hidden = outputs.last_hidden_state
        if not (hidden.is_cuda or _ALLOW_CPU):
            from .head_stream import materialized

            capture.step = None
            with materialized(head.weight):
                logits = head(hidden)
            loss = self.loss_function(logits=logits, labels=labels,
                                      vocab_size=self.config.vocab_size, **kwargs)
            return CausalLMOutputWithPast(loss=loss, logits=logits,
                                          past_key_values=outputs.past_key_values,
                                          hidden_states=outputs.hidden_states,
                                          attentions=outputs.attentions)
        loss, capture.step = fused_linear_cross_entropy(
            hidden, head.weight, labels, attention_mask, kwargs.get("num_items_in_batch"))
        return CausalLMOutputWithPast(loss=loss, past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    forward._soup_patched = True
    cls.forward = forward


def _causal_lm_of(model):
    model = getattr(model, "module", model)
    get_base_model = getattr(model, "get_base_model", None)
    return get_base_model() if callable(get_base_model) else model


def _log_trl_metrics(trainer, mode, inputs, step):
    """`entropy`, `num_tokens` and `mean_token_accuracy` exactly as TRL 0.29.1 logs them."""
    gather = trainer.accelerator.gather_for_metrics
    metrics = trainer._metrics
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        entropy = torch.sum(step.token_entropy * attention_mask) / attention_mask.sum()
    else:
        entropy = torch.mean(step.token_entropy)
    metrics[mode]["entropy"].append(gather(entropy).mean().item())
    if mode == "train":
        if attention_mask is not None:
            tokens = gather(attention_mask.sum()).sum().item()
        else:
            local = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
            tokens = gather(local).sum().item()
        trainer._total_train_tokens += tokens
    metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    correct = gather(step.correct)
    total = gather(step.labelled)
    total_sum = total.sum()
    accuracy = (correct.sum() / total_sum).item() if total_sum > 0 else 0.0
    metrics[mode]["mean_token_accuracy"].append(accuracy)


def patch_sft_trainer():
    """TRL's SFTTrainer.compute_loss for a training step of a Qwen2 causal LM: the model runs the
    fused head, and the metrics TRL would read from the logits come from the fused pass."""
    import transformers
    from trl import SFTTrainer

    if getattr(SFTTrainer.compute_loss, "_soup_patched", False):
        return
    original = SFTTrainer.compute_loss
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        eligible = (
            isinstance(model, torch.nn.Module) and getattr(self, "model", None) is not None
            and getattr(self.model, "training", False) and "labels" in inputs
            and "shift_labels" not in inputs and "attention_mask" in inputs
            and not getattr(self.args, "use_liger_kernel", False)
            and getattr(self, "num_virtual_tokens", 0) == 0
            and not getattr(self, "aux_loss_enabled", False)
            and self.label_smoother is None and self.compute_loss_func is None
            and isinstance(_causal_lm_of(model), Qwen2ForCausalLM)
            and hasattr(self, "_metrics") and hasattr(self, "_total_train_tokens")
        )
        if not eligible:
            setattr(self, STEP_ATTR, None)
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        from .embed import shrink_tied_embedding
        from .warmup import wait

        wait()  # our kernels finished compiling while the model loaded
        if getattr(self.args, "fp16", False):
            shrink_tied_embedding(self, _causal_lm_of(model))
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        capture = SimpleNamespace(step=None)
        _LOCAL.capture = capture
        try:
            loss, outputs = transformers.Trainer.compute_loss(
                self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        finally:
            _LOCAL.capture = None
        step = capture.step
        if step is None:  # the model took its own path and returned logits
            step = _step_from_logits(outputs.logits, inputs["labels"])
        setattr(self, STEP_ATTR, step)
        _log_trl_metrics(self, "train", inputs, step)
        return (loss, outputs) if return_outputs else loss

    compute_loss._soup_patched = True
    SFTTrainer.compute_loss = compute_loss


def _step_from_logits(logits, labels):
    """The same per-position terms from full logits (a fallback that should not be reached)."""
    from trl.trainer.utils import entropy_from_logits

    with torch.no_grad():
        targets = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
        labelled = targets != IGNORE_INDEX
        token_loss = torch.nn.functional.cross_entropy(
            logits.float().transpose(1, 2), targets.clamp(min=0), reduction="none") * labelled
        correct = ((logits.argmax(-1) == targets) & labelled).sum()
        return SimpleNamespace(token_entropy=entropy_from_logits(logits), token_loss=token_loss,
                               correct=correct, labelled=labelled.sum())
