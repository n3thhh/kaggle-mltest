"""Our kernels against plain PyTorch (and bitsandbytes), forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


# --- the organizers' example (kept running in our solution) ------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


# --- nf4.py ---------------------------------------------------------------------------------

def _quantized(n, k, seed=0):
    import bitsandbytes.functional as bnb_f

    torch.manual_seed(seed)
    w = (torch.randn(n, k, device="cuda") * 0.05).half()
    packed, state = bnb_f.quantize_4bit(w, blocksize=64, compress_statistics=True,
                                        quant_type="nf4")
    return packed, state


@pytest.mark.parametrize("shape", [(1536, 1536), (256, 1536), (8960, 1536), (1536, 8960)])
def test_nf4_dequant_is_bitsandbytes_bit_for_bit(shape):
    import bitsandbytes.functional as bnb_f

    from solution.nf4 import dequantize_nf4

    packed, state = _quantized(*shape)
    ref = bnb_f.dequantize_4bit(packed, state).half()
    assert torch.equal(dequantize_nf4(packed.view(-1), state, torch.float16), ref)


def test_nf4_linear_forward_backward():
    import bitsandbytes.functional as bnb_f

    from solution.nf4 import NF4Linear

    packed, state = _quantized(256, 1536)
    w = bnb_f.dequantize_4bit(packed, state).half()
    x = torch.randn(3, 100, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
    bias = torch.randn(256, device="cuda", dtype=torch.float16)
    grad = torch.randn(3, 100, 256, device="cuda", dtype=torch.float16)
    out = NF4Linear.apply(x, packed.view(-1), state, bias)
    out.backward(grad)
    assert torch.equal(out, torch.nn.functional.linear(x.detach(), w, bias))
    assert torch.equal(x.grad, torch.matmul(grad, w))


# --- fused_ce.py ----------------------------------------------------------------------------

def test_fused_head_and_loss():
    from solution.fused_ce import fused_linear_cross_entropy

    torch.manual_seed(0)
    batch, seq, hidden, vocab = 3, 70, 256, 5003
    h = torch.randn(batch, seq, hidden, device="cuda", requires_grad=True)
    w = torch.randn(vocab, hidden, device="cuda") * 0.05
    labels = torch.randint(0, vocab, (batch, seq), device="cuda")
    labels[:, :10] = -100
    mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
    mask[2, 50:] = 0
    labels[2, 50:] = -100
    grad_scale = 1000.0  # the loss scaler's role

    with torch.autocast("cuda", dtype=torch.float16):
        loss, step = fused_linear_cross_entropy(h, w, labels, mask)
    (loss * grad_scale).backward()
    ours = h.grad.clone()
    h.grad = None

    with torch.autocast("cuda", dtype=torch.float16):
        logits = torch.nn.functional.linear(h, w)  # fp16, as autocast computes it in the base
    logits = logits.float()
    target = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
    ref = torch.nn.functional.cross_entropy(logits.view(-1, vocab), target.reshape(-1),
                                            ignore_index=-100)
    (ref * grad_scale).backward()
    torch.testing.assert_close(loss, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ours, h.grad, rtol=2e-3, atol=2e-3)

    logp = torch.log_softmax(logits.detach(), -1)
    entropy = -(logp.exp() * logp).sum(-1)
    torch.testing.assert_close(step.token_entropy * mask, entropy * mask, rtol=1e-4, atol=1e-4)
    labelled = target != -100
    assert step.labelled.item() == labelled.sum().item()
    assert step.correct.item() == ((logits.argmax(-1) == target) & labelled).sum().item()
    token_loss = torch.nn.functional.cross_entropy(logits.detach().transpose(1, 2),
                                                   target.clamp(min=0), reduction="none")
    torch.testing.assert_close(step.token_loss, token_loss * labelled, rtol=1e-4, atol=1e-4)


# --- rmsnorm.py -----------------------------------------------------------------------------

def test_rmsnorm_both_outputs_and_gradients():
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(0)
    x = (torch.randn(5, 33, 1536, device="cuda") * 3).requires_grad_(True)
    w = torch.rand(1536, device="cuda") + 0.5
    a = torch.randn_like(x)
    b = torch.randn_like(x).half()

    def ref(x):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        return w * h.to(x.dtype)

    r = ref(x)
    ((r * a).sum() + (r.half().float() * b.float()).sum()).backward()
    expected = x.grad.clone()
    x.grad = None
    y = rmsnorm(x, w, 1e-6, torch.float16)
    ((y * a).sum() + (y._soup_half.float() * b.float()).sum()).backward()
    torch.testing.assert_close(y, r, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y._soup_half, r.half(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x.grad, expected, rtol=1e-4, atol=1e-4)


# --- rope.py --------------------------------------------------------------------------------

def test_rope_expand_matches_transformers():
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    from solution.rope import RopeExpand

    torch.manual_seed(0)
    n, heads, kv, dim = 300, 12, 2, 128
    q = torch.randn(n, heads, dim, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(n, kv, dim, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(n, kv, dim, device="cuda", dtype=torch.float16, requires_grad=True)
    angle = torch.rand(n, dim // 2, device="cuda") * 50
    cos = torch.cat([angle.cos(), angle.cos()], -1)
    sin = torch.cat([angle.sin(), angle.sin()], -1)
    gq, gk, gv = (torch.randn(heads, n, dim, device="cuda", dtype=torch.float16) for _ in range(3))

    rq, rk = apply_rotary_pos_emb(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                  cos[None], sin[None])
    rq, rk = rq.half()[0], repeat_kv(rk.half(), heads // kv)[0]
    rv = repeat_kv(v.transpose(0, 1)[None], heads // kv)[0]
    torch.autograd.backward((rq, rk, rv), (gq, gk, gv))
    expected = [t.grad.clone() for t in (q, k, v)]
    q.grad = k.grad = v.grad = None

    oq, ok, ov = RopeExpand.apply(q, k, v, cos, sin, torch.float16)
    torch.autograd.backward((oq, ok, ov), (gq, gk, gv))
    for ours, ref in ((oq, rq), (ok, rk), (ov, rv)):
        torch.testing.assert_close(ours, ref, rtol=2e-3, atol=2e-3)
    for t, ref in zip((q, k, v), expected):
        torch.testing.assert_close(t.grad, ref, rtol=2e-3, atol=2e-2)


# --- unpad.py -------------------------------------------------------------------------------

def test_row_dropout_applies_the_padded_mask():
    from solution.unpad import _RowDropout

    torch.manual_seed(0)
    rows = torch.tensor([0, 1, 2, 5, 6, 9], device="cuda")
    keep = torch.rand(10, 1536, device="cuda") > 0.05
    x = torch.randn(1, 6, 1536, device="cuda", requires_grad=True)
    g = torch.randn(1, 6, 1536, device="cuda")
    out = _RowDropout.apply(x, keep, rows, 1 / 0.95)
    out.backward(g)
    mask = keep[rows].float()[None]
    torch.testing.assert_close(out, x.detach() * mask * (1 / 0.95))
    torch.testing.assert_close(x.grad, g * mask * (1 / 0.95))


# --- mlp.py ---------------------------------------------------------------------------------

def test_recomputed_mlp_matches_unfused():
    import bitsandbytes.functional as bnb_f

    from solution.mlp import RecomputedMLP

    gate, gate_s = _quantized(1024, 512, 1)
    up, up_s = _quantized(1024, 512, 2)
    down, down_s = _quantized(512, 1024, 3)
    wg, wu, wd = (bnb_f.dequantize_4bit(p, s).half() for p, s in
                  ((gate, gate_s), (up, up_s), (down, down_s)))
    x = torch.randn(1, 200, 512, device="cuda", dtype=torch.float16, requires_grad=True)
    g = torch.randn(1, 200, 512, device="cuda", dtype=torch.float16)
    ref = torch.nn.functional.linear(
        (torch.nn.functional.silu(torch.nn.functional.linear(x, wg).float())
         * torch.nn.functional.linear(x, wu).float()).half(), wd)
    ref.backward(g)
    expected = x.grad.clone()
    x.grad = None
    out = RecomputedMLP.apply(x, gate.view(-1), gate_s, up.view(-1), up_s, down.view(-1), down_s)
    out.backward(g)
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.grad, expected, rtol=5e-3, atol=5e-3)


# --- layer.py + offload.py: a whole tiny model against the stock modules -------------------

def _tiny_qlora(seed=0):
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=512, hidden_size=256, intermediate_size=512,
                      num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                      tie_word_embeddings=True, attn_implementation="sdpa")
    model = Qwen2ForCausalLM(cfg).half()

    def quantize(linear):
        q = bnb.nn.Linear4bit(linear.in_features, linear.out_features,
                              bias=linear.bias is not None, compute_dtype=torch.float16,
                              quant_type="nf4", compress_statistics=True)
        q.weight = bnb.nn.Params4bit(linear.weight.data.clone(), requires_grad=False,
                                     quant_type="nf4", compress_statistics=True, blocksize=64)
        if linear.bias is not None:
            q.bias = torch.nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        return q.cuda()

    for layer in model.model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                              (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                setattr(parent, name, quantize(getattr(parent, name)))
    model.cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.parameters():  # as prepare_model_for_kbit_training does
        if p.dtype == torch.float16 and type(p).__name__ != "Params4bit":
            p.data = p.data.float()
    try:  # peft asks for torchao when a layer is not bnb-dispatched; Kaggle's torchao is too old
        import peft.tuners.lora.torchao as peft_torchao

        peft_torchao.is_torchao_available = lambda: False
    except Exception:  # noqa: BLE001
        pass
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1,
                                             target_modules=["q_proj", "v_proj"]))
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.02)
    model.train()
    batch, seq = 3, 40
    ids = torch.randint(0, 512, (batch, seq), device="cuda")
    mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
    mask[0, 25:] = 0
    mask[2, 9:] = 0
    labels = ids.clone()
    labels[:, :5] = -100
    labels[mask == 0] = -100
    return model, ids, mask, labels


def _loss_and_grads(model, ids, mask, labels, fused):
    from types import SimpleNamespace

    import solution.fused_ce as fc

    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    model.zero_grad(set_to_none=True)
    fc._LOCAL.capture = SimpleNamespace(step=None) if fused else None
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    finally:
        fc._LOCAL.capture = None
    loss.backward()
    grads = {n: p.grad.float().clone() for n, p in model.named_parameters() if p.grad is not None}
    return loss.item(), grads, torch.cuda.get_rng_state()


@pytest.mark.parametrize("offloaded", [False, True])
def test_fused_layer_matches_stock_modules(offloaded):
    import solution
    import solution.layer as layer
    import solution.offload as offload

    if offloaded:
        offload.patch_quantize()  # weights quantized from here on live in host memory
    model, ids, mask, labels = _tiny_qlora()
    if offloaded:
        qs = model.base_model.model.model.layers[0].mlp.down_proj.weight.quant_state
        assert offload.is_offloaded(qs)
    loss0, grads0, rng0 = _loss_and_grads(model, ids, mask, labels, fused=False)
    solution.apply()
    loss1, grads1, rng1 = _loss_and_grads(model, ids, mask, labels, fused=True)
    assert all(v[1] for v in layer._ELIGIBLE.values()), "the fused layer was not used"
    assert torch.equal(rng0, rng1), "dropout drew different random numbers"
    assert abs(loss0 - loss1) <= 2e-3 * abs(loss0)
    for name, g in grads0.items():
        scale = g.abs().max().item() + 1e-12
        assert (grads1[name] - g).abs().max().item() <= 3e-2 * scale, name


def test_int8_weights_and_activations():
    """layer.py's int8 operands: the NF4 weight to int8 (per-row scales, and transposed with
    per-column scales), a row to int8, and torch._int_mm on them against fp32 GEMMs."""
    import bitsandbytes.functional as bnb_f

    import solution.layer as layer

    class Fake:  # what _int8_rows / _int8_t read from a Linear4bit
        def __init__(self, packed, state):
            self.weight = packed
            self.weight.quant_state = state

    rows, cols = 512, 1536
    (p0, s0), (p1, s1) = _quantized(rows, cols, 1), _quantized(rows, cols, 2)
    w = torch.cat([bnb_f.dequantize_4bit(p0, s0), bnb_f.dequantize_4bit(p1, s1)]).float()
    gate, up = Fake(p0, s0), Fake(p1, s1)
    q, s = layer._int8_rows((gate, up), None)
    torch.testing.assert_close(s, w.abs().amax(1) / 127, rtol=1e-3, atol=0)
    assert (q.float() * s[:, None] - w).abs().max() <= 0.6 * s.max()
    cs = w.abs().amax(0) / 127
    qt = layer._int8_t((gate, up), None, cs)
    assert qt.shape == (cols, 2 * rows)
    assert (qt.float().t() * cs[None, :] - w).abs().max() <= 0.6 * cs.max()
    x = torch.randn(300, cols, device="cuda").half()
    x8, sx = layer._row_quant(x)
    y = torch._int_mm(x8, q.t()).float() * sx[:, None] * s[None, :]
    ref = x.float() @ w.t()
    assert (y - ref).abs().max() <= 2e-2 * ref.abs().max()
    dy = torch.randn(300, 2 * rows, device="cuda")
    d8, sd = layer._row_quant(dy)
    dx = torch._int_mm(d8, qt.t()).float() * sd[:, None] * cs[None, :]
    ref = dy @ w
    assert (dx - ref).abs().max() <= 2e-2 * ref.abs().max()


# --- head_stream.py ---------------------------------------------------------------------------

def test_streamed_head_matches_resident():
    import solution.fused_ce as fc
    import solution.head_stream as hs

    torch.manual_seed(0)
    n, hidden, vocab = 300, 256, 20011  # three chunks, a ragged last one
    h = (torch.randn(n, hidden, device="cuda") * 0.5).half()
    w = (torch.randn(vocab, hidden, device="cuda") * 0.05).half()
    t = torch.randint(0, vocab, (n,), device="cuda")
    t[200:] = -100
    ref = fc._rows_through_head(h, w, t, True, 200)
    stream = hs.HeadStream(w.cpu().pin_memory(), torch.device("cuda"), chunk=8192)
    got = hs.rows_through_streamed_head(h, stream, t, True, 200, fc.GRAD_SCALE)
    torch.testing.assert_close(got[0], ref[0], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(got[1], ref[1], rtol=1e-3, atol=1e-3)
    assert (got[2] != ref[2]).sum().item() <= 2
    torch.testing.assert_close(got[3], ref[3].float(), rtol=2e-2, atol=2e-2 * 4096)
