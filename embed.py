"""One fp16 copy of Qwen's tied embedding / output head instead of fp32 + fp16.

The model is loaded in fp16. `prepare_model_for_kbit_training` then upcasts every fp16 parameter
that is not 4-bit to fp32, the tied embedding included: 933 MB, made while the 467 MB fp16 copy
still exists, so loading peaks at 1.4 GB above the weights. `patch_prepare_kbit` keeps that one
weight out of the upcast (it shows peft an empty fp32 placeholder while peft loops over the
parameters), so it stays fp16, and the embedding returns its rows upcast to fp32: the numbers the
fp32 weight gives. The output head runs under fp16 autocast in the base run, which casts the fp32
weight to these same fp16 values on every call. `shrink_tied_embedding` below is the fallback for
a run where the weight was upcast anyway.

`prepare_model_for_kbit_training` upcasts the tied embedding (151,936 x 1,536, also the output
head) to fp32: 933 MB that the run reads in two ways. The input embedding looks up a few thousand
rows and hands them on in fp32; the output head runs under fp16 autocast, so every step it reads
an fp16 copy of the weight (the copy our fused head keeps, 467 MB more).

The model is loaded in fp16 and upcast, so every fp32 value is an fp16 value: we check that, row
block by row block, and only then keep the weight in fp16 for the training steps. The input
embedding then returns its rows upcast to fp32 (the same numbers as before), and the head reads
the weight directly. After training the fp32 weight is put back, so whatever runs after the
trainer sees the model as Soup built it. If any value is not exactly an fp16 value, nothing changes.
"""

import torch

_UPCAST_ATTR = "_soup_upcast_dtype"


def patch_embedding():
    cls = torch.nn.Embedding
    if getattr(cls.forward, "_soup_patched", False):
        return
    original = cls.forward

    def forward(self, input):  # noqa: A002 - the signature of nn.Embedding.forward
        from .head_stream import embed_from_host, host_of

        if host_of(self.weight) is not None and self.padding_idx is None and \
                self.max_norm is None:
            out = embed_from_host(self.weight, input)  # the weight lives in host memory
        else:
            out = original(self, input)
        dtype = getattr(self, _UPCAST_ATTR, None)
        return out if dtype is None else out.to(dtype)

    forward._soup_patched = True
    cls.forward = forward


def _model_device(model):
    """The GPU the model computes on (its first parameter that is not the embedding)."""
    embedding = model.get_input_embeddings()
    skip = getattr(embedding, "weight", None)
    for param in model.parameters():
        if param is not skip and param.device.type != "meta":
            return param.device
    return None


def patch_prepare_kbit():
    import peft

    original = peft.prepare_model_for_kbit_training
    if getattr(original, "_soup_patched", False):
        return

    def prepare_model_for_kbit_training(model, *args, **kwargs):
        from . import mark

        mark("prepare_model_for_kbit_training starts")
        try:
            return _prepare(model, *args, **kwargs)
        finally:
            mark("prepare_model_for_kbit_training done")

    def _prepare(model, *args, **kwargs):
        try:
            embedding = model.get_input_embeddings()
            head = model.get_output_embeddings()
        except Exception:  # noqa: BLE001
            embedding = head = None
        weight = getattr(embedding, "weight", None)
        device = _model_device(model)
        if type(embedding) is not torch.nn.Embedding or weight is None or \
                weight.dtype != torch.float16 or device is None or device.type != "cuda" or \
                (head is not None and getattr(head, "weight", None) is not weight and
                 getattr(head, "weight", None) is not None):
            result = original(model, *args, **kwargs)
            # loaded to host memory (offload.patch_loading) but not handled here: to the GPU
            if weight is not None and device is not None and weight.device != device and \
                    weight.device.type == "cpu":
                weight.data = weight.data.to(device)
            return result
        real = weight.data
        weight.data = torch.empty(0, dtype=torch.float32, device=device)
        try:
            result = original(model, *args, **kwargs)
        finally:
            weight.data = real
        setattr(embedding, _UPCAST_ATTR, torch.float32)
        from . import offload

        offloaded = False
        if offload.ENABLED and offload.HEAD and embedding.padding_idx is None:
            try:
                from .head_stream import offload_tied_embedding

                offload_tied_embedding(embedding, device)  # also the head: the weight is tied
                offloaded = True
            except Exception as exc:  # noqa: BLE001 - keep it on the GPU
                import sys

                print(f"[solution] embedding offload skipped: {exc!r}", file=sys.stderr)
        if not offloaded and not weight.is_cuda:  # loaded to host memory but kept: to the GPU
            weight.data = weight.data.to(device)
        return result

    prepare_model_for_kbit_training._soup_patched = True
    peft.prepare_model_for_kbit_training = prepare_model_for_kbit_training
    try:
        import peft.utils.other as other

        other.prepare_model_for_kbit_training = prepare_model_for_kbit_training
    except Exception:  # noqa: BLE001
        pass


def _exact_in(weight, dtype, rows=8192):
    for block in weight.detach().split(rows):
        if not torch.equal(block.to(dtype).to(weight.dtype), block):
            return False
    return True


def shrink_tied_embedding(trainer, causal_lm, dtype=torch.float16):
    """Keep the tied embedding / head weight in `dtype` for this training run, if exact."""
    if getattr(causal_lm, "_soup_embedding_checked", False):
        return
    causal_lm._soup_embedding_checked = True
    embedding = causal_lm.get_input_embeddings()
    head = causal_lm.get_output_embeddings()
    if type(embedding) is not torch.nn.Embedding or head is None or \
            getattr(head, "weight", None) is not embedding.weight:
        return
    weight = embedding.weight
    if weight.dtype != torch.float32 or weight.requires_grad or not weight.is_cuda or \
            not _exact_in(weight, dtype):
        return
    from transformers import TrainerCallback

    weight.data = weight.data.to(dtype)
    setattr(embedding, _UPCAST_ATTR, torch.float32)

    class Restore(TrainerCallback):
        def on_train_end(self, args, state, control, **kwargs):
            if weight.dtype == dtype:
                weight.data = weight.data.to(torch.float32)
            setattr(embedding, _UPCAST_ATTR, None)

    trainer.add_callback(Restore())
