"""One fp16 copy of Qwen's tied embedding / output head instead of fp32 + fp16.

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
        out = original(self, input)
        dtype = getattr(self, _UPCAST_ATTR, None)
        return out if dtype is None else out.to(dtype)

    forward._soup_patched = True
    cls.forward = forward


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
