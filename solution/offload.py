"""4-bit weights and saved activations in pinned CPU memory, moved per layer on side streams.

Weights. As bitsandbytes quantizes each NF4 weight on the GPU during loading, its packed bytes
and its per-block absmax (reconstructed in fp32 with bitsandbytes' own ops, as nf4.py uses it)
are copied to pinned host memory and the GPU copy is released; the parameter keeps its shape
through a zero-stride placeholder, so peft, Soup and the trainer see the model as before. A layer's
seven weights travel to one of two GPU slots on a copy stream while the previous layer computes
(the next layer forward, the previous one backward), so each pass over the model moves 0.7 GB
over PCIe in the background instead of keeping it resident.

Activations. What a fused layer keeps for its backward pass (layer.py) is copied to pinned host
memory on a second copy stream as soon as the layer's forward pass has produced it, and brought
back one layer ahead of the backward pass.

Everything is copied, not changed: the bytes the GPU computes with are the bytes it would have
kept. The copies cost PCIe time, which overlaps compute on the T4's separate copy engines.
"""

import os
import threading

import torch

ENABLED = os.environ.get("SOUP_SOLUTION_OFFLOAD", "1") == "1"
ACTIVATIONS = os.environ.get("SOUP_SOLUTION_OFFLOAD_ACTS", "1") == "1"
HEAD = os.environ.get("SOUP_SOLUTION_OFFLOAD_HEAD", "1") == "1"


class HostArena:
    """Pinned host memory handed out from a few large blocks (one cudaHostAlloc per block, not
    per tensor)."""

    def __init__(self, block_bytes):
        self.block_bytes = block_bytes
        self.blocks, self.used, self.spare = [], 0, []
        self.lock = threading.Lock()

    def reserve(self, n_blocks):
        """Allocate `n_blocks` blocks ahead of use (pinning memory takes time: the warm-up thread
        does it while the main thread is still importing libraries)."""
        for _ in range(n_blocks):
            block = torch.empty(self.block_bytes, dtype=torch.uint8, pin_memory=True)
            with self.lock:
                self.spare.append(block)

    def alloc(self, numel, dtype=torch.uint8):
        """`numel` elements of `dtype`, 512-byte aligned."""
        itemsize = torch.empty((), dtype=dtype).element_size()
        nbytes = (numel * itemsize + 511) // 512 * 512
        with self.lock:
            if not self.blocks or self.used + nbytes > self.blocks[-1].numel():
                if self.spare and self.spare[0].numel() >= nbytes:
                    self.blocks.append(self.spare.pop(0))
                else:
                    self.blocks.append(torch.empty(max(self.block_bytes, nbytes),
                                                   dtype=torch.uint8, pin_memory=True))
                self.used = 0
            view = self.blocks[-1][self.used:self.used + nbytes]
            self.used += nbytes
        return view.view(dtype)[:numel]


WEIGHTS = HostArena(256 * 2**20)


def _placeholder(like):
    """A tensor of `like`'s shape and dtype on its device, backed by one element."""
    return torch.zeros(1, dtype=like.dtype, device=like.device).as_strided(
        like.shape, (0,) * like.dim())


_COPIES_DONE = [False]


def host_ready():
    """Once, before the host copies of the weights are first read: wait for the copies queued
    during loading (they may have been queued on another stream than the one reading)."""
    if not _COPIES_DONE[0]:
        _COPIES_DONE[0] = True
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def is_offloaded(quant_state):
    return getattr(quant_state, "_soup_host", None) is not None


def fetch(quant_state, device):
    """The packed weight on the GPU (synchronous use outside the fused layers)."""
    host_ready()
    return quant_state._soup_host.to(device, non_blocking=True)


def patch_quantize():
    """Params4bit._quantize: quantize on the GPU as before, then keep the packed bytes and the
    fp32 absmax in pinned host memory."""
    import bitsandbytes as bnb

    cls = bnb.nn.Params4bit
    if getattr(cls._quantize, "_soup_patched", False):
        return
    original = cls._quantize

    def _quantize(self, device):
        result = original(self, device)
        try:
            data = self.data
            qs = self.quant_state
            if data.is_cuda and qs is not None and qs.quant_type == "nf4" and \
                    data.dtype == torch.uint8 and data.numel() >= 2**14 and \
                    (not qs.nested or qs.state2.blocksize == 256):
                from .nf4 import _absmax_fp32

                absmax, _ = _absmax_fp32(qs)
                # Copies queued on the current stream, without waiting for them: everything
                # that later reads these host copies is queued after them on the GPU (the
                # weight stream waits for this stream), and the freed GPU memory is reused in
                # stream order.
                host = WEIGHTS.alloc(data.numel())
                host.copy_(data.reshape(-1), non_blocking=True)
                host_absmax = WEIGHTS.alloc(absmax.numel(), torch.float32)
                host_absmax.copy_(absmax, non_blocking=True)
                qs._soup_host, qs._soup_host_absmax = host, host_absmax
                qs._soup_absmax_fp32 = None  # the GPU copy is released; nf4.py refetches
                self.data = _placeholder(data)
        except Exception as exc:  # noqa: BLE001 - keep the weight on the GPU
            import sys

            print(f"[solution] weight offload skipped: {exc!r}", file=sys.stderr)
        return result

    _quantize._soup_patched = True
    cls._quantize = _quantize


def patch_loading():
    """Two things transformers does while loading that would put the weights we keep in host
    memory on the GPU anyway:

    * `caching_allocator_warmup` reserves GPU memory for the whole model up front (one large
      allocation, to make loading faster): about 1.2 GB here, released only much later, so it
      would set the run's memory peak. The weights do not stay on the GPU, so it is skipped.
    * the tied embedding (467 MB) would be loaded to the GPU and only then moved to host memory
      (head_stream.py); it is loaded to host memory directly instead.
    """
    import transformers.core_model_loading as loading
    import transformers.modeling_utils as modeling_utils

    if getattr(loading.get_device, "_soup_patched", False):
        return
    modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    if not HEAD:
        return
    original = loading.get_device
    active = {"on": True}

    def get_device(device_map, param_name, *args, **kwargs):
        device = original(device_map, param_name, *args, **kwargs)
        if active["on"] and param_name.endswith("embed_tokens.weight") and device != "disk":
            return torch.device("cpu") if kwargs.get("valid_torch_device") or args else "cpu"
        return device

    get_device._soup_patched = True
    loading.get_device = get_device

    # If loading with the embedding in host memory fails for any reason, load again as usual.
    import transformers

    cls = transformers.PreTrainedModel
    load = cls.from_pretrained.__func__

    def from_pretrained(klass, *args, **kwargs):
        from . import mark

        mark(f"from_pretrained({klass.__name__}) starts")
        try:
            return _from_pretrained(klass, *args, **kwargs)
        finally:
            mark(f"from_pretrained({klass.__name__}) done")

    def _from_pretrained(klass, *args, **kwargs):
        if not active["on"]:
            return load(klass, *args, **kwargs)
        try:
            return load(klass, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            import sys

            print(f"[solution] loading with the embedding in host memory failed ({exc!r}); "
                  "loading as usual", file=sys.stderr)
            active["on"] = False
            torch.cuda.empty_cache()
            return load(klass, *args, **kwargs)

    cls.from_pretrained = classmethod(from_pretrained)


class WeightStream:
    """Two GPU slots, each holding one decoder layer's weights (packed bytes + fp32 absmax);
    copies run on their own stream, one layer ahead of compute."""

    def __init__(self, device, layout):
        # layout: per layer, a list of quant_states in a fixed order
        self.device, self.layout = device, layout
        self.packed_bytes = [int(qs._soup_host.numel()) for qs in layout[0]]
        self.absmax_numel = [int(qs._soup_host_absmax.numel()) for qs in layout[0]]
        self.stream = torch.cuda.Stream(device)
        self.slots = []
        for _ in range(2):
            packed = torch.empty(sum(self.packed_bytes), dtype=torch.uint8, device=device)
            absmax = torch.empty(sum(self.absmax_numel), dtype=torch.float32, device=device)
            views_p, views_a, p0, a0 = [], [], 0, 0
            for nb, na in zip(self.packed_bytes, self.absmax_numel):
                views_p.append(packed[p0:p0 + nb])
                views_a.append(absmax[a0:a0 + na])
                p0, a0 = p0 + nb, a0 + na
            self.slots.append({"idx": None, "packed": views_p, "absmax": views_a,
                               "ready": torch.cuda.Event()})

    def _load(self, slot, idx):
        compute = torch.cuda.current_stream(self.device)
        free = torch.cuda.Event()
        free.record(compute)  # everything already queued that reads this slot
        self.stream.wait_event(free)
        with torch.cuda.stream(self.stream):
            for dst_p, dst_a, qs in zip(slot["packed"], slot["absmax"], self.layout[idx]):
                dst_p.copy_(qs._soup_host, non_blocking=True)
                dst_a.copy_(qs._soup_host_absmax, non_blocking=True)
        slot["ready"].record(self.stream)
        slot["idx"] = idx

    def acquire(self, idx, prefetch):
        """Layer `idx`'s weights as [(packed, absmax_fp32)] in layout order, ready for the
        current stream; starts copying layer `prefetch` into the other slot."""
        here = next((s for s in self.slots if s["idx"] == idx), None)
        if here is None:
            spare = [s for s in self.slots if s["idx"] != prefetch]
            here = spare[0] if spare else self.slots[0]
            self._load(here, idx)
        torch.cuda.current_stream(self.device).wait_event(here["ready"])
        if prefetch is not None and 0 <= prefetch < len(self.layout):
            other = self.slots[1] if here is self.slots[0] else self.slots[0]
            if other["idx"] != prefetch:
                self._load(other, prefetch)
        return list(zip(here["packed"], here["absmax"]))


class ActivationStore:
    """What fused layers keep for the backward pass, in pinned host memory between the two."""

    def __init__(self, device):
        self.device = device
        self.d2h = torch.cuda.Stream(device)
        self.h2d = torch.cuda.Stream(device)
        self.blocks, self.block, self.offset = [], 0, 0
        self.lock = threading.Lock()
        self.pending = {}
        self.in_flight = []

    def capacity(self):
        return sum(block.numel() for block in self.blocks)

    def reserve(self, nbytes):
        """Grow the pinned blocks to at least `nbytes` (blocks are kept from step to step)."""
        with self.lock:
            missing = nbytes - self.capacity()
            if missing > 0:
                self.blocks.append(torch.empty(missing, dtype=torch.uint8, pin_memory=True))

    def begin_step(self, nbytes):
        self.reserve(nbytes)
        self.block, self.offset = 0, 0
        self.in_flight = []
        # the previous step's last host-to-device reads must finish before this step writes
        self.d2h.wait_stream(self.h2d)

    def _take(self, nbytes):
        """The next `nbytes` of pinned memory (512-byte aligned) in the blocks, in order."""
        while True:
            if self.block < len(self.blocks):
                block = self.blocks[self.block]
                start = (self.offset + 511) // 512 * 512
                if start + nbytes <= block.numel():
                    self.offset = start + nbytes
                    return block[start:start + nbytes]
                self.block, self.offset = self.block + 1, 0
            else:  # more than reserved (should not happen): one more block
                with self.lock:
                    self.blocks.append(torch.empty(max(nbytes, 256 * 2**20), dtype=torch.uint8,
                                                   pin_memory=True))

    def save(self, tensors, handle=None):
        """Start copying `tensors` (GPU) to host; returns a handle (or adds them to `handle`)."""
        compute = torch.cuda.current_stream(self.device)
        ready = torch.cuda.Event()
        ready.record(compute)
        self.d2h.wait_event(ready)
        hosts = []
        with torch.cuda.stream(self.d2h):
            for t in tensors:
                nbytes = t.numel() * t.element_size()
                host = self._take(nbytes).view(t.dtype).view(t.shape)
                host.copy_(t, non_blocking=True)
                t.record_stream(self.d2h)  # the allocator keeps t until the copy is done
                hosts.append(host)
        done = torch.cuda.Event()
        done.record(self.d2h)
        # At most two batches of copies in flight (about one layer's): the GPU memory of a
        # tensor being copied is held until its copy ends, so a backlog would raise the memory
        # peak. Waiting here (CPU side) costs nothing while the copies keep pace.
        self.in_flight.append(done)
        while len(self.in_flight) > 2:
            self.in_flight.pop(0).synchronize()
        if handle is not None:
            handle["hosts"].extend(hosts)
            handle["d2h"] = done  # the copy stream is in order: this one ends after the rest
            return handle
        return {"hosts": hosts, "d2h": done, "gpu": None, "h2d": None}

    def prefetch(self, handle):
        if handle is None or handle["gpu"] is not None:
            return
        compute = torch.cuda.current_stream(self.device)
        gpus = [torch.empty(h.shape, dtype=h.dtype, device=self.device) for h in handle["hosts"]]
        allocated = torch.cuda.Event()
        allocated.record(compute)
        self.h2d.wait_event(allocated)
        self.h2d.wait_event(handle["d2h"])
        with torch.cuda.stream(self.h2d):
            for g, h in zip(gpus, handle["hosts"]):
                g.copy_(h, non_blocking=True)
        done = torch.cuda.Event()
        done.record(self.h2d)
        handle["gpu"], handle["h2d"] = gpus, done

    def load(self, handle):
        self.prefetch(handle)
        torch.cuda.current_stream(self.device).wait_event(handle["h2d"])
        gpus, handle["gpu"] = handle["gpu"], None
        return gpus


_STATE = {}


def weight_stream(model_layers, device):
    """The model's WeightStream (built at first use), or None when the weights are on the GPU."""
    host_ready()
    first = model_layers[0] if len(model_layers) else None
    key = (id(first), len(model_layers))
    hit = _STATE.get(key)
    if hit is not None and hit[0] is first:
        return hit[1]
    stream = None
    try:
        layout = [layer_quant_states(layer) for layer in model_layers]
        if layout and all(all(is_offloaded(qs) for qs in states) for states in layout) and \
                all([int(qs._soup_host.numel()) for qs in states] ==
                    [int(qs._soup_host.numel()) for qs in layout[0]] for states in layout):
            stream = WeightStream(device, layout)
    except Exception as exc:  # noqa: BLE001
        import sys

        print(f"[solution] weight streaming unavailable: {exc!r}", file=sys.stderr)
        stream = None
    _STATE[key] = (first, stream)
    return stream


def layer_quant_states(layer):
    attn, mlp = layer.self_attn, layer.mlp
    modules = (attn.q_proj.base_layer, attn.k_proj, attn.v_proj.base_layer, attn.o_proj,
               mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    return [m.weight.quant_state for m in modules]


def restore_weights(root):
    """Every offloaded 4-bit weight under `root` back on the GPU (the per-module path's
    fallback when the fused layer is not used: weights fetched per use would be slow)."""
    host_ready()
    for module in root.modules():
        weight = getattr(module, "weight", None)
        qs = getattr(weight, "quant_state", None)
        if qs is not None and is_offloaded(qs) and weight.is_cuda:
            weight.data = qs._soup_host.to(weight.device).view(weight.shape)
            qs._soup_host = qs._soup_host_absmax = None
    _STATE.clear()


_ACTS = {}


def activation_store(device):
    if not ACTIVATIONS or device.type != "cuda":
        return None
    if device not in _ACTS:
        _ACTS[device] = ActivationStore(device)
    return _ACTS[device]
