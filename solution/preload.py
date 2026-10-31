"""Soup's model loaded in a thread of this process while apply() imports TRL and Soup.

`soup train` imports TRL and Soup's trainer (about 7.5 s on a Kaggle T4), reads its config and
tokenizer, and only then loads and quantizes the model (about 2 s, mostly copying 3 GB of
checkpoint to the GPU, which does not need Python's lock). Here the same
`AutoModelForCausalLM.from_pretrained(...)` call starts in a thread right at apply(), with the
arguments Soup will pass for this run's config, and runs while the main thread imports. When
Soup makes its call, the thread is joined and the loaded model is handed over only if

* the call is the predicted one: same model id, same keyword arguments (the quantization config
  compared field by field), nothing else;
* the load did not draw random numbers (Soup seeds before loading; the base run's RNG state after
  loading must be ours too);
* the load finished without an error.

Otherwise the preloaded model is dropped and Soup's call loads as usual. Organizers' answer
(Soup Tasters chat, 31.10, 17:41): allowed when all significant arguments and the model's state
match, Soup receives a fully loaded object, and any mismatch falls back to the usual loading.

Model construction briefly changes process-wide state (torch's default dtype, a few patched torch
functions); the main thread waits until that part is over before it goes on importing, so
nothing it imports can see that state. The modules the loader would import lazily are imported
first, in the main thread, so the two threads never import the same module at once.
"""

import copy
import contextlib
import sys
import threading

STATE = {"thread": None, "model": None, "error": None, "call": None, "expected": None,
         "rng_same": None, "taken": False}
_CONSTRUCTED = threading.Event()
_ALLOW_CPU = False  # tests only
_TRAINING_KEYS = {"epochs", "lr", "batch_size", "gradient_accumulation_steps", "seed",
                  "logging_steps", "save_steps", "lora", "quantization"}
_TOP_KEYS = {"base", "task", "data", "training", "output"}
_LOADER_MODULES = (
    "transformers.modeling_utils", "transformers.core_model_loading",
    "transformers.models.auto.modeling_auto", "transformers.models.qwen2.modeling_qwen2",
    "transformers.quantizers.auto", "transformers.quantizers.quantizer_bnb_4bit",
    "transformers.integrations.bitsandbytes", "transformers.integrations.accelerate",
    "transformers.integrations.sdpa_attention", "transformers.integrations.hub_kernels",
    "transformers.loss.loss_utils", "transformers.safetensors_conversion",
    "transformers.utils.loading_report", "huggingface_hub.serialization._torch",
    "bitsandbytes", "safetensors.torch",
)


def _config():
    """This run's Soup config (the harness runs `soup train --config <file> --yes`)."""
    argv = list(sys.argv)
    if len(argv) < 2 or argv[1] != "train":
        return None
    path = None
    for i, arg in enumerate(argv):
        if arg in ("--config", "-c") and i + 1 < len(argv):
            path = argv[i + 1]
        elif arg.startswith("--config="):
            path = arg.split("=", 1)[1]
    if path is None:
        return None
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _predicted_call(cfg):
    """(model id, kwargs) of Soup's from_pretrained call for a plain 4-bit LoRA SFT config, as
    soup_cli.trainer.sft builds them; None for any other config."""
    if not isinstance(cfg, dict) or not set(cfg) <= _TOP_KEYS:
        return None
    training = cfg.get("training") or {}
    if cfg.get("task", "sft") != "sft" or not isinstance(cfg.get("base"), str) or \
            not isinstance(training, dict) or not set(training) <= _TRAINING_KEYS or \
            training.get("quantization") != "4bit":
        return None
    import torch
    from soup_cli.utils.gpu import get_compute_dtype, resolve_base_load_dtype, resolve_device_map
    from transformers import BitsAndBytesConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {
        "trust_remote_code": False,
        "device_map": resolve_device_map(device),
        "torch_dtype": resolve_base_load_dtype(device, full_finetune=False),
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=get_compute_dtype(), bnb_4bit_use_double_quant=True),
    }
    return cfg["base"], kwargs


def _same(a, b):
    from transformers import BitsAndBytesConfig

    if isinstance(a, BitsAndBytesConfig) or isinstance(b, BitsAndBytesConfig):
        return type(a) is type(b) and a.to_dict() == b.to_dict()
    return type(a) is type(b) and a == b


def _matches(args, kwargs):
    model_id, expected = STATE["expected"]
    if list(args) != [model_id]:
        return f"model {args!r} instead of {model_id!r}"
    if set(kwargs) != set(expected):
        return f"arguments {sorted(kwargs)} instead of {sorted(expected)}"
    for key in kwargs:
        if not _same(kwargs[key], expected[key]):
            return f"{key}={kwargs[key]!r} instead of {expected[key]!r}"
    return None


@contextlib.contextmanager
def _signal_when_constructed():
    try:
        yield
    finally:
        _CONSTRUCTED.set()


def _load(load, model_id, kwargs):
    import torch

    try:
        cpu = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        STATE["model"] = load(model_id, **kwargs)
        STATE["rng_same"] = torch.equal(torch.get_rng_state(), cpu) and (
            cuda is None or torch.equal(torch.cuda.get_rng_state(), cuda))
    except BaseException as exc:  # noqa: BLE001 - Soup's own call loads as usual
        STATE["error"] = f"{exc.__class__.__name__}: {exc}"[:300]
    finally:
        _CONSTRUCTED.set()


def start():
    """Start the load in a thread if this run's config is the plain 4-bit SFT one."""
    import importlib

    import torch

    if STATE["thread"] is not None or not (torch.cuda.is_available() or _ALLOW_CPU):
        return
    predicted = _predicted_call(_config())
    if predicted is None:
        return
    for name in _LOADER_MODULES:
        with contextlib.suppress(Exception):
            importlib.import_module(name)
    import transformers
    from transformers import AutoModelForCausalLM

    from . import mark

    model_id, kwargs = predicted
    STATE["expected"] = (model_id, copy.deepcopy(kwargs))
    load = AutoModelForCausalLM.from_pretrained  # bound to the class, unpatched

    # the end of model construction, when torch's process-wide state is restored
    base_cls = transformers.modeling_utils.PreTrainedModel
    get_init_context = base_cls.get_init_context.__func__

    def init_context(cls, *args, **kw):
        contexts = get_init_context(cls, *args, **kw)
        if threading.current_thread() is STATE["thread"]:
            contexts = [_signal_when_constructed()] + list(contexts)
        return contexts

    base_cls.get_init_context = classmethod(init_context)

    def from_pretrained(cls, *args, **kw):
        if cls is AutoModelForCausalLM and STATE["thread"] is not None and not STATE["taken"]:
            STATE["taken"] = True
            STATE["thread"].join()
            model, STATE["model"] = STATE["model"], None
            why = STATE["error"] or (None if STATE["rng_same"] else "the load drew random numbers") \
                or _matches(args, kw)
            if why is None and model is not None:
                mark("from_pretrained: the model loaded in a thread matches Soup's call, "
                     "handed over")
                return model
            print(f"[solution] model loaded in a thread not used ({why}); loading as usual",
                  file=sys.stderr)
            del model
            import gc

            gc.collect()
            torch.cuda.empty_cache()
        return load.__func__(cls, *args, **kw)

    AutoModelForCausalLM.from_pretrained = classmethod(from_pretrained)
    STATE["thread"] = threading.Thread(target=_load, args=(load, model_id, kwargs),
                                       name="soup-model-preload", daemon=True)
    mark("model load started in a thread")
    STATE["thread"].start()
    _CONSTRUCTED.wait(timeout=30)
    mark("model constructed in the thread; imports go on")
