"""Optional integrations this run never uses, not imported.

* transformers' assisted-generation helpers (`transformers.generation.candidate_generator`) import
  `sklearn.metrics.roc_curve` when scikit-learn is installed (it is on Kaggle), which pulls in
  scikit-learn and SciPy's stats: about 1.5 s of every `soup train`. `roc_curve` only tunes the
  confidence threshold of an assistant model during `generate(..., assistant_model=...)`; training
  never generates. The module is imported here as transformers imports it on a machine without
  scikit-learn, and keeps answering "not available" to its own checks, so it stays consistent.
* TRL's base trainer (`trl.trainer.base_trainer`) imports `wandb` when it is installed (it is on
  Kaggle), about 0.6 s, only to put a W&B run URL in the model card. Soup trains with
  `report_to: none`, so no W&B run exists and the URL is None either way; same treatment.
* `transformers.data` (imported with the data collators TRL uses) imports its GLUE/XNLI metric
  helpers, which import scikit-learn's `f1_score` and SciPy's `pearsonr` when scikit-learn is
  installed: the same scikit-learn + SciPy import the first point avoids. The helpers score GLUE
  benchmark predictions; nothing in training calls them. Same treatment.
* transformers' DETR-family object-detection losses (imported with the loss registry
  `transformers.loss.loss_utils`, which every model imports) import SciPy's
  `linear_sum_assignment` when SciPy is installed, for their Hungarian matching. A causal LM never
  uses them. Same treatment, with `is_scipy_available`.

Organizers' answer (Soup Tasters chat, 31.10, 11:27): skipping unused optional imports is allowed
when training, Soup's outputs and all checks stay the same; stubbing packages through
`sys.modules` is not, and is not done here: the packages stay installed and importable.

The global availability checks are restored right after each import: every other module sees the
installed packages as before. Nothing the training computes, logs or saves depends on either.
"""

import contextlib
import importlib
import sys

_MISSING = object()


def _unavailable():
    return False


@contextlib.contextmanager
def _reported_unavailable(owners, name):
    saved = [(owner, vars(owner).get(name, _MISSING)) for owner in owners]
    for owner in owners:
        setattr(owner, name, _unavailable)
    try:
        yield
    finally:
        for owner, value in saved:
            if value is _MISSING:
                delattr(owner, name)  # lazy modules resolve it again on access
            else:
                setattr(owner, name, value)


def import_candidate_generator_without_sklearn():
    name = "transformers.generation.candidate_generator"
    if name in sys.modules:
        return
    from transformers import utils
    from transformers.utils import import_utils

    with _reported_unavailable((utils, import_utils), "is_sklearn_available"):
        module = importlib.import_module(name)
    module.is_sklearn_available = _unavailable


def import_trl_base_trainer_without_wandb():
    name = "trl.trainer.base_trainer"
    if name in sys.modules:
        return
    import transformers

    with _reported_unavailable((transformers,), "is_wandb_available"):
        module = importlib.import_module(name)
    module.is_wandb_available = _unavailable


def import_data_metrics_without_sklearn():
    name = "transformers.data.metrics"
    if name in sys.modules:
        return
    from transformers import utils
    from transformers.utils import import_utils

    with _reported_unavailable((utils, import_utils), "is_sklearn_available"):
        module = importlib.import_module(name)
    module.is_sklearn_available = _unavailable


_DETECTION_LOSSES = ("loss_for_object_detection", "loss_deformable_detr", "loss_grounding_dino",
                     "loss_lw_detr", "loss_rf_detr", "loss_rt_detr")


def import_detection_losses_without_scipy():
    """transformers' loss registry (`transformers.loss.loss_utils`, imported with every model)
    with its DETR-family losses imported while SciPy reports unavailable."""
    name = "transformers.loss.loss_utils"
    if name in sys.modules:
        return
    from transformers import utils
    from transformers.utils import import_utils

    with _reported_unavailable((utils, import_utils), "is_scipy_available"):
        importlib.import_module(name)
    for loss in _DETECTION_LOSSES:
        module = sys.modules.get(f"transformers.loss.{loss}")
        if module is not None and hasattr(module, "is_scipy_available"):
            module.is_scipy_available = _unavailable
