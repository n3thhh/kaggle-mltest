"""Two optional integrations this run never uses, not imported.

* transformers' assisted-generation helpers (`transformers.generation.candidate_generator`) import
  `sklearn.metrics.roc_curve` when scikit-learn is installed (it is on Kaggle), which pulls in
  scikit-learn and SciPy's stats: about 1.5 s of every `soup train`. `roc_curve` only tunes the
  confidence threshold of an assistant model during `generate(..., assistant_model=...)`; training
  never generates. The module is imported here as transformers imports it on a machine without
  scikit-learn, and keeps answering "not available" to its own checks, so it stays consistent.
* TRL's base trainer (`trl.trainer.base_trainer`) imports `wandb` when it is installed (it is on
  Kaggle), about 0.6 s, only to put a W&B run URL in the model card. Soup trains with
  `report_to: none`, so no W&B run exists and the URL is None either way; same treatment.

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
