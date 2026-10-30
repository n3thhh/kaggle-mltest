"""One scan of the installed packages' metadata instead of two.

On import, transformers and then TRL each call importlib.metadata.packages_distributions(), which
opens the metadata of every installed distribution (hundreds on a Kaggle image) and walks their
file lists: a few seconds of every `soup train`, spent twice on the same answer. The harness has
already imported transformers when `apply()` runs, so transformers' mapping exists; while we
import TRL, packages_distributions() hands TRL a copy of it instead of scanning again. Nothing is
installed or removed in between (the harness checks that no package changes during a run), so the
mapping is the one TRL would compute.
"""

import contextlib
import importlib.metadata


@contextlib.contextmanager
def reuse_package_mapping():
    try:
        from transformers.utils import import_utils

        mapping = getattr(import_utils, "PACKAGE_DISTRIBUTION_MAPPING", None)
    except Exception:  # noqa: BLE001
        mapping = None
    if not isinstance(mapping, dict):
        yield
        return
    original = importlib.metadata.packages_distributions

    def packages_distributions():
        return {name: list(dists) for name, dists in mapping.items()}

    importlib.metadata.packages_distributions = packages_distributions
    try:
        yield
    finally:
        importlib.metadata.packages_distributions = original


def import_trl():
    with reuse_package_mapping():
        import trl  # noqa: F401
