"""Soup's tokenization `Dataset.map` without hashing the tokenizer to name its result.

`datasets` names the result of every `Dataset.map` by hashing the mapped function with dill, so it
can find a cached result next time. Soup's text SFT helper (`soup_cli.trainer.sft.
_map_text_sft_rows`) maps a closure over the chat formatter, which holds the tokenizer: the hash
serializes its whole 151k-token vocabulary (about 760k Python objects), about 2 s of every
`soup train` on a Kaggle T4. That helper maps `Dataset.from_list(rows)`, a dataset in memory with
no cache files, so the name is never used to read or write a cache; Soup and TRL's SFT path never
read it either. While that helper runs, its map gets a random name from datasets' own generator
(a private `random.Random`, the same thing datasets does when a function cannot be hashed), so the
global random state, the rows, the tokens and their order are unchanged. Every other map (TRL's
own, anything with cache files, anything with an explicit name or cache file) is left as it is.
"""

import contextvars
import functools

_IN_SOUP_TEXT_MAP = contextvars.ContextVar("soup_text_sft_map", default=False)


def patch_soup_text_map():
    import datasets
    from datasets import fingerprint
    from soup_cli.trainer import sft

    helper = sft._map_text_sft_rows
    if getattr(helper, "_soup_patched", False):
        return
    cls = datasets.Dataset
    original_map = cls.map

    @functools.wraps(original_map)
    def map(self, *args, **kwargs):  # noqa: A001 - the name of Dataset.map
        if _IN_SOUP_TEXT_MAP.get() and type(self) is cls and len(args) <= 1 and \
                kwargs.get("new_fingerprint") is None and kwargs.get("cache_file_name") is None \
                and kwargs.get("num_proc") in (None, 0, 1) and \
                getattr(self, "cache_files", None) == []:
            kwargs["new_fingerprint"] = fingerprint.generate_random_fingerprint()
        return original_map(self, *args, **kwargs)

    @functools.wraps(helper)
    def map_text_sft_rows(*args, **kwargs):
        token = _IN_SOUP_TEXT_MAP.set(True)
        try:
            return helper(*args, **kwargs)
        finally:
            _IN_SOUP_TEXT_MAP.reset(token)

    map_text_sft_rows._soup_patched = True
    cls.map = map
    sft._map_text_sft_rows = map_text_sft_rows
