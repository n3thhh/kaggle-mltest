"""Soup's text SFT rows tokenized by our own loop instead of `Dataset.map`.

Soup's text SFT helper (`soup_cli.trainer.sft._map_text_sft_rows`) builds
`Dataset.from_list(rows)` and maps a closure over the chat formatter onto it. `datasets` names the
result of every map by hashing the mapped function with dill, and this closure holds the tokenizer:
the hash serializes its whole 151k-token vocabulary (about 760k Python objects), about 2 s of every
`soup train` on a Kaggle T4, only to name a cache file that an in-memory dataset never writes.

Here the same rows go through the same formatter, row by row, in our own loop, with Soup's own
check and error message for rows without a loss target, and the dataset is built from the results
in memory (`Dataset.from_list`, the way Soup builds its input). The columns, their order, their
types and every value are the ones the map produces (checked by test_kernels.py). `datasets`'
fingerprinting, its cache and the random state are not touched: the result gets the fingerprint
`datasets` gives any in-memory dataset. Organizers' answer (Soup Tasters chat, 31.10, 14:13): own
tokenization code, rows bit for bit the same, datasets kept in memory without a cache on disk,
is allowed.
"""


def patch_soup_text_map():
    from soup_cli.trainer import sft

    helper = sft._map_text_sft_rows
    if getattr(helper, "_soup_patched", False):
        return

    def map_text_sft_rows(rows, *, format_row, split, max_length):
        from datasets import Dataset
        from soup_cli.data.loss_mask import NoCausalLossTargetError, ensure_causal_loss_target

        source = Dataset.from_list(rows)
        if "messages" not in source.column_names:  # what map would reject: Soup's own path
            return helper(rows, format_row=format_row, split=split, max_length=max_length)
        out = []
        for index, example in enumerate(source):  # the python rows map hands its function
            try:
                formatted = format_row(example)
                labels = formatted.get("labels")
                if labels is not None:
                    ensure_causal_loss_target(labels, max_length=max_length)
            except NoCausalLossTargetError as exc:
                raise ValueError(f"{split} row {index + 1}: {exc}") from exc
            # map's row: the input columns but "messages", updated with the formatter's output
            row = {key: value for key, value in example.items() if key != "messages"}
            row.update(formatted)
            out.append(row)
        if not out:
            return helper(rows, format_row=format_row, split=split, max_length=max_length)
        return Dataset.from_list(out)

    map_text_sft_rows._soup_patched = True
    sft._map_text_sft_rows = map_text_sft_rows
