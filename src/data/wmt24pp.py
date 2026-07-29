"""
wmt24pp.py
──────────
WMT24++ (google/wmt24pp) loader that mirrors the FLORES loader contract used
throughout the pipeline: `load_wmt24pp_sentences(lang_code)` returns
{"dev": [...], "devtest": [...]} where "dev" is the SELECTION POOL (860
sentences) and "devtest" is the FIXED TEST SET (100 sentences). The split is
read from data/wmt24pp_split.json (generated once by
scripts/make_wmt24pp_split.py with seed 12345) so it is identical across every
model, language, k, and reasoning run.

WMT24++ is en→xx only. The evaluated targets for the WMT24++ arm are:
  cat_Latn → en-ca_ES (Catalan)
  zul_Latn → en-zu_ZA (Zulu)
  mal_Mlym → en-ml_IN (Malayalam)
  slk_Latn → en-sk_SK (Slovak)
  isl_Latn → en-is_IS (Icelandic)
English source sentences are identical across configs (verified at split
generation), so the "eng_Latn" side is loaded from any target config, and the
committed split's segment ids are language-independent.

Inputs:  lang_code in {eng_Latn, swh_Latn, tam_Taml, tel_Telu}; the committed
         split JSON; HuggingFace `datasets` (downloads google/wmt24pp).
Outputs: dict with 'dev' (pool) and 'devtest' (test) sentence lists, ordered
         by ascending segment_id within each set.
"""

import json
import os
from typing import Dict, List

# WMT_DATASET_ID: HuggingFace dataset id for WMT24++.
WMT_DATASET_ID = "google/wmt24pp"

# WMT_CONFIG_FOR_LANG: FLORES-style lang codes → WMT24++ config names.
WMT_CONFIG_FOR_LANG: Dict[str, str] = {
    "cat_Latn": "en-ca_ES",
    "zul_Latn": "en-zu_ZA",
    "mal_Mlym": "en-ml_IN",
    "slk_Latn": "en-sk_SK",
    "isl_Latn": "en-is_IS",
}

# WMT_TGT_LANGS: the subset of our target languages WMT24++ covers.
WMT_TGT_LANGS: List[str] = list(WMT_CONFIG_FOR_LANG.keys())

# _SPLIT_PATH: committed fixed-split JSON; env-overridable for cluster layouts.
_SPLIT_PATH = os.environ.get(
    "RTRACE_WMT_SPLIT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "wmt24pp_split.json"),
)

_split_cache = None
_rows_cache: Dict[str, Dict[int, dict]] = {}


def _load_split() -> dict:
    """
    Purpose: Load (once) the committed fixed split JSON.
    Inputs: none (reads _SPLIT_PATH).
    Outputs: split dict with test_segment_ids / pool_segment_ids.
    """
    global _split_cache
    if _split_cache is None:
        with open(_SPLIT_PATH, "r", encoding="utf-8") as f:
            _split_cache = json.load(f)
        if len(_split_cache["test_segment_ids"]) != _split_cache["n_test"]:
            raise ValueError(f"Corrupt split file: {_SPLIT_PATH}")
    return _split_cache


def _rows_by_segment_id(config: str) -> Dict[int, dict]:
    """
    Purpose: Load (once per config) all WMT24++ rows keyed by segment_id.
    Inputs: WMT24++ config name.
    Outputs: dict segment_id -> row dict.
    """
    if config not in _rows_cache:
        from datasets import load_dataset

        ds = load_dataset(WMT_DATASET_ID, config, split="train")
        _rows_cache[config] = {int(r["segment_id"]): r for r in ds}
    return _rows_cache[config]


def load_wmt24pp_sentences(lang: str) -> Dict[str, List[str]]:
    """
    Purpose: Return WMT24++ sentences in the FLORES loader shape, using the fixed split.
    Inputs: lang code — "eng_Latn" for the shared source side, else a covered target code.
    Outputs: dict with 'dev' (selection pool, 860) and 'devtest' (test, 100) sentence lists.
    """
    split = _load_split()

    if lang == "eng_Latn":
        config = WMT_CONFIG_FOR_LANG[WMT_TGT_LANGS[0]]
        field = "source"
    else:
        if lang not in WMT_CONFIG_FOR_LANG:
            raise ValueError(
                f"WMT24++ does not cover '{lang}'. Covered targets: {sorted(WMT_CONFIG_FOR_LANG)}"
            )
        config = WMT_CONFIG_FOR_LANG[lang]
        field = "target"

    rows = _rows_by_segment_id(config)

    dev = [str(rows[sid][field]) for sid in split["pool_segment_ids"]]
    devtest = [str(rows[sid][field]) for sid in split["test_segment_ids"]]

    if len(dev) != split["n_pool"] or len(devtest) != split["n_test"]:
        raise ValueError(
            f"WMT24++ split mismatch for {lang}: got pool={len(dev)}, test={len(devtest)}"
        )
    return {"dev": dev, "devtest": devtest}
