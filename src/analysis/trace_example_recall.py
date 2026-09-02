#!/usr/bin/env python3
"""
trace_example_recall.py
───────────────────────
Quantify in-context example reuse with a graded **chrF character n-gram
recall** score: for every in-context example slot in a reasoning trace,
compute the fraction of the example's character (and word, via word_order=2)
n-grams reproduced in the trace, separately for its source (English) and
target sides, then take the per-slot max to capture "reused on either side".
Higher = stronger lexical reuse of that example.

Preprocessing matches eval_pipeline.py's `chrfpp = CHRF(word_order=2)`
exactly (char_order=6, word_order=2, β=2, lowercase=False, whitespace=False,
eps_smoothing=False); only the F-score extraction is overridden to return
the recall component (chrR) instead, via the `CHRFRecall` subclass below.
The reuse number is therefore methodologically parallel to the translation-
quality numbers in the paper.

Replaces the prior binary detector (exact full-sentence substring + ordinal
phrase) — those signals systematically *undercounted* real reuse and the
table was, by its own file name, a lower bound. Table 1's meaning shifts
from "% of examples referenced" to "mean reuse (chrF recall, 0–100)".

Layout mirrors the prior scripts in this directory:
  • directions are FLORES directions such as eng_to_wol
  • k values follow the same K_LIST sweep
  • example selection is regenerated with the same helper-function pipeline
  • reasoning traces are read from
        reasoning_traces_{family}_all{suffix}/{model_key}/{direction}/
  • method configs include rrf / random / sentinel / edit_dist

This version:
  • treats traces with an explicit null/None reasoning field as unusable and skips them
  • scores every retained (model × method × direction × k × example slot) with
    chrF recall on both source and target plus their max
  • produces a per-(model, method, direction, k) recall CSV aggregated from the
    long per-example rows (no fixed-k assumption — unambiguous grand mean)

Assumptions:
  • generation-pipeline helpers are imported from src.retrieval.retrieval_helpers
    (the same functions reasoning_main.py uses, so selections reproduce exactly);
    the dataset arm resolves through src/common/dataset_registry.py
    (RTRACE_DATASET: flores | wmt24pp) via the {"dev","devtest"} loader contract
  • no models are run here; this script only reconstructs selected examples and
    inspects existing reasoning trace files
  • the installed sacrebleu exposes CHRF._compute_f_score with the
    `(n_hyp, n_ref, n_match) = statistics[3 * i: 3 * i + 3]` slice and the
    `if n_hyp > 0 and n_ref > 0` effective-order guard that `CHRFRecall`
    overrides; pin the sacrebleu version to keep this contract.

Requirements:
  pip install sacrebleu pandas numpy
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sacrebleu.metrics import CHRF
from scipy.stats import pearsonr, spearmanr

from src.common.dataset_registry import get_dataset, filter_models
from src.retrieval.retrieval_helpers import (
    _apply_devtest_limit,
    build_fragmentshot_pools,
    ensemble_topk_dispatch,
    load_retrieval_embeddings,
    load_sentinel_src_scores,
    parse_ensemble_method,
    topk_cosine_indices_and_scores,
    topk_cosine_indices_from_pools,
)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Dataset arm (RTRACE_DATASET: flores | wmt24pp) — every root, language list,
# and k value below derives from the registry spec, so the script is
# dataset-agnostic given the {"dev","devtest"} loader contract.
DS = get_dataset()
OUT_BASE = DS.out_base()

MODELS: Dict[str, str] = {
    "ministral_8b": "Ministral 8B",
    "ministral_14b": "Ministral 14B",
    "magistral_small": "Magistral 24B",
    "qwen3_8b": "Qwen3 8B",
    "qwen3_14b": "Qwen3 14B",
    "qwen3_32b": "Qwen3 32B",
}
# Optional csv filter (RTRACE_EVAL_MODELS), e.g. Qwen-only while Mistral runs.
MODELS = filter_models(MODELS)

MODEL_FAMILY_ROOT: Dict[str, str] = {
    "ministral_8b": "Mistral",
    "ministral_14b": "Mistral",
    "magistral_small": "Mistral",
    "qwen3_8b": "Qwen",
    "qwen3_14b": "Qwen",
    "qwen3_32b": "Qwen",
}

SRC_LANG_CODES: List[str] = list(DS.src_langs)

TGT_LANG_CODES: List[str] = list(DS.tgt_langs)

# Union over both arms; FLORES strings preserved verbatim (mirrors
# reasoning_main.py's LANG_NAME / LANG_DIRNAME).
LANG_NAME: Dict[str, str] = {
    "eng_Latn": "English",
    "wol_Latn": "Wolof",
    "swh_Latn": "Swahili",
    "lus_Latn": "Mizo",
    "mni_Beng": "Manipuri",
    "tel_Telu": "Telugu",
    "tam_Taml": "Tamil",
    "uzn_Latn": "Northern Uzbek",
    # WMT24++ arm
    "cat_Latn": "Catalan",
    "zul_Latn": "Zulu",
    "mal_Mlym": "Malayalam",
    "slk_Latn": "Slovak",
    "isl_Latn": "Icelandic",
}

LANG_DIRNAME: Dict[str, str] = {
    "eng_Latn": "eng",
    "wol_Latn": "wol",
    "swh_Latn": "swh",
    "lus_Latn": "lus",
    "mni_Beng": "mni",
    "tel_Telu": "tel",
    "tam_Taml": "tam",
    "uzn_Latn": "uzn",
    # WMT24++ arm
    "cat_Latn": "cat",
    "zul_Latn": "zul",
    "mal_Mlym": "mal",
    "slk_Latn": "slk",
    "isl_Latn": "isl",
}

# k=0 carries no examples to recall, so it is excluded on every arm.
K_LIST: List[int] = sorted({int(k) for k in DS.k_list if int(k) > 0})
K_MAX = max(K_LIST) if K_LIST else 0
M_PER_MODEL = K_MAX

EMB_ROOT = DS.emb_root()
EMB_METHODS = ["cohere", "sonar", "labse", "MiniLM", "e5"]
BM25_METHOD_NAME = "bm25"
RRF_K0 = 60
FRAGMENTSHOT_MAX_FRAGMENT_SIZE = 5
FRAGMENTSHOT_OVERLAPS = False
DEVTEST_N: Optional[int] = 100
TOPK_SIM_CHUNK = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Committed, seeded selection artifacts (same files the generation used).
RANDOM_SELECTION_FILEPATHS: List[str] = [
    os.environ.get(
        "RTRACE_RANDOM_SELECTIONS",
        os.path.join("data", "random_pool_selections", f"{DS.key}_eng_random_pool.json"),
    ),
]

METHOD_CONFIGS: List[Dict[str, Any]] = [
    {
        "method_key": "rrf",
        "method_label": "RRF",
        "ensemble_method": "rrf",
        "reasoning_root_suffix": "",
        "trace_filename": "k{K}_rrf_template11_reasoning.jsonl",
    },
    {
        "method_key": "random",
        "method_label": "Random",
        "ensemble_method": "random_pool",
        "reasoning_root_suffix": "_random",
        "trace_filename": "k{K}_random_pool_template11_reasoning.jsonl",
    },
    {
        "method_key": "sentinel",
        "method_label": "Sentinel",
        "ensemble_method": "pool_sentinel_src_rerank",
        "reasoning_root_suffix": "_sentinel",
        "trace_filename": "k{K}_pool_sentinel_src_rerank_template11_reasoning.jsonl",
    },
    {
        "method_key": "edit_dist",
        "method_label": "Edit Distance",
        "ensemble_method": "edit_dist",
        "reasoning_root_suffix": "_edit_dist",
        "trace_filename": "k{K}_edit_dist_template11_reasoning.jsonl",
    },
]

METHOD_KEYS_TO_RUN: List[str] = ["rrf", "random", "sentinel", "edit_dist"]

OUTPUT_ROOT = os.environ.get("RTRACE_RECALL_DIR", DS.analysis_dir("trace_example_recall"))

# When True, methods whose three per-method CSVs already exist under
# OUTPUT_ROOT/<method_key>/ are skipped: the cached frames are loaded
# instead of re-running analyze_one_method. Only methods missing from the
# cache are computed this run; the global *_all_methods.csv files and the
# chrF++ correlation step are always rewritten from the resulting union.
# Set to False to force a full recompute.
LOAD_FROM_CSV: bool = True

# ── chrF++ correlation inputs ──
# Source of truth for the per-(model, run, direction, k) chrF++ translation
# scores: the raw_scores_long_chrfpp.csv produced by
# paired_bootstrap_chrfpp_significance.py. We reuse those numbers as-is
# (no recomputation) and join them against the per-cell mean_recall here.
CHRFPP_SCORES_CSV: str = os.environ.get(
    "RTRACE_CHRFPP_SCORES_CSV",
    os.path.join(
        DS.analysis_dir("eval_plots_paper_final_paired_bs_chrfpp"), "raw_scores_long_chrfpp.csv"
    ),
)

# Permissive — match either spelling the chrF script may have written.
CHRFPP_METRIC_NAMES: Tuple[str, ...] = ("chrfpp", "chrf++")

# Maps our method_key → the `run_key` the chrF CSV uses for the reasoning-ON
# row of that method. RRF uses the bare "reasoning_on" key (no suffix).
METHOD_KEY_TO_REASONING_ON_RUN_KEY: Dict[str, str] = {
    "rrf": "reasoning_on",
    "random": "reasoning_on_random",
    "sentinel": "reasoning_on_sentinel",
    "edit_dist": "reasoning_on_edit_dist",
}

# Field-priority order used when extracting reasoning text from a JSONL line.
# An explicit `null` value on any of these keys means the trace is unusable
# for that sentence and is skipped (handled by `extract_reasoning_text`).
TRACE_TEXT_KEYS_PRIORITY: List[str] = [
    "reasoning",
    "reasoning_trace",
    "reasoning_content",
    "reasoning_text",
    "thinking",
    "thoughts",
    "cot",
    "trace",
]

# Per-example long-form columns. example_position is preserved as a plain
# metadata column (still useful for "which slot was most reused") but no
# longer drives any phrase-matching logic.
RECALL_LONG_COLUMNS: List[str] = [
    "model_key",
    "model_display",
    "model_family_root",
    "method_key",
    "method_label",
    "ensemble_method",
    "src_lang",
    "tgt_lang",
    "direction",
    "k",
    "devtest_index",
    "trace_path",
    "trace_found",
    "trace_text_length",
    "query_src_sentence",
    "example_position",
    "example_dev_index",
    "src_example_sentence",
    "tgt_example_sentence",
    "src_recall",
    "tgt_recall",
    "recall",
]

# One row per reasoning trace (i.e. per query): mean/max recall across the
# k example slots, plus identifiers/trace metadata.
TRACE_SUMMARY_COLUMNS: List[str] = [
    "model_key",
    "model_display",
    "model_family_root",
    "method_key",
    "method_label",
    "ensemble_method",
    "src_lang",
    "tgt_lang",
    "direction",
    "k",
    "devtest_index",
    "trace_path",
    "trace_found",
    "trace_text_length",
    "query_src_sentence",
    "n_examples",
    "mean_recall",
    "mean_src_recall",
    "mean_tgt_recall",
    "max_recall",
]

# One row per (model, method, direction, k) — grand mean of recall over every
# example slot in the cell (0–100). `mean_recall` is the headline for Table 1.
RECALL_RATE_COLUMNS: List[str] = [
    "model_key",
    "model_display",
    "model_family_root",
    "method_key",
    "method_label",
    "ensemble_method",
    "src_lang",
    "tgt_lang",
    "direction",
    "k",
    "n_query_sentences",
    "n_examples_total",
    "mean_recall",
    "mean_src_recall",
    "mean_tgt_recall",
]

# Per-cell rows used as the correlation input: one row per
# (model, method, direction, k) holding both mean_recall (example usage)
# and the chrF++ translation score from the existing CSV.
CORRELATION_JOIN_COLUMNS: List[str] = [
    "model_key",
    "model_display",
    "model_family_root",
    "method_key",
    "method_label",
    "ensemble_method",
    "src_lang",
    "tgt_lang",
    "direction",
    "k",
    "n_query_sentences",
    "n_examples_total",
    "mean_recall",
    "mean_src_recall",
    "mean_tgt_recall",
    "chrfpp_run_key",
    "chrfpp_score",
    "chrfpp_path",
]

# One row per breakdown × group: Pearson + Spearman correlation of
# mean_recall vs chrfpp_score across the cells in that group.
CORRELATION_SUMMARY_COLUMNS: List[str] = [
    "axis",
    "group",
    "n",
    "pearson_r",
    "pearson_p",
    "spearman_r",
    "spearman_p",
]


# ─────────────────────────────────────────────────────────────────────────────
# chrF recall scoring layer (mirrors eval_pipeline.py's CHRF(word_order=2))
# ─────────────────────────────────────────────────────────────────────────────


class CHRFRecall(CHRF):
    """Identical extraction/preprocessing to CHRF(word_order=2); `.score`
    returns chrR (the recall component) instead of the F-β score.

    The override mirrors sacrebleu's `_compute_f_score` slice/guard verbatim —
    `(n_hyp, n_ref, n_match) = statistics[3 * i: 3 * i + 3]` and
    `if n_hyp > 0 and n_ref > 0` — so every other config knob (char_order=6,
    word_order=2, lowercase=False, whitespace=False, eps_smoothing=False) is
    inherited untouched. If a future sacrebleu release changes that slice
    layout, this override needs revisiting.
    """

    def _compute_f_score(self, statistics):
        avg_rec, effective_order = 0.0, 0
        for i in range(self.order):
            n_hyp, n_ref, n_match = statistics[3 * i: 3 * i + 3]
            if n_hyp > 0 and n_ref > 0:
                avg_rec += n_match / n_ref
                effective_order += 1
        return 0.0 if effective_order == 0 else 100.0 * avg_rec / effective_order


_CHRF_RECALL = CHRFRecall(word_order=2)  # parallel to eval_pipeline.chrfpp


def chrf_recall(example: str, trace: str) -> float:
    """
    Purpose: Compute chrF character n-gram recall of `example` reproduced in `trace`.
    Inputs: The example sentence (reference) and the reasoning trace (hypothesis).
    Outputs: chrR on a 0–100 scale, or NaN when either side is empty/missing.
    """
    if not example or not trace:
        return float("nan")
    return float(_CHRF_RECALL.sentence_score(trace, [example]).score)


def score_record_for_example(
    trace_text: str,
    source_sentence: str,
    target_sentence: str,
) -> Dict[str, float]:
    """
    Purpose: Per-example chrF recall on source and target sides plus the per-slot max ("reused on either side").
    Inputs: The full reasoning trace, the example's English source sentence, and its translation.
    Outputs: A dict with src_recall, tgt_recall, and recall = max(src_recall, tgt_recall), each in [0, 100] or NaN.
    """
    src_r = chrf_recall(source_sentence, trace_text)
    tgt_r = chrf_recall(target_sentence, trace_text)
    if np.isnan(src_r) and np.isnan(tgt_r):
        rec = float("nan")
    else:
        rec = float(np.nanmax([src_r, tgt_r]))
    return {"src_recall": src_r, "tgt_recall": tgt_r, "recall": rec}


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────


def ensure_dir(path: str) -> None:
    """
    Purpose: Create a directory path if it does not already exist.
    Inputs: A filesystem path string.
    Outputs: None; the directory is created in place when needed.
    """
    os.makedirs(path, exist_ok=True)


def slugify(text: str) -> str:
    """
    Purpose: Convert text into a filesystem-safe slug.
    Inputs: Any short label or title string.
    Outputs: A cleaned string containing only safe filename characters.
    """
    text = re.sub(r"\s+", "_", text.strip())
    return re.sub(r"[^A-Za-z0-9_\-]+", "", text)


def direction_folder_name(src_lang: str, tgt_lang: str) -> str:
    """
    Purpose: Build the same direction folder slug used by your eval pipeline.
    Inputs: FLORES source language code and target language code.
    Outputs: A folder name such as 'eng_to_wol' or 'eng_to_swh'.
    """
    return f"{src_lang.split('_')[0].lower()}_to_{tgt_lang.split('_')[0].lower()}"


def _apply_limit(n: int, limit_m: Optional[int]) -> int:
    """
    Purpose: Apply an optional sentence ceiling to a dataset length.
    Inputs: A raw length and an optional integer ceiling.
    Outputs: The truncated length that should be used downstream.
    """
    if limit_m is None:
        return n
    m = int(limit_m)
    return 0 if m <= 0 else min(n, m)


def _list_immediate_subdirs(parent: str) -> List[str]:
    """
    Purpose: List the immediate child directories of a parent path.
    Inputs: An absolute or relative parent directory path.
    Outputs: A sorted list of child directory names, or [] when parent is missing.
    """
    try:
        return sorted(
            [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
        )
    except FileNotFoundError:
        return []


def resolve_model_dirname(base_dir: str, model_key: str) -> Optional[str]:
    """
    Purpose: Resolve the actual on-disk model folder name for a configured model key.
    Inputs: A reasoning-trace root such as reasoning_traces_Mistral_all_random and a model_key like 'ministral_8b'.
    Outputs: The matching subdir name (case-insensitive fallback), or None if no candidate folder exists.
    """
    candidate = os.path.join(base_dir, model_key)
    if os.path.isdir(candidate):
        return model_key
    norm = model_key.lower()
    matches = [d for d in _list_immediate_subdirs(base_dir) if d.lower() == norm]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return sorted(matches)[0]
    return None


def method_config_map() -> Dict[str, Dict[str, Any]]:
    """
    Purpose: Index method configs by method_key for easy filtering.
    Inputs: None; this uses the METHOD_CONFIGS constant above.
    Outputs: A dict from method_key to its full config dictionary.
    """
    return {cfg["method_key"]: cfg for cfg in METHOD_CONFIGS}


def selected_method_configs() -> List[Dict[str, Any]]:
    """
    Purpose: Return only the method configs requested for this run.
    Inputs: The METHOD_KEYS_TO_RUN config constant.
    Outputs: An ordered list of active method configuration dicts.
    """
    cfg_map = method_config_map()
    return [cfg_map[key] for key in METHOD_KEYS_TO_RUN]


def reasoning_root_for_model(model_key: str, reasoning_root_suffix: str) -> str:
    """
    Purpose: Build the reasoning-trace root directory for one model family/method.
    Inputs: A model key and the method-specific suffix such as '' or '_random'.
    Outputs: The dataset-appropriate trace root (legacy flores 'all' layout, or
             the per-method per-state layout the WMT sbatch scripts write).
    """
    family_root = MODEL_FAMILY_ROOT[model_key]
    return DS.reasoning_trace_root(family_root, reasoning_root_suffix)


# ─────────────────────────────────────────────────────────────────────────────
# Trace reading helpers
# ─────────────────────────────────────────────────────────────────────────────


def read_jsonl_objects(path: str) -> List[Dict[str, Any]]:
    """
    Purpose: Read a JSONL file into a list of dictionaries.
    Inputs: A path to a JSONL file with one JSON object per line.
    Outputs: A list of parsed dictionaries, skipping blank lines.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_reasoning_text(obj: Dict[str, Any]) -> Optional[str]:
    """
    Purpose: Extract the reasoning trace string from a JSONL record robustly.
    Inputs: One parsed JSON object from a reasoning trace JSONL file.
    Outputs: The best-guess reasoning text string, or None when the trace is unusable.
    """
    for key in TRACE_TEXT_KEYS_PRIORITY:
        if key in obj and obj.get(key) is None:
            return None

    for key in TRACE_TEXT_KEYS_PRIORITY:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value

    for key, value in obj.items():
        if value is None and any(tok in key.lower() for tok in ["reason", "think", "trace", "cot"]):
            return None
        if isinstance(value, str) and value.strip() and "translation" not in key.lower():
            if any(tok in key.lower() for tok in ["reason", "think", "trace", "cot"]):
                return value

    for value in obj.values():
        if isinstance(value, str) and value.strip():
            return value

    return ""


def load_reasoning_texts(path: str) -> List[Optional[str]]:
    """
    Purpose: Load a reasoning JSONL file and extract trace text per line.
    Inputs: A reasoning JSONL path produced by the generation pipeline.
    Outputs: A list of optional reasoning trace strings aligned to file line order.
    """
    records = read_jsonl_objects(path)
    return [extract_reasoning_text(obj) for obj in records]


# ─────────────────────────────────────────────────────────────────────────────
# Example reconstruction helpers
# ─────────────────────────────────────────────────────────────────────────────


def build_jobs_for_method(
    ensemble_method: str,
    src_lang_codes: Sequence[str],
    tgt_lang_codes: Sequence[str],
) -> List[Dict[str, Any]]:
    """
    Purpose: Recreate the per-direction job structure from the reasoning script.
    Inputs: An ensemble method and the source/target FLORES language lists.
    Outputs: A list of job dicts holding src/tgt dev sets and selected examples.
    """
    use_fragment_shot, ensemble_submethod = parse_ensemble_method(ensemble_method)

    if M_PER_MODEL < K_MAX:
        raise ValueError(f"M_PER_MODEL must be >= K_MAX (got {M_PER_MODEL} < {K_MAX}).")

    if ensemble_submethod == "random_pool" and len(RANDOM_SELECTION_FILEPATHS) != len(src_lang_codes):
        raise ValueError(
            "RANDOM_SELECTION_FILEPATHS must match SRC_LANG_CODES length when using random_pool."
        )

    jobs: List[Dict[str, Any]] = []

    for src_idx, src_lang in enumerate(src_lang_codes):
        src_name = LANG_NAME.get(src_lang, src_lang)
        src_dir = LANG_DIRNAME.get(src_lang, src_lang)

        random_selection_path = (
            RANDOM_SELECTION_FILEPATHS[src_idx] if ensemble_submethod == "random_pool" else None
        )

        src_data = DS.load_sentences(src_lang)
        src_dev = src_data["dev"]
        src_devtest_full = src_data["devtest"]
        src_devtest = _apply_devtest_limit(src_devtest_full, DEVTEST_N)

        n_dev = len(src_dev)
        n_devtest_full = len(src_devtest_full)
        n_devtest = len(src_devtest)

        if n_devtest == 0:
            raise ValueError(f"DEVTEST_N produced empty devtest for src_lang={src_lang}.")

        emb_full = load_retrieval_embeddings(
            emb_root=EMB_ROOT,
            methods=EMB_METHODS + ([BM25_METHOD_NAME] if ensemble_submethod == "pool_bm25_rerank" else []),
            lang_code=src_lang,
            n_dev=n_dev,
            n_devtest=n_devtest_full,
        )

        if ensemble_submethod == "pool_bm25_rerank":
            _, bm25_devtest_scores_full = emb_full[BM25_METHOD_NAME]
            bm25_devtest_scores = bm25_devtest_scores_full[:n_devtest, :]
        else:
            bm25_devtest_scores = None

        if ensemble_submethod == "pool_sentinel_src_rerank":
            sentinel_src_scores = load_sentinel_src_scores(
                emb_root=EMB_ROOT,
                lang_code=src_lang,
                n_dev=n_dev,
            )
        else:
            sentinel_src_scores = None

        per_method_idx_m_full: Dict[str, np.ndarray] = {}

        if not use_fragment_shot:
            for emb_method in EMB_METHODS:
                dev_emb, devtest_emb_full = emb_full[emb_method]
                devtest_emb = devtest_emb_full[:n_devtest, :]
                idx, _ = topk_cosine_indices_and_scores(
                    dev_emb,
                    devtest_emb,
                    M_PER_MODEL,
                    device=DEVICE,
                    torch_dtype=TORCH_DTYPE,
                    chunk=TOPK_SIM_CHUNK,
                )
                per_method_idx_m_full[emb_method] = idx

        for tgt_lang in tgt_lang_codes:
            if tgt_lang == src_lang:
                continue

            tgt_name = LANG_NAME.get(tgt_lang, tgt_lang)
            tgt_dir = LANG_DIRNAME.get(tgt_lang, tgt_lang)

            tgt_data = DS.load_sentences(tgt_lang)
            tgt_dev = tgt_data["dev"]

            if use_fragment_shot:
                pools = build_fragmentshot_pools(
                    src_dev=src_dev,
                    tgt_dev=tgt_dev,
                    src_devtest=src_devtest,
                    k_min=M_PER_MODEL,
                    max_fragment_size=FRAGMENTSHOT_MAX_FRAGMENT_SIZE,
                    overlaps=FRAGMENTSHOT_OVERLAPS,
                )

                per_method_idx_m: Dict[str, np.ndarray] = {}
                for emb_method in EMB_METHODS:
                    dev_emb, devtest_emb_full = emb_full[emb_method]
                    devtest_emb = devtest_emb_full[:n_devtest, :]
                    idx = topk_cosine_indices_from_pools(
                        dev_emb,
                        devtest_emb,
                        pools,
                        M_PER_MODEL,
                        device=DEVICE,
                        torch_dtype=TORCH_DTYPE,
                    )
                    per_method_idx_m[emb_method] = idx
            else:
                per_method_idx_m = per_method_idx_m_full

            final_indices_by_k: Dict[int, List[List[int]]] = {}
            for k in K_LIST:
                if k == 0:
                    final_indices_by_k[k] = [[] for _ in range(n_devtest)]
                    continue

                per_idx_k = {m: per_method_idx_m[m][:, :k] for m in EMB_METHODS}
                final_indices_by_k[k] = ensemble_topk_dispatch(
                    ensemble_submethod,
                    per_idx_k,
                    EMB_METHODS,
                    k,
                    rrf_k0=RRF_K0,
                    bm25_devtest_scores=bm25_devtest_scores,
                    random_selection_path=random_selection_path,
                    random_source_idx=per_method_idx_m,
                    random_dev_size=len(src_dev),
                    sentinel_src_scores=sentinel_src_scores,
                )

            jobs.append(
                {
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "src_name": src_name,
                    "tgt_name": tgt_name,
                    "direction_dir": f"{src_dir}_to_{tgt_dir}",
                    "src_dev": src_dev,
                    "src_devtest": src_devtest,
                    "tgt_dev": tgt_dev,
                    "final_indices_by_k": final_indices_by_k,
                }
            )

    return jobs


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────


def summarize_trace_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Purpose: Aggregate per-example recall rows into one trace-level summary.
    Inputs: All long-form rows corresponding to a single reasoning trace.
    Outputs: A one-row summary dict with n_examples and mean/max recall fields.
    """
    if not rows:
        raise ValueError("summarize_trace_rows received an empty row list.")

    src_arr = np.array([r["src_recall"] for r in rows], dtype=np.float64)
    tgt_arr = np.array([r["tgt_recall"] for r in rows], dtype=np.float64)
    rec_arr = np.array([r["recall"] for r in rows], dtype=np.float64)

    # `np.nanmean` / `np.nanmax` would warn (and return NaN) if every slot is
    # NaN — guard against that for traces whose every example happened to be
    # empty (vanishingly rare in practice, but cheap to handle).
    def _safe_nanmean(a: np.ndarray) -> float:
        return float(np.nanmean(a)) if np.any(~np.isnan(a)) else float("nan")

    def _safe_nanmax(a: np.ndarray) -> float:
        return float(np.nanmax(a)) if np.any(~np.isnan(a)) else float("nan")

    template = rows[0]
    return {
        "model_key": template["model_key"],
        "model_display": template["model_display"],
        "model_family_root": template["model_family_root"],
        "method_key": template["method_key"],
        "method_label": template["method_label"],
        "ensemble_method": template["ensemble_method"],
        "src_lang": template["src_lang"],
        "tgt_lang": template["tgt_lang"],
        "direction": template["direction"],
        "k": template["k"],
        "devtest_index": template["devtest_index"],
        "trace_path": template["trace_path"],
        "trace_found": template["trace_found"],
        "trace_text_length": template["trace_text_length"],
        "query_src_sentence": template["query_src_sentence"],
        "n_examples": len(rows),
        "mean_recall": _safe_nanmean(rec_arr),
        "mean_src_recall": _safe_nanmean(src_arr),
        "mean_tgt_recall": _safe_nanmean(tgt_arr),
        "max_recall": _safe_nanmax(rec_arr),
    }


def compute_recall_rate_df(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose: Aggregate per-example recall rows into per-(model, method, direction, k) means.
    Inputs: The long-form per-example DataFrame from analyze_one_method.
    Outputs: One row per (model, method, direction, k) with grand-mean recall and slot counts.

    Aggregating directly from the long DataFrame (rather than from the per-trace
    summary) avoids any fixed-k assumption and produces the unambiguous grand
    mean across every example slot in the cell.
    """
    if df_long.empty:
        return pd.DataFrame(columns=RECALL_RATE_COLUMNS)

    group_cols = [
        "model_key",
        "model_display",
        "model_family_root",
        "method_key",
        "method_label",
        "ensemble_method",
        "src_lang",
        "tgt_lang",
        "direction",
        "k",
    ]

    df = df_long.copy()
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            n_query_sentences=("devtest_index", "nunique"),
            n_examples_total=("recall", "size"),
            mean_recall=("recall", "mean"),
            mean_src_recall=("src_recall", "mean"),
            mean_tgt_recall=("tgt_recall", "mean"),
        )
    )

    return agg.reindex(columns=RECALL_RATE_COLUMNS)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────


def analyze_one_method(
    method_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Purpose: Run chrF-recall scoring of in-context example reuse for one ensembling method.
    Inputs: One active method configuration dict from METHOD_CONFIGS.
    Outputs: A long-form per-example DataFrame, a trace-level summary DataFrame,
             and an aggregated per-(model, method, direction, k) recall DataFrame.
    """
    jobs = build_jobs_for_method(
        ensemble_method=method_cfg["ensemble_method"],
        src_lang_codes=SRC_LANG_CODES,
        tgt_lang_codes=TGT_LANG_CODES,
    )

    long_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for model_key, model_display in MODELS.items():
        model_family_root = MODEL_FAMILY_ROOT[model_key]
        reasoning_root = reasoning_root_for_model(
            model_key=model_key,
            reasoning_root_suffix=method_cfg["reasoning_root_suffix"],
        )

        # Resolve the on-disk model folder name once per (model, method).
        # Falls back to the configured model_key if no folder is present yet —
        # the per-direction loop below already handles missing trace files.
        model_dirname = resolve_model_dirname(reasoning_root, model_key) or model_key

        print(
            f"\n[{method_cfg['method_label']}] model={model_key} "
            f"root={reasoning_root} dirname={model_dirname}"
        )
        for job in jobs:
            direction = str(job["direction_dir"])
            src_lang = str(job["src_lang"])
            tgt_lang = str(job["tgt_lang"])
            src_dev = job["src_dev"]  # type: ignore[assignment]
            src_devtest = job["src_devtest"]  # type: ignore[assignment]
            tgt_dev = job["tgt_dev"]  # type: ignore[assignment]
            final_indices_by_k = job["final_indices_by_k"]  # type: ignore[assignment]

            for k in K_LIST:
                trace_dir = os.path.join(reasoning_root, model_dirname, direction)
                trace_path = os.path.join(
                    trace_dir,
                    method_cfg["trace_filename"].format(K=k),
                )

                if not os.path.exists(trace_path):
                    print(f"  missing trace file: {trace_path}")
                    continue

                trace_texts = load_reasoning_texts(trace_path)
                aligned_n = min(len(src_devtest), len(final_indices_by_k[k]), len(trace_texts))
                usable_indices = [i for i in range(aligned_n) if trace_texts[i] is not None]

                print(
                    f"  {model_key} | {direction} | k={k} "
                    f"-> traces={len(trace_texts)} usable={len(usable_indices)}"
                )

                for devtest_index in usable_indices:
                    trace_text = trace_texts[devtest_index] or ""
                    query_src_sentence = src_devtest[devtest_index]
                    example_dev_indices = final_indices_by_k[k][devtest_index]

                    trace_example_rows: List[Dict[str, Any]] = []

                    for pos0, example_dev_index in enumerate(example_dev_indices):
                        position_1based = pos0 + 1
                        src_example_sentence = src_dev[example_dev_index]
                        tgt_example_sentence = tgt_dev[example_dev_index]

                        score_info = score_record_for_example(
                            trace_text=trace_text,
                            source_sentence=src_example_sentence,
                            target_sentence=tgt_example_sentence,
                        )

                        row = {
                            "model_key": model_key,
                            "model_display": model_display,
                            "model_family_root": model_family_root,
                            "method_key": method_cfg["method_key"],
                            "method_label": method_cfg["method_label"],
                            "ensemble_method": method_cfg["ensemble_method"],
                            "src_lang": src_lang,
                            "tgt_lang": tgt_lang,
                            "direction": direction,
                            "k": k,
                            "devtest_index": devtest_index,
                            "trace_path": trace_path,
                            "trace_found": True,
                            "trace_text_length": len(trace_text),
                            "query_src_sentence": query_src_sentence,
                            "example_position": position_1based,
                            "example_dev_index": int(example_dev_index),
                            "src_example_sentence": src_example_sentence,
                            "tgt_example_sentence": tgt_example_sentence,
                            **score_info,
                        }
                        trace_example_rows.append(row)
                        long_rows.append(row)

                    if trace_example_rows:
                        summary_rows.append(summarize_trace_rows(trace_example_rows))

    df_long = pd.DataFrame(long_rows, columns=RECALL_LONG_COLUMNS)
    df_summary = pd.DataFrame(summary_rows, columns=TRACE_SUMMARY_COLUMNS)
    df_recall = compute_recall_rate_df(df_long)
    return df_long, df_summary, df_recall


def _method_output_paths(method_cfg: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Purpose: Compute the per-method directory and the three CSV paths under it.
    Inputs: One method configuration dict.
    Outputs: (method_dir, long_path, summary_path, recall_path) for that method.
    """
    method_dir = os.path.join(OUTPUT_ROOT, slugify(method_cfg["method_key"]))
    return (
        method_dir,
        os.path.join(method_dir, "trace_example_recall_long.csv"),
        os.path.join(method_dir, "trace_recall_summary.csv"),
        os.path.join(method_dir, "sentence_recall_rates.csv"),
    )


def export_method_outputs(
    df_long: pd.DataFrame,
    df_summary: pd.DataFrame,
    df_recall: pd.DataFrame,
    method_cfg: Dict[str, Any],
) -> None:
    """
    Purpose: Save long-form, trace-summary, and recall-rate CSV outputs for one method.
    Inputs: The three DataFrames produced by analyze_one_method and the method config.
    Outputs: CSV files in a method-specific output directory under OUTPUT_ROOT.
    """
    method_dir, long_path, summary_path, recall_path = _method_output_paths(method_cfg)
    ensure_dir(method_dir)

    df_long.to_csv(long_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    df_recall.to_csv(recall_path, index=False)


def _coerce_numeric_inplace(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """
    Purpose: Coerce listed columns to numeric in place (NaN-on-failure), if present.
    Inputs: A DataFrame and an iterable of candidate column names.
    Outputs: None; the DataFrame is mutated in place.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def load_cached_method_outputs(
    method_cfg: Dict[str, Any],
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Purpose: Reload one method's three per-method CSVs from a prior run, with the same dtypes downstream code expects.
    Inputs: One method configuration dict.
    Outputs: (df_long, df_summary, df_recall) if all three CSVs exist, else None.
    """
    _, long_path, summary_path, recall_path = _method_output_paths(method_cfg)
    if not (
        os.path.exists(long_path)
        and os.path.exists(summary_path)
        and os.path.exists(recall_path)
    ):
        return None

    df_long = pd.read_csv(long_path)
    df_summary = pd.read_csv(summary_path)
    df_recall = pd.read_csv(recall_path)

    # Same type cleanup the other LOAD_FROM_CSV sites do: integer k, numeric
    # score columns so any downstream math (correlation, np.nanmean) keeps
    # working on the reloaded frames.
    for df in (df_long, df_summary, df_recall):
        if "k" in df.columns:
            df["k"] = df["k"].astype(int)
    _coerce_numeric_inplace(df_long, ["src_recall", "tgt_recall", "recall"])
    _coerce_numeric_inplace(
        df_summary,
        ["mean_recall", "mean_src_recall", "mean_tgt_recall", "max_recall"],
    )
    _coerce_numeric_inplace(
        df_recall,
        ["mean_recall", "mean_src_recall", "mean_tgt_recall"],
    )
    return df_long, df_summary, df_recall


def export_global_outputs(
    df_long_all: pd.DataFrame,
    df_summary_all: pd.DataFrame,
    df_recall_all: pd.DataFrame,
) -> None:
    """
    Purpose: Save combined outputs across every configured method and model.
    Inputs: Concatenated long-form, summary, and recall-rate DataFrames.
    Outputs: Global CSV files in OUTPUT_ROOT for downstream aggregation.
    """
    ensure_dir(OUTPUT_ROOT)

    df_long_all.to_csv(
        os.path.join(OUTPUT_ROOT, "trace_example_recall_long_all_methods.csv"),
        index=False,
    )
    df_summary_all.to_csv(
        os.path.join(OUTPUT_ROOT, "trace_recall_summary_all_methods.csv"),
        index=False,
    )
    df_recall_all.to_csv(
        os.path.join(OUTPUT_ROOT, "sentence_recall_rates_all_methods.csv"),
        index=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# chrF++ score × recall correlation (reuses translation chrF++ scores
# from paired_bootstrap_chrfpp_significance.py — no recomputation)
# ─────────────────────────────────────────────────────────────────────────────


def load_chrfpp_scores(path: str) -> pd.DataFrame:
    """
    Purpose: Load per-file chrF++ scores produced by paired_bootstrap_chrfpp_significance.py.
    Inputs: Path to raw_scores_long_chrfpp.csv (schema: model_key, run_key, src_lang, tgt_lang, direction, k, metric, score, path).
    Outputs: A DataFrame filtered to reasoning-ON × chrF++ rows, with columns ready to merge against the per-cell recall df.
    """
    if not os.path.exists(path):
        print(f"  chrF++ score CSV not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    keep = df["metric"].astype(str).str.lower().isin(set(CHRFPP_METRIC_NAMES))
    keep &= df["run_key"].isin(set(METHOD_KEY_TO_REASONING_ON_RUN_KEY.values()))
    cols = ["model_key", "run_key", "src_lang", "tgt_lang", "direction", "k", "score", "path"]
    out = df.loc[keep, cols].rename(
        columns={
            "run_key": "chrfpp_run_key",
            "score": "chrfpp_score",
            "path": "chrfpp_path",
        }
    )
    return out


def join_recall_with_chrfpp(
    df_recall: pd.DataFrame,
    df_chrfpp: pd.DataFrame,
) -> pd.DataFrame:
    """
    Purpose: Join per-(model, method, direction, k) recall with the matching reasoning-ON chrF++ score.
    Inputs: The aggregated recall DataFrame and the loaded chrF++ score DataFrame.
    Outputs: A merged DataFrame, one row per cell, with both mean_recall and chrfpp_score.
    """
    if df_recall.empty or df_chrfpp.empty:
        return pd.DataFrame(columns=CORRELATION_JOIN_COLUMNS)

    df_recall = df_recall.copy()
    df_recall["chrfpp_run_key"] = df_recall["method_key"].map(METHOD_KEY_TO_REASONING_ON_RUN_KEY)

    merged = df_recall.merge(
        df_chrfpp,
        on=["model_key", "chrfpp_run_key", "src_lang", "tgt_lang", "direction", "k"],
        how="inner",
    )
    return merged.reindex(columns=CORRELATION_JOIN_COLUMNS)


def _pearson_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Purpose: Pearson + Spearman correlations and two-sided p-values, NaN-safe.
    Inputs: Two equal-length arrays (any NaN-aligned entries are dropped before correlating).
    Outputs: (pearson_r, pearson_p, spearman_r, spearman_p); all NaN when n < 3 or x/y is constant.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    xv = x[mask]
    yv = y[mask]
    # scipy returns NaN with a warning if either side is constant; short-circuit
    # cleanly to NaN so the CSV stays interpretable.
    if np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return float("nan"), float("nan"), float("nan"), float("nan")
    pr = pearsonr(xv, yv)
    sr = spearmanr(xv, yv)
    return (
        float(pr.statistic),
        float(pr.pvalue),
        float(sr.statistic),
        float(sr.pvalue),
    )


def compute_correlation_stats(df_joined: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose: Pearson + Spearman correlation of mean_recall vs chrfpp_score across several breakdowns.
    Inputs: The merged per-cell recall × chrF++ DataFrame.
    Outputs: One row per (axis, group): "overall", "model", "method", "direction", "k".
    """
    if df_joined.empty:
        return pd.DataFrame(columns=CORRELATION_SUMMARY_COLUMNS)

    rows: List[Dict[str, Any]] = []

    def _stat(axis: str, group: Any, sub: pd.DataFrame) -> None:
        x = sub["mean_recall"].to_numpy(dtype=np.float64)
        y = sub["chrfpp_score"].to_numpy(dtype=np.float64)
        pr, pp, sr, sp = _pearson_spearman(x, y)
        rows.append({
            "axis": axis,
            "group": str(group),
            "n": int(len(sub)),
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_r": sr,
            "spearman_p": sp,
        })

    _stat("overall", "all", df_joined)
    for axis_name, axis_col in [
        ("model", "model_key"),
        ("method", "method_key"),
        ("direction", "direction"),
        ("k", "k"),
    ]:
        for grp, sub in df_joined.groupby(axis_col, sort=True):
            _stat(axis_name, grp, sub)

    # ── Per-method-stratified breakdowns: within each method's cells, also
    #    correlate by model / direction / k. Lets the confound check ("is the
    #    overall negative trend really within-method, or is it a between-
    #    method artefact?") be read off one CSV. ──
    for method_key, method_sub in df_joined.groupby("method_key", sort=True):
        for axis_name, axis_col in [
            ("method|model", "model_key"),
            ("method|direction", "direction"),
            ("method|k", "k"),
        ]:
            for grp, sub in method_sub.groupby(axis_col, sort=True):
                _stat(axis_name, f"{method_key}|{grp}", sub)

    return pd.DataFrame(rows, columns=CORRELATION_SUMMARY_COLUMNS)


def export_correlation_outputs(
    df_corr_joined: pd.DataFrame,
    df_corr_summary: pd.DataFrame,
) -> None:
    """
    Purpose: Save the per-cell join and the correlation summary.
    Inputs: The merged per-cell DataFrame and the correlation-summary DataFrame.
    Outputs: chrfpp_recall_correlation_long.csv and chrfpp_recall_correlation_summary.csv under OUTPUT_ROOT.
    """
    ensure_dir(OUTPUT_ROOT)
    df_corr_joined.to_csv(
        os.path.join(OUTPUT_ROOT, "chrfpp_recall_correlation_long.csv"),
        index=False,
    )
    df_corr_summary.to_csv(
        os.path.join(OUTPUT_ROOT, "chrfpp_recall_correlation_summary.csv"),
        index=False,
    )


def main() -> None:
    """
    Purpose: Run chrF-recall in-context-example-reuse analysis across methods, models, and directions, then correlate with chrF++ translation scores.
    Inputs: Config variables at the top of this file plus helper functions already in scope.
    Outputs: Per-method and global CSVs describing per-example chrF recall and aggregated recall rates, plus chrF++ correlation CSVs.
    """
    ensure_dir(OUTPUT_ROOT)

    all_long_parts: List[pd.DataFrame] = []
    all_summary_parts: List[pd.DataFrame] = []
    all_recall_parts: List[pd.DataFrame] = []

    # ── Per-method incremental cache. Same contract as the LOAD_FROM_CSV
    #    feature in the other scripts: if every per-method CSV for a method
    #    already exists under OUTPUT_ROOT/<method>/, reload it; otherwise
    #    run analyze_one_method and write the CSVs. ──
    methods = selected_method_configs()
    n_cached = 0
    n_computed = 0
    for method_cfg in methods:
        cached: Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = None
        if LOAD_FROM_CSV:
            cached = load_cached_method_outputs(method_cfg)

        if cached is not None:
            df_long, df_summary, df_recall = cached
            _, long_path, _, _ = _method_output_paths(method_cfg)
            print(
                f"\n[LOAD_FROM_CSV] Reusing cached method='{method_cfg['method_key']}' "
                f"from {long_path}\n"
                f"  long rows={len(df_long):,} | trace summaries={len(df_summary):,} "
                f"| (model, dir, k) cells={len(df_recall):,}"
            )
            n_cached += 1
        else:
            if LOAD_FROM_CSV:
                print(
                    f"\n[LOAD_FROM_CSV] No cache for method='{method_cfg['method_key']}' "
                    "— running analyze_one_method."
                )
            df_long, df_summary, df_recall = analyze_one_method(method_cfg)
            export_method_outputs(df_long, df_summary, df_recall, method_cfg)
            n_computed += 1

        all_long_parts.append(df_long)
        all_summary_parts.append(df_summary)
        all_recall_parts.append(df_recall)

    if LOAD_FROM_CSV:
        print(
            f"\n[LOAD_FROM_CSV] Reused {n_cached}/{len(methods)} method(s) from cache, "
            f"computed {n_computed}/{len(methods)} fresh."
        )

    df_long_all = (
        pd.concat(all_long_parts, ignore_index=True)
        if all_long_parts else pd.DataFrame(columns=RECALL_LONG_COLUMNS)
    )
    df_summary_all = (
        pd.concat(all_summary_parts, ignore_index=True)
        if all_summary_parts else pd.DataFrame(columns=TRACE_SUMMARY_COLUMNS)
    )
    df_recall_all = (
        pd.concat(all_recall_parts, ignore_index=True)
        if all_recall_parts else pd.DataFrame(columns=RECALL_RATE_COLUMNS)
    )

    export_global_outputs(df_long_all, df_summary_all, df_recall_all)

    # ── Correlate mean_recall (example usage) with chrF++ translation quality.
    #    Uses the existing per-file chrF++ scores from
    #    paired_bootstrap_chrfpp_significance.py — no recomputation. ──
    print(f"\nLoading chrF++ translation scores from {CHRFPP_SCORES_CSV}")
    df_chrfpp = load_chrfpp_scores(CHRFPP_SCORES_CSV)
    print(f"  loaded {len(df_chrfpp):,} chrF++ score rows (reasoning-ON only)")

    df_corr_joined = join_recall_with_chrfpp(df_recall_all, df_chrfpp)
    print(f"  joined {len(df_corr_joined):,} (model, method, direction, k) cells")

    df_corr_summary = compute_correlation_stats(df_corr_joined)
    export_correlation_outputs(df_corr_joined, df_corr_summary)

    def _fmt_corr_row(label: str, row: pd.Series) -> str:
        return (
            f"  {label:<28s} n={int(row['n']):4d}  "
            f"Pearson r={row['pearson_r']:+.4f} (p={row['pearson_p']:.3g})  "
            f"Spearman ρ={row['spearman_r']:+.4f} (p={row['spearman_p']:.3g})"
        )

    overall = df_corr_summary.loc[df_corr_summary["axis"] == "overall"]
    method_rows = df_corr_summary.loc[df_corr_summary["axis"] == "method"]
    if not overall.empty or not method_rows.empty:
        print("\nrecall vs chrF++ correlation:")
        if not overall.empty:
            print(_fmt_corr_row("overall", overall.iloc[0]))
        for _, row in method_rows.sort_values("group").iterrows():
            print(_fmt_corr_row(f"method={row['group']}", row))

    print(f"\nAll trace example-recall artefacts saved to: {OUTPUT_ROOT}/")
    print("Done.")


if __name__ == "__main__":
    main()
