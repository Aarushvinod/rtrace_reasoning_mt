#!/usr/bin/env python3
"""
chrfpp_per_sentence_analysis.py
────────────────────────────────
Per-sentence chrF++ scoring + battery of pairwise correlations + Type-II/III
regression ANOVA for the FLORES-200 multilingual MT pipeline.

The existing chrF++ CSV (`raw_scores_long_chrfpp.csv`) holds only corpus-
aggregated file scores; corpus chrF++ pools n-gram match/total stats across a
whole file and can't be decomposed into per-sentence numbers. This script
recomputes per-sentence chrF++ from the translation JSONLs once (cached to
`per_sentence_chrfpp.csv`), then joins it against the per-sentence token-count
CSV and the per-sentence trace-recall summaries to produce one master
sentence-level design matrix. From there:

  • Stage 3 — every pairwise Pearson + Spearman correlation across the four
    continuous columns (chrF++, mean_recall, reasoning_tokens, response_tokens),
    sliced by the same eight axes used in trace_example_recall.py
    (overall / model / method / direction / k + method-stratified variants).

  • Stage 4 — per (family, tgt_lang) Type-II ANOVA on ON-only data with
    two-way cluster-robust SEs (file_id, source_id):
        chrfpp ~ C(method_label) + C(k) + reasoning_tokens + mean_recall
    Run separately for the Mistral and Qwen families so family-level
    reasoning behaviour isn't averaged together.

  • Stage 4b — per (family, tgt_lang) linear mixed-effects model on the
    same ON-only data and the same fixed-effect formula, with a random
    intercept on `source_id` to absorb the cross-run dependence introduced
    by every model × method × k cell sharing the same English source
    sentence. The LMM is the correct primary read for per-sentence
    inference; the ANOVA in Stage 4 is kept as the conventional
    SS-decomposition view.

  • Stage 5 — per-(tgt_lang × reasoning_state) estimated marginal means from
    Model 1 (covariates at their means, other categoricals at modal levels).

  • Stage 6 — per-language Δ(chrF++ ON − OFF) ranked by mean reasoning-token
    usage on, plus a single Spearman ρ across the 7 languages (the
    "do high-token languages benefit more from reasoning?" check).

File layout — dataset-agnostic via src/common/dataset_registry.py
(RTRACE_DATASET selects the arm; paths below use its out_base + prefix):
  • translations:   {out_base}/{prefix}{Mistral|Qwen}_All_Reasoning_{On|Off}{|_random|_sentinel|_edit_dist}/<model>/<dir>/k{K}_*.jsonl
  • token counts:   {out_base}/{prefix}eval_token_counts/per_sentence_token_counts.csv
  • trace recall:   {out_base}/{prefix}trace_example_recall/<method_key>/trace_recall_summary.csv  (also _all_methods)
  • references:     DS.load_sentences(tgt_lang)["devtest"]  (standardized loader contract)

Output: {out_base}/{prefix}chrfpp_per_sentence_analysis/*.csv

Requirements:
  pip install sacrebleu pandas numpy scipy statsmodels
"""

import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sacrebleu.metrics import CHRF
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from src.common.plots import _collect_legend
from src.common.dataset_registry import get_dataset, filter_models


# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (mirrors token_count_inference_budget.py / trace_example_recall.py)
# ─────────────────────────────────────────────────────────────────────────────

# Dataset arm (RTRACE_DATASET: flores | wmt24pp). Every root, language list,
# and k value below derives from the spec (src/common/dataset_registry.py),
# so the script is dataset-agnostic given the {"dev","devtest"} loader contract.
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


def _model_root_dir_map(mistral_root: str, qwen_root: str) -> Dict[str, str]:
    """Map each model key to its translation root directory."""
    return {
        "ministral_8b": mistral_root,
        "ministral_14b": mistral_root,
        "magistral_small": mistral_root,
        "qwen3_8b": qwen_root,
        "qwen3_14b": qwen_root,
        "qwen3_32b": qwen_root,
    }


# (root-suffix, method_key, method label, filename pattern) per selection
# method — run keys/labels are built from these EXACTLY as the legacy literal
# lists did, so cached CSVs keyed on run_key/method_label stay valid.
_METHODS: List[Tuple[str, str, str, str]] = [
    ("",           "rrf",       "RRF",           "k{K}_rrf_template11.jsonl"),
    ("_random",    "random",    "Random",        "k{K}_random_pool_template11.jsonl"),
    ("_sentinel",  "sentinel",  "Sentinel",      "k{K}_pool_sentinel_src_rerank_template11.jsonl"),
    ("_edit_dist", "edit_dist", "Edit Distance", "k{K}_edit_dist_template11.jsonl"),
]


def _runs_for_state(state_title: str) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for suffix, method_key, method_label, filename_pattern in _METHODS:
        runs.append({
            "key": f"reasoning_{state_title.lower()}{suffix}",
            "label": f"{method_label} Reasoning {state_title}",
            "method_key": method_key,
            "method_label": method_label,
            "reasoning_state": state_title,
            "root_dir": _model_root_dir_map(
                DS.generation_root("Mistral", state_title, suffix),
                DS.generation_root("Qwen", state_title, suffix),
            ),
            "filename_pattern": filename_pattern,
        })
    return runs


GENERATION_RUNS_REASONING_ON: List[Dict[str, Any]] = _runs_for_state("On")
GENERATION_RUNS_REASONING_OFF: List[Dict[str, Any]] = _runs_for_state("Off")

SRC_LANGS: List[str] = list(DS.src_langs)
TGT_LANGS: List[str] = list(DS.tgt_langs)

LANG_DISPLAY: Dict[str, str] = dict(DS.lang_display)

K_LIST: List[int] = list(DS.k_list)
EVAL_FIRST_M: Optional[int] = 100

# Map our method_key → the reasoning-ON run_key (used to join trace_recall
# summaries against the chrF++ frame). RRF uses the bare run_key.
METHOD_KEY_TO_REASONING_ON_RUN_KEY: Dict[str, str] = {
    "rrf": "reasoning_on",
    "random": "reasoning_on_random",
    "sentinel": "reasoning_on_sentinel",
    "edit_dist": "reasoning_on_edit_dist",
}

# Output root for this script.
OUTPUT_ROOT = os.environ.get("RTRACE_CHRFPP_DIR", DS.analysis_dir("chrfpp_per_sentence_analysis"))

# Existing CSVs we read but never rewrite.
TOKEN_COUNTS_CSV = os.environ.get(
    "RTRACE_TOKENS_CSV",
    os.path.join(DS.analysis_dir("eval_token_counts"), "per_sentence_token_counts.csv"),
)
TRACE_RECALL_SUMMARY_PER_METHOD_DIR = DS.analysis_dir("trace_example_recall")
TRACE_RECALL_SUMMARY_GLOBAL_CSV = os.path.join(
    TRACE_RECALL_SUMMARY_PER_METHOD_DIR, "trace_recall_summary_all_methods.csv"
)
TRACE_RECALL_METHOD_KEYS: List[str] = ["rrf", "random", "sentinel", "edit_dist"]

# When True, skip Stage 1 / Stage 2 etc. if their CSV is already on disk.
# Same contract as trace_example_recall.py's LOAD_FROM_CSV.
LOAD_FROM_CSV: bool = True

# Sentence-level chrF++ scoring config — must match eval_pipeline.py's
# chrfpp = CHRF(word_order=2) exactly so the new numbers are methodologically
# parallel to the existing file-level chrF++ scores.
_CHRFPP = CHRF(word_order=2)

# ─── Column schemas ────────────────────────────────────────────────────────

PER_SENTENCE_CHRFPP_COLUMNS: List[str] = [
    "model_key", "model_display",
    "run_key", "run_display",
    "reasoning_state",
    "method_key", "method_label",
    "src_lang", "tgt_lang", "direction",
    "k", "sentence_idx",
    "chrfpp_sentence",
    "translation_path",
]

MASTER_COLUMNS: List[str] = [
    "model_key", "model_display",
    "run_key", "run_display",
    "reasoning_state",
    "method_key", "method_label",
    "src_lang", "tgt_lang", "direction",
    "k", "sentence_idx",
    "file_id", "source_id",
    "chrfpp_sentence",
    "reasoning_tokens", "response_tokens",
    "mean_recall", "mean_src_recall", "mean_tgt_recall", "max_recall",
    "trace_path",
]

CORRELATION_COLUMNS: List[str] = [
    "pair_name", "axis", "group",
    "n", "n_valid_traces",
    "pearson_r", "pearson_p",
    "spearman_r", "spearman_p",
]

ANOVA_COLUMNS: List[str] = [
    "family", "tgt_lang", "term", "sum_sq", "df", "F", "p_value", "partial_eta_sq",
]

# LMM (linear mixed-effects) output schema. Fit per (family, tgt_lang) on ON-only
# rows with a source-level random intercept that absorbs the cross-run dependence
# from the same English sentence being translated by every model/method/k cell.
# Each fixed-effect coefficient gets one row; categorical term levels share a
# `term` key (e.g. "C(method_label)") so they're easy to group/filter.
LMM_COLUMNS: List[str] = [
    "family", "tgt_lang", "term", "coef_name",
    "coef", "std_err", "z", "p_value", "ci_low", "ci_high",
    "n_obs", "n_source_groups",
    # Per-fit diagnostics — same value repeated across every coefficient
    # row in a given (family, tgt_lang) fit; carried in the CSV so a reader
    # can immediately tell when σ²_source collapsed to the boundary and the
    # LMM degenerated to OLS (ICC ≈ 0 is the diagnostic).
    "re_var", "resid_var", "icc",
]

# Joint per-term test schema: one row per (family, tgt_lang, term) holding the
# proper joint Wald χ² for that term (β̂' V̂⁻¹ β̂ across all that term's
# coefficients), the term's df, and the joint p-value. Replaces the
# back-of-envelope Σz²/df importance heuristic with statsmodels' actual
# linear-restriction test against the fixed-effect covariance matrix.
LMM_JOINT_COLUMNS: List[str] = [
    "family", "tgt_lang", "term", "df", "chi2", "p_value", "n_obs",
]

TABLE3_COLUMNS: List[str] = [
    "tgt_lang", "tgt_lang_display",
    "mean_reasoning_tokens_on", "rank_by_tokens",
    "chrfpp_on", "chrfpp_off", "delta_on_minus_off",
]

# Stage 6b — paired bootstrap. Schema mirrors paired_bootstrap_chrfpp_significance.py's
# SIG_LONG_COLUMNS / paired_bootstrap_lid_significance.py's SIG_LONG_COLUMNS verbatim
# so the resulting CSV is a drop-in replacement for downstream consumers.
PAIRED_BS_N: int = 1000
SACREBLEU_SEED: int = 12345

PAIRED_BS_LONG_COLUMNS: List[str] = [
    "model_key", "model_display",
    "method_key", "method_label",
    "src_lang", "tgt_lang", "direction",
    "k",
    "on_run_key", "on_run_display",
    "off_run_key", "off_run_display",
    "score_on", "score_off", "delta",
    "mean_on", "ci_on",
    "mean_off", "ci_off",
    "p_value",
    "significant_p_0_05", "significant_p_0_01", "sig_marker",
    "better_than_off",
    "n_segments",
]

# Stage 7 — eval_pipeline-style plotting. k=0 fixed out of plots
# (eval_pipeline.py's default; the user explicitly asked we "make that fixed").
GENERATE_PLOTS: bool = True
# k=0 (zero-shot) included by default. For edit_dist / sentinel / rrf the k=0
# chrF++ comes from the random run's k=0 file via the fallback wired in
# `_build_k0_fallback_map` — so all four method lines converge at k=0 (the
# prompt is identical when there are no in-context examples).
INCLUDE_K0_IN_PLOTS: bool = True
K_LIST_PLOT: List[int] = (
    sorted(set(K_LIST)) if INCLUDE_K0_IN_PLOTS
    else sorted(k for k in set(K_LIST) if k > 0)
)
MODEL_ORDER: List[str] = list(MODELS.keys())
MODEL_FAMILIES: Dict[str, List[str]] = {
    "Mistral": ["ministral_8b", "ministral_14b", "magistral_small"],
    "Qwen":    ["qwen3_8b",     "qwen3_14b",     "qwen3_32b"],
}
# Respect the RTRACE_EVAL_MODELS filter everywhere family lists are used
# (per-family ANOVA/LMM loops, cross-model plots): keep every family key so
# output-path dicts stay stable, but drop filtered-out models — an empty
# family then contributes zero rows / gets skipped instead of KeyError-ing.
MODEL_FAMILIES = {fam: [m for m in ms if m in MODELS] for fam, ms in MODEL_FAMILIES.items()}
MODEL_SIZE_THRESHOLD: float = 14.0
METRIC_NAME_FOR_PLOTS: str = "chrF++"  # the metric label printed on plots & tables

# Map method_key → (on_run_key, off_run_key, method_label) for Stage 6b/7 pairing.
METHOD_KEY_TO_RUN_KEYS: Dict[str, Tuple[str, str, str]] = {
    "rrf":       ("reasoning_on",            "reasoning_off",            "RRF"),
    "random":    ("reasoning_on_random",     "reasoning_off_random",     "Random"),
    "sentinel":  ("reasoning_on_sentinel",   "reasoning_off_sentinel",   "Sentinel"),
    "edit_dist": ("reasoning_on_edit_dist",  "reasoning_off_edit_dist",  "Edit Distance"),
}

# Output filenames.
PER_SENTENCE_CHRFPP_OUT = os.path.join(OUTPUT_ROOT, "per_sentence_chrfpp.csv")
MASTER_OUT = os.path.join(OUTPUT_ROOT, "regression_dataset_per_sentence.csv")
CORRELATIONS_OUT = os.path.join(OUTPUT_ROOT, "per_sentence_correlations.csv")
ANOVA_ON_ONLY_OUT = os.path.join(OUTPUT_ROOT, "anova_per_language_on_only.csv")
ANOVA_BY_FAMILY_OUTS: Dict[str, str] = {
    fam: os.path.join(OUTPUT_ROOT, f"anova_per_language_on_only_{fam.lower()}.csv")
    for fam in MODEL_FAMILIES
}
LMM_BY_FAMILY_OUTS: Dict[str, str] = {
    fam: os.path.join(OUTPUT_ROOT, f"lmm_per_language_on_only_{fam.lower()}.csv")
    for fam in MODEL_FAMILIES
}
LMM_JOINT_BY_FAMILY_OUTS: Dict[str, str] = {
    fam: os.path.join(OUTPUT_ROOT, f"lmm_joint_tests_{fam.lower()}.csv")
    for fam in MODEL_FAMILIES
}
TABLE3_OUT = os.path.join(OUTPUT_ROOT, "reasoning_token_vs_delta_per_lang.csv")
PAIRED_BS_OUT = os.path.join(OUTPUT_ROOT, "paired_bs_significance_long.csv")
PLOTS_ROOT = os.path.join(OUTPUT_ROOT, "plots")


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers (lifted from token_count_inference_budget.py / trace_example_recall.py)
# ─────────────────────────────────────────────────────────────────────────────


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def slugify(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    return re.sub(r"[^A-Za-z0-9_\-]+", "", s)


def _apply_limit(n: int, limit_m: Optional[int]) -> int:
    if limit_m is None:
        return n
    m = int(limit_m)
    return 0 if m <= 0 else min(n, m)


def _list_immediate_subdirs(parent: str) -> List[str]:
    try:
        return sorted(
            [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
        )
    except FileNotFoundError:
        return []


def resolve_model_dirname(base_dir: str, model_key: str) -> Optional[str]:
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


def direction_folder_name(src_lang: str, tgt_lang: str) -> str:
    return f"{src_lang.split('_')[0].lower()}_to_{tgt_lang.split('_')[0].lower()}"


def _resolve_root_dir(run: Dict[str, Any], model_key: str) -> Optional[str]:
    rd = run["root_dir"]
    if isinstance(rd, str):
        return rd
    if isinstance(rd, dict):
        return rd.get(model_key)
    return None


def build_translation_path(
    base_dir: str,
    model_dirname: str,
    direction: str,
    k: int,
    filename_pattern: str,
) -> str:
    return os.path.join(base_dir, model_dirname, direction, filename_pattern.format(K=k))


def _build_k0_fallback_map(
    generation_runs: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Non-random runs reuse the matching random run's k=0 file (k=0 is
    method-agnostic — zero in-context examples means an identical prompt)."""

    def _state_of(key_lower: str) -> str:
        return "on" if ("_on" in key_lower or key_lower.endswith("on")) else "off"

    random_by_state: Dict[str, Dict[str, Any]] = {}
    for r in generation_runs:
        kl = r["key"].lower()
        if "random" in kl:
            random_by_state[_state_of(kl)] = r

    fb: Dict[str, Dict[str, Any]] = {}
    for r in generation_runs:
        kl = r["key"].lower()
        if "random" in kl:
            continue
        state = _state_of(kl)
        if state in random_by_state:
            fb[r["key"]] = random_by_state[state]
    return fb


def read_jsonl_translations(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line)["translation"])
    return out


def _coerce_numeric_inplace(df: pd.DataFrame, columns: Sequence[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


# ─────────────────────────────────────────────────────────────────────────────
# Reference cache
# ─────────────────────────────────────────────────────────────────────────────

# References come from the dataset registry's standardized loader
# (load_sentences(lang) -> {"dev", "devtest"}); cached per language so the
# per-sentence loop doesn't re-load them for every file.
_REF_CACHE: Dict[str, List[str]] = {}


def get_flores_devtest_refs(tgt_lang: str) -> List[str]:
    """Cached devtest references for tgt_lang (name kept for call-site
    stability; dataset-agnostic via the registry loader)."""
    if tgt_lang in _REF_CACHE:
        return _REF_CACHE[tgt_lang]
    refs = list(DS.load_sentences(tgt_lang)["devtest"])
    _REF_CACHE[tgt_lang] = refs
    return refs


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Per-sentence chrF++ scoring
# ─────────────────────────────────────────────────────────────────────────────


def score_one_file(
    predictions: Sequence[str],
    references: Sequence[str],
) -> List[float]:
    """
    Purpose: Score chrF++ for each (prediction, reference) pair in a file.
    Inputs: Aligned prediction and reference lists.
    Outputs: List of chrF++ floats in [0, 100]; empty predictions score 0.
    """
    scores: List[float] = []
    for pred, ref in zip(predictions, references):
        if pred is None or pred == "":
            scores.append(0.0)
            continue
        scores.append(float(_CHRFPP.sentence_score(pred, [ref]).score))
    return scores


def compute_per_sentence_chrfpp(
    only_combos: Optional[set] = None,
) -> pd.DataFrame:
    """
    Purpose: Walk every translation JSONL, score per-sentence chrF++, return one row per (model, run, k, direction, sentence_idx).
    Inputs: Configuration variables; FLORES references via get_flores_devtest_refs.
            Optional `only_combos` is a set of (run_key, k) tuples to restrict
            scoring to a specific incremental slice (used by the cache layer
            to fill missing combinations like `(reasoning_on_edit_dist, 0)`).
    Outputs: Long-form DataFrame with PER_SENTENCE_CHRFPP_COLUMNS.
    """
    all_runs = GENERATION_RUNS_REASONING_ON + GENERATION_RUNS_REASONING_OFF
    k0_fallback = _build_k0_fallback_map(all_runs)

    # Resolve translation root → on-disk model dirname for each (run, model).
    resolved: Dict[Tuple[str, str], Optional[Tuple[str, str]]] = {}
    for run in all_runs:
        for model_key in MODELS:
            root = _resolve_root_dir(run, model_key)
            if root is None:
                resolved[(run["key"], model_key)] = None
                continue
            dirname = resolve_model_dirname(root, model_key)
            resolved[(run["key"], model_key)] = None if dirname is None else (root, dirname)

    print("\n[Stage 1] Translation root resolution:")
    for run in all_runs:
        print(f"  {run['key']}:")
        for model_key in MODELS:
            entry = resolved.get((run["key"], model_key))
            if entry is None:
                root = _resolve_root_dir(run, model_key)
                print(f"    • {model_key} → NOT FOUND (root: {root or 'NO ROOT'})")
            else:
                print(f"    • {model_key} → {entry[0]}/{entry[1]}")

    if k0_fallback:
        print("\n[Stage 1] k=0 fallback map:")
        for src_key, fb_run in k0_fallback.items():
            print(f"  • {src_key} → {fb_run['key']}  (only when k=0 file missing)")

    rows: List[Dict[str, Any]] = []
    n_files = 0
    n_files_with_score = 0
    n_files_empty = 0

    for src_lang in SRC_LANGS:
        for tgt_lang in TGT_LANGS:
            direction = direction_folder_name(src_lang, tgt_lang)
            references = get_flores_devtest_refs(tgt_lang)
            ref_ceiling = _apply_limit(len(references), EVAL_FIRST_M)
            references_capped = references[:ref_ceiling]

            for model_key, model_display in MODELS.items():
                for run in all_runs:
                    run_key = run["key"]
                    resolved_entry = resolved.get((run_key, model_key))

                    for k in K_LIST:
                        # Incremental-cache slice: skip combinations the cache
                        # already has when `only_combos` is set.
                        if only_combos is not None and (run_key, k) not in only_combos:
                            continue
                        n_files += 1

                        # Locate translation file (with k=0 fallback).
                        translation_path = ""
                        if resolved_entry is not None:
                            root_dir, model_dirname = resolved_entry
                            cand = build_translation_path(
                                root_dir, model_dirname, direction, k,
                                run["filename_pattern"],
                            )
                            if os.path.exists(cand):
                                translation_path = cand

                        if k == 0 and not translation_path and run_key in k0_fallback:
                            fb_run = k0_fallback[run_key]
                            fb_resolved = resolved.get((fb_run["key"], model_key))
                            if fb_resolved is not None:
                                fb_root, fb_dirname = fb_resolved
                                fb_cand = build_translation_path(
                                    fb_root, fb_dirname, direction, k,
                                    fb_run["filename_pattern"],
                                )
                                if os.path.exists(fb_cand):
                                    translation_path = fb_cand

                        if not translation_path:
                            continue

                        # Read predictions, cap to EVAL_FIRST_M.
                        try:
                            predictions = read_jsonl_translations(translation_path)
                        except Exception as e:
                            print(f"  ! read failed for {translation_path}: {e}")
                            continue
                        ceiling = _apply_limit(len(predictions), EVAL_FIRST_M)
                        predictions = predictions[:ceiling]

                        # Align with references (pad with "" if shorter).
                        n_sent = min(len(predictions), len(references_capped))
                        if n_sent == 0:
                            n_files_empty += 1
                            continue

                        scores = score_one_file(
                            predictions[:n_sent],
                            references_capped[:n_sent],
                        )
                        for i, score in enumerate(scores):
                            rows.append({
                                "model_key": model_key,
                                "model_display": model_display,
                                "run_key": run_key,
                                "run_display": run["label"],
                                "reasoning_state": run["reasoning_state"],
                                "method_key": run["method_key"],
                                "method_label": run["method_label"],
                                "src_lang": src_lang,
                                "tgt_lang": tgt_lang,
                                "direction": direction,
                                "k": k,
                                "sentence_idx": i,
                                "chrfpp_sentence": float(score),
                                "translation_path": translation_path,
                            })
                        n_files_with_score += 1

                # Progress per (model, direction).
                print(
                    f"  [Stage 1] scored {n_files_with_score}/{n_files} files so far "
                    f"({n_files_empty} empty)  current={model_display} | {direction}"
                )

    df = pd.DataFrame(rows, columns=PER_SENTENCE_CHRFPP_COLUMNS)
    print(f"\n[Stage 1] Total per-sentence rows: {len(df):,}  "
          f"(from {n_files_with_score:,} files)")
    return df


def _expected_run_k_combos() -> set:
    """The (run_key, k) grid we expect to find in the Stage-1 cache, given
    the current GENERATION_RUNS_* + K_LIST config."""
    all_runs = GENERATION_RUNS_REASONING_ON + GENERATION_RUNS_REASONING_OFF
    return {(r["key"], int(k)) for r in all_runs for k in K_LIST}


def load_or_compute_per_sentence_chrfpp() -> pd.DataFrame:
    """
    Purpose: Return Stage-1 DataFrame; incrementally fill missing (run_key, k) combinations when the cache is partial.
    Inputs: LOAD_FROM_CSV, PER_SENTENCE_CHRFPP_OUT, GENERATION_RUNS_*, K_LIST.
    Outputs: Stage-1 long-form DataFrame.

    Cache contract — same idea as eval_pipeline.py's incremental compute, but
    at (run_key, k) granularity instead of run_key only. This handles the
    case where a cache was written before some (run_key, k) cell had data —
    most commonly `(reasoning_on_edit_dist, 0)` and
    `(reasoning_off_edit_dist, 0)`, which only materialise via the k=0
    random fallback in `_build_k0_fallback_map`. On reload we detect missing
    combinations, score only those, append, and persist the union.
    """
    expected = _expected_run_k_combos()

    if LOAD_FROM_CSV and os.path.exists(PER_SENTENCE_CHRFPP_OUT):
        print(f"\n[LOAD_FROM_CSV] Stage 1 cache hit: {PER_SENTENCE_CHRFPP_OUT}")
        df_cached = pd.read_csv(PER_SENTENCE_CHRFPP_OUT)
        if "k" in df_cached.columns:
            df_cached["k"] = df_cached["k"].astype(int)
        _coerce_numeric_inplace(df_cached, ["chrfpp_sentence", "sentence_idx"])
        print(f"  Loaded {len(df_cached):,} rows.")

        cached_combos = set(zip(df_cached["run_key"], df_cached["k"]))
        missing = expected - cached_combos
        # Tag combinations whose run_key is in the current config but isn't
        # in the cache for any k — same shape as eval_pipeline's "n run(s)
        # missing from cache" log.
        cached_run_keys = {rk for rk, _ in cached_combos}
        expected_run_keys = {rk for rk, _ in expected}
        unexpected_run_keys = cached_run_keys - expected_run_keys
        if unexpected_run_keys:
            print(f"  [note] cache contains {len(unexpected_run_keys)} run(s) "
                  f"not in current config; they pass through unchanged: "
                  f"{sorted(unexpected_run_keys)}")

        if missing:
            print(f"\n[Incremental compute] {len(missing)} (run_key, k) "
                  f"combination(s) missing from cache:")
            for rk, k in sorted(missing):
                print(f"  • run_key={rk:<32s} k={k}")
            print("\n  Computing missing combinations only "
                  "(k=0 fallback will fill edit_dist via the random run).")
            df_new = compute_per_sentence_chrfpp(only_combos=missing)
            df = pd.concat([df_cached, df_new], ignore_index=True)
            ensure_dir(OUTPUT_ROOT)
            df.to_csv(PER_SENTENCE_CHRFPP_OUT, index=False)
            print(f"[Stage 1] Cache updated: {len(df_cached):,} cached + "
                  f"{len(df_new):,} new = {len(df):,} rows.")

            # Stage 1 changed → every downstream cache built from the master
            # frame is stale. Delete them so Stages 2–6b regenerate against
            # the now-complete per-sentence data.
            for stale in [
                MASTER_OUT,
                CORRELATIONS_OUT,
                ANOVA_ON_ONLY_OUT,
                *ANOVA_BY_FAMILY_OUTS.values(),
                *LMM_BY_FAMILY_OUTS.values(),
                *LMM_JOINT_BY_FAMILY_OUTS.values(),
                TABLE3_OUT,
                PAIRED_BS_OUT,
            ]:
                if os.path.exists(stale):
                    try:
                        os.remove(stale)
                        print(f"  [Stage 1 invalidation] removed stale "
                              f"downstream cache: {stale}")
                    except OSError as e:
                        print(f"  ! could not remove {stale}: {e}")
            return df

        print("  All expected (run_key, k) combinations are present; "
              "skipping the Stage-1 compute step.")
        return df_cached

    print("\n[Stage 1] Computing per-sentence chrF++ from scratch...")
    df = compute_per_sentence_chrfpp()
    ensure_dir(OUTPUT_ROOT)
    df.to_csv(PER_SENTENCE_CHRFPP_OUT, index=False)
    print(f"[Stage 1] Wrote {PER_SENTENCE_CHRFPP_OUT}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Master per-sentence join
# ─────────────────────────────────────────────────────────────────────────────


def _load_trace_recall_summaries() -> pd.DataFrame:
    """Load and concatenate per-method trace_recall_summary.csv files. Prefers
    the global aggregate; falls back to per-method copies."""
    if os.path.exists(TRACE_RECALL_SUMMARY_GLOBAL_CSV):
        print(f"  Reading trace recall summaries from global aggregate: "
              f"{TRACE_RECALL_SUMMARY_GLOBAL_CSV}")
        return pd.read_csv(TRACE_RECALL_SUMMARY_GLOBAL_CSV)

    print("  Global trace recall aggregate not found — loading per-method copies.")
    parts: List[pd.DataFrame] = []
    for method_key in TRACE_RECALL_METHOD_KEYS:
        path = os.path.join(
            TRACE_RECALL_SUMMARY_PER_METHOD_DIR,
            slugify(method_key),
            "trace_recall_summary.csv",
        )
        if os.path.exists(path):
            parts.append(pd.read_csv(path))
        else:
            print(f"  ! missing trace recall summary: {path}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_master_frame(df_chrfpp: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose: Join per-sentence chrF++ with per-sentence tokens and per-sentence trace recall.
    Inputs: Stage-1 DataFrame; reads TOKEN_COUNTS_CSV and trace_recall_summary CSVs from disk.
    Outputs: Master per-sentence design matrix with MASTER_COLUMNS.
    """
    print("\n[Stage 2] Building master per-sentence frame.")

    # Tokens.
    if not os.path.exists(TOKEN_COUNTS_CSV):
        raise FileNotFoundError(
            f"Token-count CSV missing — needed for the master join: {TOKEN_COUNTS_CSV}"
        )
    df_tok = pd.read_csv(TOKEN_COUNTS_CSV)
    if "k" in df_tok.columns:
        df_tok["k"] = df_tok["k"].astype(int)
    _coerce_numeric_inplace(df_tok, ["reasoning_tokens", "response_tokens", "sentence_idx"])
    df_tok = df_tok[[
        "model_key", "run_key", "src_lang", "tgt_lang", "direction", "k",
        "sentence_idx", "reasoning_tokens", "response_tokens",
    ]]
    print(f"  Loaded {len(df_tok):,} per-sentence token rows from "
          f"{TOKEN_COUNTS_CSV}")

    # Recall — only meaningful for reasoning-ON; OFF rows will join to NaN.
    df_rec = _load_trace_recall_summaries()
    if df_rec.empty:
        print("  ! No trace_recall_summary rows found; recall columns will be NaN.")
        df_chrfpp = df_chrfpp.copy()
        for col in ["mean_recall", "mean_src_recall", "mean_tgt_recall",
                    "max_recall", "trace_path"]:
            df_chrfpp[col] = pd.NA
    else:
        if "k" in df_rec.columns:
            df_rec["k"] = df_rec["k"].astype(int)
        _coerce_numeric_inplace(df_rec, [
            "mean_recall", "mean_src_recall", "mean_tgt_recall", "max_recall",
            "devtest_index",
        ])
        # Build the reasoning-ON run_key for each recall row so the join key
        # matches the chrF++ frame's run_key on ON rows.
        if "method_key" in df_rec.columns:
            df_rec["recall_run_key"] = df_rec["method_key"].map(
                METHOD_KEY_TO_REASONING_ON_RUN_KEY
            )
        else:
            raise ValueError(
                "trace_recall_summary CSV missing 'method_key' column — "
                "can't map to reasoning-ON run_key."
            )
        df_rec = df_rec.rename(columns={"devtest_index": "sentence_idx"})
        df_rec_for_join = df_rec[[
            "model_key", "recall_run_key", "src_lang", "tgt_lang",
            "direction", "k", "sentence_idx",
            "mean_recall", "mean_src_recall", "mean_tgt_recall", "max_recall",
            "trace_path",
        ]].rename(columns={"recall_run_key": "run_key"})
        print(f"  Loaded {len(df_rec_for_join):,} per-sentence recall rows.")

    # Left-join chrfpp × tokens — keep every chrfpp row even when the token
    # CSV is missing that (run_key, k) cell. Most common case: the token CSV
    # was generated before its own k=0 fallback was applied to edit_dist /
    # sentinel, so it has no `(reasoning_*_edit_dist, k=0)` rows. With an
    # inner join, those chrfpp rows would be silently dropped — and then
    # edit_dist k=0 would never appear in the master frame, the plots, or
    # the bootstrap CSV (which is the symptom you saw).
    n_chrfpp_before = len(df_chrfpp)
    merged = df_chrfpp.merge(
        df_tok,
        on=["model_key", "run_key", "src_lang", "tgt_lang", "direction", "k", "sentence_idx"],
        how="left",
    )
    n_no_tokens = int(merged["reasoning_tokens"].isna().sum()
                      + merged["response_tokens"].isna().sum())
    n_rows_no_tokens = int(
        merged["reasoning_tokens"].isna().sum()
    )
    if n_rows_no_tokens > 0:
        # Surface a brief audit by (run_key, k) so it's obvious which slices
        # weren't covered by the token CSV. Token-dependent stages (ANOVA,
        # correlations) will dropna those rows; plots will keep them.
        missing_by_combo = (
            merged.loc[merged["reasoning_tokens"].isna()]
            .groupby(["run_key", "k"]).size().reset_index(name="n")
            .sort_values(["run_key", "k"])
        )
        print(f"  chrfpp × tokens (LEFT join): "
              f"{len(merged):,} sentence rows kept; "
              f"{n_rows_no_tokens:,} have no token row in "
              f"per_sentence_token_counts.csv.")
        print(f"  Missing-token breakdown by (run_key, k):")
        for _, mc in missing_by_combo.iterrows():
            print(f"    • {mc['run_key']:<32s} k={int(mc['k'])}  n={int(mc['n'])}")
    else:
        print(f"  chrfpp × tokens (LEFT join): {len(merged):,} sentence rows "
              "(every chrfpp row matched a token row).")

    # Then left-join recall on top (OFF rows keep NaN by construction).
    if not df_rec.empty:
        merged = merged.merge(
            df_rec_for_join,
            on=["model_key", "run_key", "src_lang", "tgt_lang", "direction", "k", "sentence_idx"],
            how="left",
        )

    # Derive cluster ids and ensure schema.
    merged["file_id"] = (
        merged["model_key"].astype(str) + "|"
        + merged["run_key"].astype(str) + "|"
        + merged["direction"].astype(str) + "|k"
        + merged["k"].astype(str)
    )
    merged["source_id"] = (
        merged["direction"].astype(str) + "|s"
        + merged["sentence_idx"].astype(str)
    )

    out = merged.reindex(columns=MASTER_COLUMNS)
    print(f"  Master frame: {len(out):,} rows × {out.shape[1]} columns.")
    print(f"  Reasoning-ON rows: {(out['reasoning_state'] == 'On').sum():,}")
    print(f"  Reasoning-OFF rows: {(out['reasoning_state'] == 'Off').sum():,}")
    print(f"  Rows with non-NaN mean_recall: {out['mean_recall'].notna().sum():,}")
    return out


def load_or_build_master(df_chrfpp: pd.DataFrame) -> pd.DataFrame:
    """Cache-guarded master frame builder with auto-invalidation.

    If the loaded master is missing any (run_key, k) combination that Stage 1
    has, treat the master as stale and rebuild. This is the auto-fix path for
    the common case where the master CSV was written with the old chrfpp ×
    tokens inner join (which silently dropped edit_dist k=0 because the token
    CSV didn't have those rows yet).
    """
    if LOAD_FROM_CSV and os.path.exists(MASTER_OUT):
        print(f"\n[LOAD_FROM_CSV] Stage 2 cache hit: {MASTER_OUT}")
        df = pd.read_csv(MASTER_OUT)
        if "k" in df.columns:
            df["k"] = df["k"].astype(int)
        _coerce_numeric_inplace(df, [
            "chrfpp_sentence", "reasoning_tokens", "response_tokens",
            "mean_recall", "mean_src_recall", "mean_tgt_recall", "max_recall",
            "sentence_idx",
        ])

        chrfpp_combos = set(zip(df_chrfpp["run_key"], df_chrfpp["k"]))
        master_combos = set(zip(df["run_key"], df["k"]))
        missing = chrfpp_combos - master_combos
        if not missing:
            print(f"  Loaded {len(df):,} rows.")
            return df

        print(f"  [Stage 2 invalidation] master cache is missing "
              f"{len(missing)} (run_key, k) combination(s) that Stage 1 has:")
        for rk, k in sorted(missing):
            print(f"    • run_key={rk:<32s} k={k}")
        print("  Rebuilding the master frame from the current Stage-1 cache.")
        try:
            os.remove(MASTER_OUT)
        except OSError as e:
            print(f"  ! could not remove stale master: {e}")
        # Fall through to the rebuild path.

    df = build_master_frame(df_chrfpp)
    ensure_dir(OUTPUT_ROOT)
    df.to_csv(MASTER_OUT, index=False)
    print(f"[Stage 2] Wrote {MASTER_OUT}")

    # Master frame rebuilt → every downstream cache that derives from it is
    # stale. Symmetric to the invalidation in Stage 1; lets the user pick
    # up master-shape changes (e.g. inner→left join restoring edit_dist k=0)
    # by deleting just MASTER_OUT.
    for stale in [
        CORRELATIONS_OUT,
        ANOVA_ON_ONLY_OUT,
        *ANOVA_BY_FAMILY_OUTS.values(),
        *LMM_BY_FAMILY_OUTS.values(),
        *LMM_JOINT_BY_FAMILY_OUTS.values(),
        TABLE3_OUT,
        PAIRED_BS_OUT,
    ]:
        if os.path.exists(stale):
            try:
                os.remove(stale)
                print(f"  [Stage 2 invalidation] removed stale downstream "
                      f"cache: {stale}")
            except OSError as e:
                print(f"  ! could not remove {stale}: {e}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Correlations (Pearson + Spearman, 9 pairs × 8 axes)
# ─────────────────────────────────────────────────────────────────────────────


# Lifted from trace_example_recall.py.
def _pearson_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """NaN-safe Pearson + Spearman with two-sided p-values."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    xv = x[mask]
    yv = y[mask]
    if np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return float("nan"), float("nan"), float("nan"), float("nan")
    pr = pearsonr(xv, yv)
    sr = spearmanr(xv, yv)
    return float(pr.statistic), float(pr.pvalue), float(sr.statistic), float(sr.pvalue)


# Pairs to score. `on_only=True` restricts to reasoning-ON rows (recall and
# reasoning_tokens are NaN for OFF).
CORRELATION_PAIRS: List[Dict[str, Any]] = [
    {"pair_name": "chrfpp × mean_recall",        "x": "chrfpp_sentence",  "y": "mean_recall",       "on_only": True},
    {"pair_name": "chrfpp × mean_src_recall",    "x": "chrfpp_sentence",  "y": "mean_src_recall",   "on_only": True},
    {"pair_name": "chrfpp × mean_tgt_recall",    "x": "chrfpp_sentence",  "y": "mean_tgt_recall",   "on_only": True},
    {"pair_name": "chrfpp × max_recall",         "x": "chrfpp_sentence",  "y": "max_recall",        "on_only": True},
    {"pair_name": "chrfpp × reasoning_tokens",   "x": "chrfpp_sentence",  "y": "reasoning_tokens",  "on_only": True},
    {"pair_name": "chrfpp × response_tokens",    "x": "chrfpp_sentence",  "y": "response_tokens",   "on_only": False},
    {"pair_name": "mean_recall × reasoning_tokens", "x": "mean_recall",   "y": "reasoning_tokens",  "on_only": True},
    {"pair_name": "mean_recall × response_tokens",  "x": "mean_recall",   "y": "response_tokens",   "on_only": True},
    {"pair_name": "reasoning_tokens × response_tokens", "x": "reasoning_tokens", "y": "response_tokens", "on_only": True},
]


def _stat_row(
    pair_name: str,
    axis: str,
    group: Any,
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> Dict[str, Any]:
    x = sub[x_col].to_numpy(dtype=np.float64)
    y = sub[y_col].to_numpy(dtype=np.float64)
    pr, pp, sr, sp = _pearson_spearman(x, y)
    return {
        "pair_name": pair_name,
        "axis": axis,
        "group": str(group),
        "n": int(len(sub)),
        "n_valid_traces": int((~np.isnan(x) & ~np.isnan(y)).sum()),
        "pearson_r": pr,
        "pearson_p": pp,
        "spearman_r": sr,
        "spearman_p": sp,
    }


def compute_correlations(df_master: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose: Pearson + Spearman correlations for every pair × every breakdown axis.
    Inputs: Master frame.
    Outputs: Long-form correlation DataFrame with CORRELATION_COLUMNS.
    """
    print("\n[Stage 3] Computing per-sentence correlations across "
          f"{len(CORRELATION_PAIRS)} pairs.")
    rows: List[Dict[str, Any]] = []

    for pair in CORRELATION_PAIRS:
        pair_name = pair["pair_name"]
        x_col = pair["x"]
        y_col = pair["y"]
        df_pair = df_master
        if pair["on_only"]:
            df_pair = df_master.loc[df_master["reasoning_state"] == "On"]

        if df_pair.empty:
            continue

        rows.append(_stat_row(pair_name, "overall", "all", df_pair, x_col, y_col))
        for axis_name, axis_col in [
            ("model", "model_key"),
            ("method", "method_label"),
            ("direction", "direction"),
            ("k", "k"),
        ]:
            for grp, sub in df_pair.groupby(axis_col, sort=True):
                rows.append(_stat_row(pair_name, axis_name, grp, sub, x_col, y_col))

        # Per-method-stratified breakdowns.
        for method_key, method_sub in df_pair.groupby("method_label", sort=True):
            for axis_name, axis_col in [
                ("method|model", "model_key"),
                ("method|direction", "direction"),
                ("method|k", "k"),
            ]:
                for grp, sub in method_sub.groupby(axis_col, sort=True):
                    rows.append(_stat_row(
                        pair_name, axis_name,
                        f"{method_key}|{grp}", sub, x_col, y_col,
                    ))

    df_corr = pd.DataFrame(rows, columns=CORRELATION_COLUMNS)
    print(f"[Stage 3] Computed {len(df_corr):,} correlation rows.")
    return df_corr


def load_or_compute_correlations(df_master: pd.DataFrame) -> pd.DataFrame:
    if LOAD_FROM_CSV and os.path.exists(CORRELATIONS_OUT):
        print(f"\n[LOAD_FROM_CSV] Stage 3 cache hit: {CORRELATIONS_OUT}")
        return pd.read_csv(CORRELATIONS_OUT)

    df = compute_correlations(df_master)
    ensure_dir(OUTPUT_ROOT)
    df.to_csv(CORRELATIONS_OUT, index=False)
    print(f"[Stage 3] Wrote {CORRELATIONS_OUT}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Regression ANOVA (Type-III full ON+OFF, Type-II ON-only)
# ─────────────────────────────────────────────────────────────────────────────


def _two_way_cluster_groups(df: pd.DataFrame) -> np.ndarray:
    """Encode (file_id, source_id) into a (n, 2) int matrix for cov_type='cluster'."""
    return np.column_stack([
        pd.Categorical(df["file_id"]).codes,
        pd.Categorical(df["source_id"]).codes,
    ])


_F_COL_CANDIDATES: Tuple[str, ...] = ("F", "statistic", "fvalue", "Fvalue", "chi2")
_P_COL_CANDIDATES: Tuple[str, ...] = ("P>F", "PR(>F)", "pvalue", "p_value", "P>chi2", "pvalues")


def _safe_float(x: Any) -> float:
    """
    Purpose: Convert any scalar-like input to a Python float; NaN on failure.
    Inputs: A Python scalar, numpy scalar, 0-d / 1-element numpy array, or single-element pandas Series.
    Outputs: A Python float; NaN whenever the value is missing or non-numeric.

    NumPy ≥1.25 deprecated `float(arr)` for arrays with ndim > 0 (emits
    DeprecationWarning, errors in a future release). Statsmodels'
    `wald_test_terms().table[col]` and some pandas `.get()` paths return
    1-element ndarrays instead of plain scalars, so we route through
    `np.asarray(...).reshape(-1)[0].item()` to extract a true Python scalar
    before the float cast.
    """
    if x is None:
        return float("nan")
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return float("nan")
        scalar = arr.reshape(-1)[0]
        # Numpy scalars expose `.item()`; Python scalars are returned by
        # asarray.flat[0] as numpy scalars too, so this branch always works.
        if hasattr(scalar, "item"):
            scalar = scalar.item()
        v = float(scalar)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _lookup_first(row: pd.Series, columns: pd.Index, candidates: Tuple[str, ...]) -> float:
    """First candidate column name that exists and yields a finite float."""
    for col in candidates:
        if col in columns:
            v = _safe_float(row.get(col, float("nan")))
            if not np.isnan(v):
                return v
    return float("nan")


def _fit_anova_with_cluster_se(
    formula: str,
    df: pd.DataFrame,
    typ: int,
) -> pd.DataFrame:
    """
    Purpose: Fit OLS with two-way cluster-robust SEs, return a combined ANOVA table.
    Inputs: A patsy formula, a DataFrame, and the SS type (2 or 3).
    Outputs: ANOVA DataFrame with ANOVA_COLUMNS (without tgt_lang — the caller
             adds it). SS / df / partial_eta_sq from classical anova_lm;
             F / p_value from wald_test_terms when available, falling back to
             classical F/p whenever the robust path returns NaN. Column-name
             variants used by `wald_test_terms().table` across statsmodels
             releases are all checked (F/statistic/fvalue/chi2 for the stat,
             P>F/pvalue/P>chi2 for the p-value).
    """
    # Vanilla OLS for classical SS / df / partial η².
    classical = smf.ols(formula=formula, data=df).fit()
    classical_anova = anova_lm(classical, typ=typ)
    ss_resid = float(classical_anova.loc["Residual", "sum_sq"])

    # Cluster-robust fit (same design matrix) for honest F-tests via Wald.
    cluster_groups = _two_way_cluster_groups(df)
    robust = smf.ols(formula=formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster_groups, "use_correction": True},
    )
    robust_wald: Optional[pd.DataFrame] = None
    try:
        wt = robust.wald_test_terms(scalar=False)
        robust_wald = wt.table.copy()
    except Exception as e:
        print(f"  ! wald_test_terms failed ({e}); falling back to classical F/p.")

    rows: List[Dict[str, Any]] = []
    for term in classical_anova.index:
        if term == "Residual":
            continue
        ss_term = float(classical_anova.loc[term, "sum_sq"])
        df_term = float(classical_anova.loc[term, "df"])
        partial_eta_sq = (
            ss_term / (ss_term + ss_resid)
            if (ss_term + ss_resid) > 0 else float("nan")
        )

        # Start with the classical F/p (always populated for non-Intercept
        # terms in anova_lm); overlay the cluster-robust value when the
        # robust path produced a finite number for this term.
        f_stat = _safe_float(classical_anova.loc[term].get("F", float("nan")))
        p_val = _safe_float(classical_anova.loc[term].get("PR(>F)", float("nan")))

        if robust_wald is not None and term in robust_wald.index:
            row = robust_wald.loc[term]
            f_robust = _lookup_first(row, robust_wald.columns, _F_COL_CANDIDATES)
            p_robust = _lookup_first(row, robust_wald.columns, _P_COL_CANDIDATES)
            if not np.isnan(f_robust):
                f_stat = f_robust
            if not np.isnan(p_robust):
                p_val = p_robust

        rows.append({
            # _classify_term collapses Patsy's contrast spec
            # ("C(x, Treatment(reference='Y'))" → "C(x)") so the term column
            # is stable regardless of which reference level the formula picks.
            "term": _classify_term(term),
            "sum_sq": ss_term,
            "df": df_term,
            "F": f_stat,
            "p_value": p_val,
            "partial_eta_sq": partial_eta_sq,
        })

    # Caller is responsible for prepending the family and tgt_lang columns.
    return pd.DataFrame(rows, columns=[c for c in ANOVA_COLUMNS if c not in {"family", "tgt_lang"}])


_PER_LANG_FORMULA: str = (
    "chrfpp_sentence ~ "
    "C(method_label, Treatment(reference='Random')) + log1p_k "
    "+ reasoning_tokens + mean_recall"
)


def fit_anova_per_language_on_only(
    df_master: pd.DataFrame,
    family: str,
    family_models: List[str],
) -> pd.DataFrame:
    """
    Purpose: Fit a Type-II ANOVA per tgt_lang on ON-only data restricted to
             one model family, with only the controllable / focal technique
             factors (no tgt_lang, no model_key, no response_tokens — the
             last is a model output, not a factor we choose, same as
             reasoning_tokens but reasoning_tokens is the focal predictor
             we want to read). Splitting by family avoids pooling Mistral
             and Qwen behaviour into a single average that hides family-
             specific reasoning effects.
    Inputs: Master per-sentence DataFrame, family label ("Mistral" / "Qwen"),
            list of model_key strings belonging to that family.
    Outputs: Long-form ANOVA DataFrame: one row per (family, tgt_lang, term)
             with sum_sq / df / F / p_value / partial_eta_sq populated.

    Within-language ANOVA answers "which technique factors drive sentence
    chrF++ in this language, holding the rest constant?" The headline use
    is `reasoning_tokens` and `mean_recall` per language. Use the matching
    LMM result alongside this — ANOVA's F-tests treat each ON-only row as
    independent (which is false: the same English sentence appears in every
    model × method × k cell), and the cluster-robust SEs only patch the
    SEs, not the F itself. The LMM puts a random intercept on `source_id`
    and is the correct primary read; the ANOVA is the conventional
    SS-decomposition view for readers who expect it.

    Scale discipline — the `sum_sq` column is *scale-dependent*: a continuous
    predictor's SS scales with its variance × its regression coefficient,
    so SS for `reasoning_tokens` (range ~0–30 k) and SS for `mean_recall`
    (range 0–100) are NOT directly comparable, nor are they comparable to
    SS for the categorical terms. The proper cross-factor comparison
    metric is `partial_eta_sq` (scale-invariant). F is also scale-invariant
    per term and tells you whether the effect is detectable. Use SS only as
    bookkeeping; lead any paper claim with η²p.

    Two-way cluster-robust SEs on (file_id, source_id) are computed inside
    `_fit_anova_with_cluster_se`; within a single language `source_id`
    reduces to `sentence_idx`.
    """
    print(f"\n[Stage 4 / {family}] Per-language Type-II ANOVA on ON-only data "
          "(no tgt_lang / model_key / response_tokens; cluster-robust SEs).")
    print(f"  family models: {family_models}")
    print(f"  formula: {_PER_LANG_FORMULA}")

    # Pre-dropna row count per language, plus per-column NaN audit so any
    # n < expected ceiling has a precise source (most commonly: mean_recall
    # NaN on sentences where extract_reasoning_text returned None and the
    # trace_recall_summary CSV therefore has no row for that sentence).
    dropna_cols = ["chrfpp_sentence", "reasoning_tokens", "mean_recall",
                   "tgt_lang", "method_label", "k"]
    df_on_pre = df_master.loc[
        (df_master["reasoning_state"] == "On")
        & (df_master["model_key"].isin(family_models))
    ].copy()
    df_on = df_on_pre.dropna(subset=dropna_cols).copy()
    df_on["k"] = df_on["k"].astype(int)
    # Continuous, log-transformed k. log1p(k) = log(k+1) gives a single slope
    # coefficient on a scale where +1 means "roughly double the example count"
    # — matches the diminishing-returns shape of in-context examples better
    # than treating k as 5 separate dummies.
    df_on["log1p_k"] = np.log1p(df_on["k"])

    parts: List[pd.DataFrame] = []
    for tgt_lang in sorted(df_on["tgt_lang"].unique()):
        sub_pre = df_on_pre.loc[df_on_pre["tgt_lang"] == tgt_lang]
        sub = df_on.loc[df_on["tgt_lang"] == tgt_lang]
        if len(sub) < 30:
            print(f"  ! skip {tgt_lang} — only {len(sub)} usable rows.")
            continue
        try:
            anova_df = _fit_anova_with_cluster_se(_PER_LANG_FORMULA, sub, typ=2)
        except Exception as e:
            print(f"  ! ANOVA fit failed for {tgt_lang}: {e}")
            continue
        anova_df.insert(0, "tgt_lang", tgt_lang)
        anova_df.insert(0, "family", family)
        parts.append(anova_df)

        # Per-language audit. The "kept" count is the n the ANOVA actually saw;
        # the per-column NaN counts (only for non-zero columns) tell you which
        # predictor was missing for the dropped rows.
        n_pre = len(sub_pre)
        n_post = len(sub)
        if n_post < n_pre:
            null_breakdown = {
                col: int(sub_pre[col].isna().sum())
                for col in dropna_cols if sub_pre[col].isna().any()
            }
            null_str = ", ".join(f"{c}={n}" for c, n in null_breakdown.items())
            print(f"  {tgt_lang}: n={len(sub):,}  terms={len(anova_df)}  "
                  f"(dropped {n_pre - n_post:,} ON rows; NaN sources → {null_str})")
        else:
            print(f"  {tgt_lang}: n={len(sub):,}  terms={len(anova_df)}")

    if not parts:
        return pd.DataFrame(columns=ANOVA_COLUMNS)
    return pd.concat(parts, ignore_index=True)[ANOVA_COLUMNS]


def load_or_fit_anova(df_master: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Purpose: Cache-guarded per-language ON-only ANOVA fit, run separately per
             model family. Each family writes to its own CSV so the cache can
             be invalidated independently.
    Inputs: Master per-sentence DataFrame.
    Outputs: Dict mapping family name → long-form ANOVA DataFrame.
    """
    out: Dict[str, pd.DataFrame] = {}
    for family, family_models in MODEL_FAMILIES.items():
        path = ANOVA_BY_FAMILY_OUTS[family]
        if LOAD_FROM_CSV and os.path.exists(path):
            print(f"\n[LOAD_FROM_CSV] Stage 4 / {family} cache hit: {path}")
            df = pd.read_csv(path)
            _coerce_numeric_inplace(df, ["sum_sq", "df", "F", "p_value", "partial_eta_sq"])
            print(f"  Loaded {len(df):,} rows.")
            out[family] = df
            continue

        df = fit_anova_per_language_on_only(df_master, family, family_models)
        ensure_dir(OUTPUT_ROOT)
        df.to_csv(path, index=False)
        print(f"[Stage 4 / {family}] Wrote {path}")
        out[family] = df
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4b — Linear Mixed-Effects Model (LMM) per language, per family
# ─────────────────────────────────────────────────────────────────────────────
#
# Why an LMM alongside the ANOVA:
#   The ANOVA above treats every per-sentence ON row as an independent draw,
#   then tries to repair the SEs after the fact via two-way clustering on
#   (file_id, source_id). That cleans up SEs but the F-statistic itself is
#   still built off independence-assumption residuals. The LMM specifies the
#   non-independence directly — every row sharing the same source English
#   sentence is correlated through a random intercept on `source_id` — so
#   the fixed-effect coefficient inference is the right one even before any
#   robust-SE adjustment.
#
# The fixed-effect part of the formula matches the ANOVA exactly so the two
# views are directly comparable. The random part is just `(1 | source_id)`
# (single grouping variable; statsmodels' formula-API mixedlm doesn't model
# crossed random effects natively, and file_id variance is largely absorbed
# by the C(method_label) + C(k) fixed effects within a single family×lang).

_PER_LANG_LMM_FIXED: str = (
    "chrfpp_sentence ~ "
    "C(method_label, Treatment(reference='Random')) + log1p_k "
    "+ reasoning_tokens + mean_recall"
)


def _classify_term(coef_name: str) -> str:
    """Group a statsmodels coefficient name back to its source term.

    Patsy expands `C(method_label)` into coefficient names like
    `C(method_label)[T.RRF]`; this helper collapses every level coefficient
    back under the same `term` key so a downstream `groupby("term")` matches
    the ANOVA's term axis.

    Also strips Patsy's explicit contrast spec — `C(x, Treatment(reference='Y'))`
    expands to coefficient names like `C(x, Treatment(reference='Y'))[T.Z]`,
    which we collapse all the way down to `C(x)` so the output CSV's term
    column doesn't depend on which reference level the formula picked.
    """
    base = coef_name.split("[", 1)[0]
    if base.startswith("C(") and "," in base:
        # e.g. "C(method_label, Treatment(reference='Random'))" → "C(method_label)"
        var_name = base[2:].split(",", 1)[0].strip()
        return f"C({var_name})"
    return base


def _joint_wald_per_term(result) -> pd.DataFrame:
    """Compute the proper joint Wald χ² for each fixed-effect term.

    For a term whose coefficients have indices i₁…i_d in the fixed-effect
    parameter vector β̂ with covariance V̂, the joint Wald statistic
    W = (Rβ̂)' (RV̂R')⁻¹ (Rβ̂) is distributed χ²_d under H₀: Rβ = 0, where R
    is a d × p matrix selecting just those coefficients. Statsmodels'
    `result.wald_test(R)` computes exactly this and accounts for the
    off-diagonals in V̂ — it's the inference-correct generalisation of the
    Σ z² heuristic, which would only equal W if the within-term coefficient
    estimates were exactly uncorrelated.

    Returns a DataFrame with columns: term, df, chi2, p_value. Intercept
    is skipped. MixedLM's full parameter vector is fixed effects followed
    by variance components; we zero-pad R across the variance-component
    columns so the restriction only touches the fixed effects.
    """
    fe_params = result.fe_params
    coef_names = list(fe_params.index)
    n_full = len(result.params)  # fe + variance components

    term_to_idx: Dict[str, List[int]] = {}
    for i, cn in enumerate(coef_names):
        if cn == "Intercept":
            continue
        term = _classify_term(cn)
        term_to_idx.setdefault(term, []).append(i)

    rows: List[Dict[str, Any]] = []
    for term, idxs in term_to_idx.items():
        df_val = len(idxs)
        R = np.zeros((df_val, n_full))
        for j, i in enumerate(idxs):
            # Fixed-effect coefs sit at the front of the param vector.
            R[j, i] = 1.0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                test = result.wald_test(R, use_f=False, scalar=False)
            chi2 = _safe_float(test.statistic)
            p_val = _safe_float(test.pvalue)
        except Exception as e:
            print(f"  ! joint Wald failed for term {term}: {e}")
            chi2 = float("nan")
            p_val = float("nan")
        rows.append({
            "term": term,
            "df": df_val,
            "chi2": chi2,
            "p_value": p_val,
        })
    return pd.DataFrame(rows)


def _fit_lmm_one_cell(
    formula: str, df: pd.DataFrame
) -> Optional[Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]]:
    """Fit a single mixed-effects model with a random intercept on source_id.

    Returns a (coef_df, diagnostics) tuple, or None when the fit fails outright.
    The diagnostics dict carries σ²_source, σ²_residual, and the intraclass
    correlation (ICC = σ²_source / (σ²_source + σ²_residual)); ICC ≈ 0 means
    the LMM has degenerated to OLS — the per-source random intercept has no
    variance left to absorb after the fixed effects are in. That happens
    legitimately when the continuous covariates (reasoning_tokens, mean_recall)
    already explain the sentence-to-sentence baseline differences, and is the
    cause of statsmodels' "Random effects covariance is singular" warnings.
    We suppress those warnings here because they're informational, not
    actionable for fixed-effect inference, and surface the ICC instead so
    degeneracy is visible in the per-language print line.
    """
    # Single optimizer (L-BFGS — statsmodels' default for MixedLM). Older
    # versions of MixedLM.fit don't accept a list of methods; the list form
    # was the most likely cause of Mistral failing silently after the prior
    # edit. A boundary fit is still a valid fit — we don't gate on
    # `result.converged` because mixedlm legitimately reports converged=True
    # even when σ²_source is clipped at zero.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            md = smf.mixedlm(formula, data=df, groups=df["source_id"])
            result = md.fit(method="lbfgs", reml=True, disp=False)
    except Exception as e:
        print(f"  ! mixedlm fit failed: {e}")
        return None

    params = result.fe_params
    bse = result.bse_fe
    pvals = result.pvalues.loc[params.index]
    conf = result.conf_int().loc[params.index]
    n_obs = int(result.nobs)
    n_groups = int(df["source_id"].nunique())

    # Variance-component diagnostics. `result.cov_re` is the RE covariance
    # matrix; with a single random intercept it's 1×1 holding σ²_source.
    # `result.scale` is the residual variance σ². ICC = between / (between +
    # within). When σ²_source is clipped to 0 by the boundary, ICC is 0.
    try:
        re_var = float(np.asarray(result.cov_re).reshape(-1)[0])
    except Exception:
        re_var = float("nan")
    resid_var = float(getattr(result, "scale", float("nan")))
    if not (np.isnan(re_var) or np.isnan(resid_var)) and (re_var + resid_var) > 0:
        icc = re_var / (re_var + resid_var)
    else:
        icc = float("nan")

    rows: List[Dict[str, Any]] = []
    for coef_name in params.index:
        if coef_name == "Intercept":
            continue
        se = float(bse.loc[coef_name])
        coef = float(params.loc[coef_name])
        z = coef / se if se > 0 else float("nan")
        rows.append({
            "term": _classify_term(coef_name),
            "coef_name": coef_name,
            "coef": coef,
            "std_err": se,
            "z": z,
            "p_value": float(pvals.loc[coef_name]),
            "ci_low": float(conf.loc[coef_name, 0]),
            "ci_high": float(conf.loc[coef_name, 1]),
            "n_obs": n_obs,
            "n_source_groups": n_groups,
            "re_var": re_var,
            "resid_var": resid_var,
            "icc": icc,
        })
    # Proper joint Wald χ² per term — the rigorous "importance" metric.
    joint_df = _joint_wald_per_term(result)
    joint_df["n_obs"] = n_obs

    return (
        pd.DataFrame(rows),
        {"re_var": re_var, "resid_var": resid_var, "icc": icc},
        joint_df,
    )


def fit_lmm_per_language_on_only(
    df_master: pd.DataFrame,
    family: str,
    family_models: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Purpose: Fit a linear mixed-effects model per (family, tgt_lang) on
             ON-only data. The fixed-effect formula matches the ANOVA;
             a random intercept on source_id absorbs the cross-run
             dependence introduced by the same English sentence appearing
             in every (model, method, k) cell.
    Inputs: Master per-sentence DataFrame, family label, list of model_keys.
    Outputs: Long-form LMM DataFrame matching `LMM_COLUMNS`.
    """
    print(f"\n[Stage 4b / {family}] Per-language LMM on ON-only data, "
          "random intercept on source_id.")
    print(f"  family models: {family_models}")
    print(f"  fixed:  {_PER_LANG_LMM_FIXED}")
    print("  random: ~ 1 | source_id")

    dropna_cols = ["chrfpp_sentence", "reasoning_tokens", "mean_recall",
                   "tgt_lang", "method_label", "k", "source_id"]
    df_on = df_master.loc[
        (df_master["reasoning_state"] == "On")
        & (df_master["model_key"].isin(family_models))
    ].dropna(subset=dropna_cols).copy()
    df_on["k"] = df_on["k"].astype(int)
    # log1p(k) = log(k+1) — single continuous slope, matches the
    # diminishing-returns shape of in-context examples. Same column the
    # ANOVA formula reads.
    df_on["log1p_k"] = np.log1p(df_on["k"])

    coef_parts: List[pd.DataFrame] = []
    joint_parts: List[pd.DataFrame] = []
    for tgt_lang in sorted(df_on["tgt_lang"].unique()):
        sub = df_on.loc[df_on["tgt_lang"] == tgt_lang]
        if len(sub) < 30:
            print(f"  ! skip {tgt_lang} — only {len(sub)} usable rows.")
            continue
        fit_out = _fit_lmm_one_cell(_PER_LANG_LMM_FIXED, sub)
        if fit_out is None:
            continue
        lmm_df, diag, joint_df = fit_out
        if lmm_df.empty:
            continue
        lmm_df.insert(0, "tgt_lang", tgt_lang)
        lmm_df.insert(0, "family", family)
        coef_parts.append(lmm_df)
        if not joint_df.empty:
            joint_df = joint_df.copy()
            joint_df.insert(0, "tgt_lang", tgt_lang)
            joint_df.insert(0, "family", family)
            joint_parts.append(joint_df)
        # ICC < ~0.01 means σ²_source collapsed to the boundary; flag it so
        # the user can see at a glance which cells effectively ran as OLS.
        flag = "  ← σ²_src ≈ 0 (≈ OLS)" if diag["icc"] < 0.01 else ""
        # Surface the dominant term (largest joint Wald χ²) per cell — this
        # is the proper "most important factor" answer for this row.
        winner = ""
        if not joint_df.empty and joint_df["chi2"].notna().any():
            top = joint_df.sort_values("chi2", ascending=False).iloc[0]
            winner = f"  top={top['term']} χ²={top['chi2']:.1f}"
        print(
            f"  {tgt_lang}: n={len(sub):,}  source_groups={sub['source_id'].nunique():,}  "
            f"coefs={len(lmm_df)}  "
            f"σ²_src={diag['re_var']:.3g}  σ²_res={diag['resid_var']:.3g}  "
            f"ICC={diag['icc']:.4f}{flag}{winner}"
        )

    coef_out = (
        pd.concat(coef_parts, ignore_index=True)[LMM_COLUMNS]
        if coef_parts else pd.DataFrame(columns=LMM_COLUMNS)
    )
    joint_out = (
        pd.concat(joint_parts, ignore_index=True)[LMM_JOINT_COLUMNS]
        if joint_parts else pd.DataFrame(columns=LMM_JOINT_COLUMNS)
    )
    return coef_out, joint_out


def load_or_fit_lmm(
    df_master: pd.DataFrame,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Purpose: Cache-guarded per-language ON-only LMM fit, run separately per
             model family. Each fit produces both a per-coefficient table
             and a per-term joint Wald χ² table; both are cached.
    Inputs: Master per-sentence DataFrame.
    Outputs: Tuple (coef_by_family, joint_by_family). Each is a dict mapping
             family name → long-form DataFrame.
    """
    coef_out: Dict[str, pd.DataFrame] = {}
    joint_out: Dict[str, pd.DataFrame] = {}
    for family, family_models in MODEL_FAMILIES.items():
        coef_path = LMM_BY_FAMILY_OUTS[family]
        joint_path = LMM_JOINT_BY_FAMILY_OUTS[family]
        # Both CSVs must exist for a clean cache hit; if either is missing
        # we re-fit and rewrite both so they stay in lockstep.
        if LOAD_FROM_CSV and os.path.exists(coef_path) and os.path.exists(joint_path):
            print(f"\n[LOAD_FROM_CSV] Stage 4b / {family} cache hit (coef + joint).")
            cdf = pd.read_csv(coef_path)
            jdf = pd.read_csv(joint_path)
            _coerce_numeric_inplace(
                cdf,
                ["coef", "std_err", "z", "p_value", "ci_low", "ci_high",
                 "n_obs", "n_source_groups",
                 "re_var", "resid_var", "icc"],
            )
            _coerce_numeric_inplace(jdf, ["df", "chi2", "p_value", "n_obs"])
            print(f"  Loaded coef={len(cdf):,} rows, joint={len(jdf):,} rows.")
            coef_out[family] = cdf
            joint_out[family] = jdf
            continue

        cdf, jdf = fit_lmm_per_language_on_only(df_master, family, family_models)
        ensure_dir(OUTPUT_ROOT)
        cdf.to_csv(coef_path, index=False)
        jdf.to_csv(joint_path, index=False)
        print(f"[Stage 4b / {family}] Wrote {coef_path}")
        print(f"[Stage 4b / {family}] Wrote {joint_path}")
        coef_out[family] = cdf
        joint_out[family] = jdf
    return coef_out, joint_out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — Table 3 side-table: reasoning-token usage vs Δ(chrF++ ON − OFF)
# ─────────────────────────────────────────────────────────────────────────────


def compute_table3(df_master: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose: Per-language mean reasoning-token usage on, chrF++ ON/OFF means, and their delta.
    Inputs: Master frame.
    Outputs: DataFrame with TABLE3_COLUMNS plus a final summary row with the Spearman ρ.
    """
    print("\n[Stage 6] Computing per-language Δ(chrF++ ON − OFF) ranked by reasoning-token usage.")

    rows: List[Dict[str, Any]] = []
    for tgt_lang in TGT_LANGS:
        sub = df_master.loc[df_master["tgt_lang"] == tgt_lang]
        on = sub.loc[sub["reasoning_state"] == "On"]
        off = sub.loc[sub["reasoning_state"] == "Off"]
        if on.empty and off.empty:
            continue
        rows.append({
            "tgt_lang": tgt_lang,
            "tgt_lang_display": LANG_DISPLAY.get(tgt_lang, tgt_lang),
            "mean_reasoning_tokens_on": float(on["reasoning_tokens"].mean()),
            "chrfpp_on": float(on["chrfpp_sentence"].mean()) if not on.empty else float("nan"),
            "chrfpp_off": float(off["chrfpp_sentence"].mean()) if not off.empty else float("nan"),
            "delta_on_minus_off": (
                float(on["chrfpp_sentence"].mean() - off["chrfpp_sentence"].mean())
                if not on.empty and not off.empty
                else float("nan")
            ),
            "rank_by_tokens": -1,  # filled after sort
        })

    df = pd.DataFrame(rows, columns=TABLE3_COLUMNS)
    df = df.sort_values("mean_reasoning_tokens_on", ascending=False).reset_index(drop=True)
    df["rank_by_tokens"] = np.arange(1, len(df) + 1)

    # Append summary row with Spearman of (tokens, delta) across the 7 langs.
    tokens = df["mean_reasoning_tokens_on"].to_numpy(dtype=np.float64)
    delta = df["delta_on_minus_off"].to_numpy(dtype=np.float64)
    mask = ~(np.isnan(tokens) | np.isnan(delta))
    if mask.sum() >= 3 and not np.allclose(tokens[mask], tokens[mask][0]):
        sr = spearmanr(tokens[mask], delta[mask])
        summary_row = {
            "tgt_lang": "_SUMMARY_",
            "tgt_lang_display": (
                f"Spearman(tokens, delta) ρ={float(sr.statistic):+.4f} "
                f"(p={float(sr.pvalue):.4g}, n={int(mask.sum())})"
            ),
            "mean_reasoning_tokens_on": float("nan"),
            "rank_by_tokens": -1,
            "chrfpp_on": float("nan"),
            "chrfpp_off": float("nan"),
            "delta_on_minus_off": float("nan"),
        }
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    return df


def load_or_compute_table3(df_master: pd.DataFrame) -> pd.DataFrame:
    if LOAD_FROM_CSV and os.path.exists(TABLE3_OUT):
        print(f"\n[LOAD_FROM_CSV] Stage 6 cache hit: {TABLE3_OUT}")
        return pd.read_csv(TABLE3_OUT)
    df = compute_table3(df_master)
    ensure_dir(OUTPUT_ROOT)
    df.to_csv(TABLE3_OUT, index=False)
    print(f"[Stage 6] Wrote {TABLE3_OUT}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6b — Paired bootstrap significance per (model, method, direction, k)
# ─────────────────────────────────────────────────────────────────────────────
#
# Verbatim port of `sacrebleu.significance.PairedTest(test_type="bs")` for the
# continuous per-sentence chrF++ signal — same machinery
# `paired_bootstrap_lid_significance.py` uses for binary LID correctness, just
# with a continuous mean instead of `mean(0/1)`. Per-sentence chrF++ pairing is
# natural at this grain (same source sentence translated by ON and OFF systems
# at the same (model, method, direction, k)), so `sentence_idx` plays the role
# of segment id in sacrebleu's bootstrap.


def estimate_ci(scores: np.ndarray) -> Tuple[float, float]:
    """Percentile-based 95% CI — mirror of sacrebleu.significance.estimate_ci."""
    scores_sorted = np.sort(scores)
    n = len(scores_sorted)
    lower_idx = n // 40
    upper_idx = n - lower_idx - 1
    lower = scores_sorted[lower_idx]
    upper = scores_sorted[upper_idx]
    ci = 0.5 * (upper - lower)
    return float(scores_sorted.mean()), float(ci)


def _compute_p_value(stats: np.ndarray, real_difference: float) -> float:
    """Mirror of sacrebleu.significance._compute_p_value: strict `>` with plus-one smoothing."""
    c = int(np.sum(stats > real_difference).item())
    return (c + 1) / (len(stats) + 1)


def run_paired_bs_chrfpp(
    scores_on: np.ndarray,
    scores_off: np.ndarray,
    n_samples: int = PAIRED_BS_N,
    seed: int = SACREBLEU_SEED,
) -> Dict[str, float]:
    """
    Purpose: Paired bootstrap of mean per-sentence chrF++, ON vs OFF.
    Inputs: Aligned per-sentence chrF++ score arrays for the two systems.
    Outputs: Dict with score_on/off, mean_on/off, ci_on/off, p_value.

    For the continuous mean statistic, sacrebleu's `_compute_score_from_stats`
    over [score_sum, 1] per segment collapses to `mean(resampled) * 100/100`
    — i.e. simple resampled mean. Same resampling (`rng.choice` with replace),
    same percentile CI, same two-sided p-value form as the chrF++ corpus path.
    """
    n = int(scores_on.shape[0])
    if n == 0 or scores_off.shape[0] != n:
        nan = float("nan")
        return {
            "score_off": nan, "score_on": nan,
            "mean_off": nan, "ci_off": nan,
            "mean_on": nan, "ci_on": nan,
            "p_value": nan,
        }

    score_on = float(scores_on.mean())
    score_off = float(scores_off.mean())

    # Degenerate-input guard: if every paired sentence has identical ON and OFF
    # chrF++, every bootstrap resample produces bs_on == bs_off and sacrebleu's
    # plus-one smoothing spuriously reports p ≈ 1/(n_samples+1). Report p=1.0
    # in this case (no inferential signal to test).
    if np.array_equal(scores_on, scores_off):
        return {
            "score_off": score_off, "score_on": score_on,
            "mean_off": score_off, "ci_off": 0.0,
            "mean_on": score_on, "ci_on": 0.0,
            "p_value": 1.0,
        }

    rng = np.random.RandomState(seed)
    idxs = rng.choice(n, size=(n_samples, n), replace=True)
    bs_on = scores_on[idxs].mean(axis=1)
    bs_off = scores_off[idxs].mean(axis=1)

    mean_on, ci_on = estimate_ci(bs_on)
    mean_off, ci_off = estimate_ci(bs_off)

    sample_diffs = np.abs(bs_on - bs_off)
    stats = sample_diffs - sample_diffs.mean()
    real_difference = abs(score_on - score_off)
    p_value = _compute_p_value(stats, real_difference)

    return {
        "score_off": score_off, "score_on": score_on,
        "mean_off": mean_off, "ci_off": ci_off,
        "mean_on": mean_on, "ci_on": ci_on,
        "p_value": float(p_value),
    }


def compute_paired_bs_significance(df_master: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose: For every (model, method, direction, k) cell, paired-bootstrap ON vs OFF per-sentence chrF++.
    Inputs: Master per-sentence DataFrame.
    Outputs: Long-form DataFrame with PAIRED_BS_LONG_COLUMNS — schema matches the file-level paired bootstrap script.
    """
    print(f"\n[Stage 6b] Paired bootstrap (n={PAIRED_BS_N}, seed={SACREBLEU_SEED}) "
          "on per-sentence chrF++: ON vs OFF, per (model, method, direction, k).")

    rows: List[Dict[str, Any]] = []
    n_cells_skipped = 0
    n_cells_run = 0

    for model_key in MODELS:
        model_display = MODELS[model_key]
        for src_lang in SRC_LANGS:
            for tgt_lang in TGT_LANGS:
                direction = direction_folder_name(src_lang, tgt_lang)
                for method_key, (on_run_key, off_run_key, method_label) in METHOD_KEY_TO_RUN_KEYS.items():
                    for k in sorted(set(K_LIST)):
                        on_sub = df_master.loc[
                            (df_master["model_key"] == model_key)
                            & (df_master["src_lang"] == src_lang)
                            & (df_master["tgt_lang"] == tgt_lang)
                            & (df_master["run_key"] == on_run_key)
                            & (df_master["k"] == k),
                            ["sentence_idx", "chrfpp_sentence"],
                        ]
                        off_sub = df_master.loc[
                            (df_master["model_key"] == model_key)
                            & (df_master["src_lang"] == src_lang)
                            & (df_master["tgt_lang"] == tgt_lang)
                            & (df_master["run_key"] == off_run_key)
                            & (df_master["k"] == k),
                            ["sentence_idx", "chrfpp_sentence"],
                        ]
                        if on_sub.empty or off_sub.empty:
                            n_cells_skipped += 1
                            continue

                        # Inner-join on sentence_idx so the arrays align.
                        merged = on_sub.merge(
                            off_sub.rename(columns={"chrfpp_sentence": "chrfpp_off"}),
                            on="sentence_idx",
                            how="inner",
                        ).rename(columns={"chrfpp_sentence": "chrfpp_on"})
                        if merged.empty:
                            n_cells_skipped += 1
                            continue

                        scores_on = merged["chrfpp_on"].to_numpy(dtype=np.float64)
                        scores_off = merged["chrfpp_off"].to_numpy(dtype=np.float64)
                        result = run_paired_bs_chrfpp(scores_on, scores_off)
                        delta = result["score_on"] - result["score_off"]
                        p_value = result["p_value"]
                        sig_005 = pd.notna(p_value) and float(p_value) < 0.05
                        sig_001 = pd.notna(p_value) and float(p_value) < 0.01
                        sig_marker = "**" if sig_001 else "*" if sig_005 else ""

                        rows.append({
                            "model_key": model_key,
                            "model_display": model_display,
                            "method_key": method_key,
                            "method_label": method_label,
                            "src_lang": src_lang,
                            "tgt_lang": tgt_lang,
                            "direction": direction,
                            "k": int(k),
                            "on_run_key": on_run_key,
                            "on_run_display": f"{method_label} Reasoning On",
                            "off_run_key": off_run_key,
                            "off_run_display": f"{method_label} Reasoning Off",
                            "score_on": result["score_on"],
                            "score_off": result["score_off"],
                            "delta": delta,
                            "mean_on": result["mean_on"],
                            "ci_on": result["ci_on"],
                            "mean_off": result["mean_off"],
                            "ci_off": result["ci_off"],
                            "p_value": p_value,
                            "significant_p_0_05": bool(sig_005),
                            "significant_p_0_01": bool(sig_001),
                            "sig_marker": sig_marker,
                            "better_than_off": bool(delta > 0),
                            "n_segments": int(len(merged)),
                        })
                        n_cells_run += 1

    print(f"[Stage 6b] Ran bootstrap on {n_cells_run:,} cells; "
          f"skipped {n_cells_skipped:,} (no paired data).")
    return pd.DataFrame(rows, columns=PAIRED_BS_LONG_COLUMNS)


def load_or_compute_paired_bs(df_master: pd.DataFrame) -> pd.DataFrame:
    """Cache-guarded paired-bootstrap CSV."""
    if LOAD_FROM_CSV and os.path.exists(PAIRED_BS_OUT):
        print(f"\n[LOAD_FROM_CSV] Stage 6b cache hit: {PAIRED_BS_OUT}")
        df = pd.read_csv(PAIRED_BS_OUT)
        if "k" in df.columns:
            df["k"] = df["k"].astype(int)
        _coerce_numeric_inplace(df, [
            "score_on", "score_off", "delta", "mean_on", "ci_on",
            "mean_off", "ci_off", "p_value", "n_segments",
        ])
        print(f"  Loaded {len(df):,} rows.")
        return df

    df = compute_paired_bs_significance(df_master)
    ensure_dir(OUTPUT_ROOT)
    df.to_csv(PAIRED_BS_OUT, index=False)
    print(f"[Stage 6b] Wrote {PAIRED_BS_OUT}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 — eval_pipeline-style plot suite for chrF++ (sentence-mean)
# ─────────────────────────────────────────────────────────────────────────────
#
# Lifted from eval_pipeline.py with two adaptations:
#   (a) we have one metric — `chrF++` derived from mean(chrfpp_sentence) per
#       file — so the per-metric loops collapse to a single pass;
#   (b) k=0 is fixed out of the plot suite (INCLUDE_K0_IN_PLOTS=False) per the
#       user's instruction; CSV cache rows for k=0 still survive untouched.
# Aesthetics (rcParams, seaborn theme, table palette) are identical so the
# output is visually a drop-in replacement.

# ── Global aesthetics (lifted from eval_pipeline.py) ────────────────────────
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "font.family":        "serif",
    "axes.titlesize":     26,
    "axes.titleweight":   "bold",
    "axes.labelsize":     22,
    "xtick.labelsize":    19,
    "ytick.labelsize":    19,
    "legend.fontsize":    19,
    "legend.title_fontsize": 21,
    "figure.dpi":         150,
    "lines.linewidth":    3.0,
    "lines.markersize":   12,
    "lines.markeredgecolor": "white",
    "lines.markeredgewidth": 1.0,
    "axes.titlepad":      14,
    "axes.labelpad":      10,
    "axes.linewidth":     1.2,
    "axes.edgecolor":     "#333333",
    "xtick.major.width":  1.0,
    "ytick.major.width":  1.0,
    "xtick.major.size":   5.5,
    "ytick.major.size":   5.5,
    "xtick.color":        "#333333",
    "ytick.color":        "#333333",
    "axes.labelcolor":    "#1a1a1a",
    "axes.titlecolor":    "#1a1a1a",
    "legend.frameon":     True,
    "legend.framealpha":  0.96,
    "legend.edgecolor":   "#999999",
    "legend.fancybox":    True,
})
_TBL_HEADER_BG = "#2c3e50"
_TBL_HEADER_FG = "white"
_TBL_ROW_ODD = "#f0f4f8"
_TBL_ROW_EVEN = "white"
_TBL_ROWLBL_BG = "#dce8f5"

# Paper-ready font sizes for the superplot grids. Sized for the
# (`_SUBPLOT_W` × `_SUBPLOT_H`) per-subplot canvas at dpi=250; pt values
# below render large enough that the labels stay clearly legible when the
# figure is shrunk into a paper column or pagewidth.
_SUPTITLE_FONTSIZE: int = 56          # figure-level suptitle
_SUBPLOT_TITLE_FONTSIZE: int = 34     # per-panel language / model name
_SUPAXIS_FONTSIZE: int = 40           # shared supxlabel + supylabel
_TICK_FONTSIZE: int = 30              # numeric tick labels on every axis
_LEGEND_FONTSIZE: int = 28
_LEGEND_TITLE_FONTSIZE: int = 32

# Per-subplot figure dimensions for every multi-panel superplot. Bumped
# from (12, 8.5) so each panel renders bigger; the full figure size is
# (`_SUBPLOT_W * nc`, `_SUBPLOT_H * nr`) — so for the 2×4 layout used by
# the 7-language plots the figure is 56 × 19 inches at 250 dpi.
_SUBPLOT_W: float = 14.0
_SUBPLOT_H: float = 9.5

_PREFIX_DISPLAY: Dict[str, str] = {
    c.split("_")[0].lower(): n for c, n in LANG_DISPLAY.items()
}


def lang_display(fc: str) -> str:
    return LANG_DISPLAY.get(fc, _PREFIX_DISPLAY.get(fc.split("_")[0].lower(), fc))


def direction_display_from_folder(folder: str) -> str:
    parts = folder.split("_to_")
    if len(parts) == 2:
        src = _PREFIX_DISPLAY.get(parts[0], parts[0].capitalize())
        tgt = _PREFIX_DISPLAY.get(parts[1], parts[1].capitalize())
        return f"{src} to {tgt}"
    return folder


def parse_model_size(dn: str) -> float:
    m = re.search(r"([\d]+(?:\.\d+)?)\s*[Bb]", dn)
    return float(m.group(1)) if m else 0.0


def _small_and_large_model_keys(
    threshold: float, restrict_to: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    small, large = [], []
    for mk in (restrict_to if restrict_to else list(MODELS.keys())):
        (small if parse_model_size(MODELS.get(mk, mk)) < threshold else large).append(mk)
    return small, large


def metric_precision(_m: str) -> int:
    return 2


def format_metric_value(metric: str, v: Any) -> str:
    if v is None:
        return "—"
    try:
        if math.isnan(float(v)):
            return "—"
    except Exception:
        return "—"
    return f"{float(v):.{metric_precision(metric)}f}"


def _apply_k_plot_filter(df_in: pd.DataFrame) -> pd.DataFrame:
    """Drop k=0 rows when INCLUDE_K0_IN_PLOTS is False; pass-through otherwise."""
    if INCLUDE_K0_IN_PLOTS or "k" not in df_in.columns:
        return df_in
    return df_in[df_in["k"] > 0]


def _superplot_grid(n: int) -> Tuple[int, int]:
    """
    Pick (nrows, ncols) for an n-subplot superplot.

    Special cases — paper-friendly: when n is in {7, 8, 10, 11} a 4-column
    grid leaves at most 1 empty cell (versus 2 with the canonical 3-column
    rule). For 7 directions this is the case that previously left a
    "half-empty bottom row"; the 2×4 layout balances the figure and gives
    the legend a clean single-cell home.
    """
    if n in (7, 8, 10, 11):
        nc = 4
    elif n > 0:
        nc = min(3, n)
    else:
        nc = 1
    return math.ceil(n / max(1, nc)), nc


def _grid_empty_count(n: int) -> Tuple[int, int]:
    """Return (n_used, n_empty) for the superplot grid that would be picked
    for n. Used by the per-superplot save_* helpers to scale legend padding
    based on how much empty cell area is actually available."""
    nr, nc = _superplot_grid(n)
    total = nr * nc
    return n, max(0, total - n)


def _polish_axes(ax, axhline_zero: bool = False) -> None:
    """
    Paper-ready cosmetic polish: subtler gridlines that sit BELOW the data,
    hidden top/right spines, slightly bolder remaining spines. Idempotent —
    safe to call once per axes after the seaborn `lineplot` finishes drawing.

    If `axhline_zero=True`, also draws the dashed reference line at y=0 (for
    delta-style plots) ABOVE the gridlines but below the data lines.
    """
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=1.0,
            color="#bdc3c7", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    if axhline_zero:
        ax.axhline(0, color="#555555", linewidth=1.4, linestyle="--",
                   alpha=0.85, zorder=1.5)


def _place_figure_legend(fig, axes_flat, n_used, legend_handles, legend_labels,
                          title="Method", max_ncol=4, gap_width_scale=1.0,
                          bottom_legend_w=0.36):
    """Lifted verbatim from eval_pipeline.py — places a shared legend either
    into an empty subplot slot or below the figure, picking ncol from the
    available aspect ratio. `bottom_legend_w` controls the bottom-right
    legend width (as a fraction of figure width) when the grid has no empty
    cells; right edge is anchored at x=0.98 regardless."""
    if not legend_handles or not legend_labels:
        return

    n_total = len(axes_flat)
    n_empty = n_total - n_used

    fig.canvas.draw()
    occ_bboxes = [axes_flat[i].get_position() for i in range(n_used)]

    if n_empty >= 1:
        empty_bboxes = [axes_flat[i].get_position() for i in range(n_used, n_total)]
        emp_x0 = min(b.x0 for b in empty_bboxes); emp_x1 = max(b.x1 for b in empty_bboxes)
        emp_y0 = min(b.y0 for b in empty_bboxes); emp_y1 = max(b.y1 for b in empty_bboxes)
        emp_w = emp_x1 - emp_x0; emp_h = emp_y1 - emp_y0
        pad_x = emp_w * 0.05; pad_y = emp_h * 0.07
        leg_x0 = emp_x0 + pad_x; leg_w = emp_w - 2 * pad_x
        leg_y0 = emp_y0 + pad_y; leg_h = emp_h - 2 * pad_y

        if gap_width_scale != 1.0:
            new_leg_w = min(leg_w * gap_width_scale, 0.98)
            leg_x0 = max(0.01, (emp_x0 + emp_x1 - new_leg_w) / 2.0)
            leg_w = new_leg_w

        fig_w_in, fig_h_in = fig.get_size_inches()
        leg_w_in = leg_w * fig_w_in
        leg_h_in = leg_h * fig_h_in
        aspect = leg_w_in / leg_h_in if leg_h_in > 0 else 1.0
        n_entries = len(legend_labels)
        if aspect >= 4.0:
            ncol = min(4, max_ncol) if n_entries >= 8 else (3 if n_entries >= 6 else min(n_entries, max_ncol))
        elif aspect >= 1.4:
            ncol = min(n_entries, 2)
        else:
            ncol = 1

        leg = fig.legend(
            legend_handles, legend_labels,
            loc="center", ncol=ncol,
            bbox_to_anchor=(leg_x0, leg_y0, leg_w, leg_h),
            bbox_transform=fig.transFigure,
            frameon=True, framealpha=0.97, edgecolor="#999999",
            fontsize=_LEGEND_FONTSIZE, title=title,
            title_fontsize=_LEGEND_TITLE_FONTSIZE,
            borderpad=1.4, labelspacing=1.4,
            handlelength=3.0, handleheight=1.9,
            handletextpad=1.0, columnspacing=2.4,
        )
        leg.get_title().set_fontweight("bold")

        renderer = fig.canvas.get_renderer()
        col_width_pts = (leg_w * fig.get_figwidth() * fig.dpi) / ncol * 0.85
        for txt in leg.get_texts():
            label = txt.get_text()
            bb = txt.get_window_extent(renderer=renderer)
            if bb.width > col_width_pts and len(label) > 12:
                mid = len(label) // 2
                best = -1
                for i in range(mid, -1, -1):
                    if label[i] == " ":
                        best = i; break
                if best == -1:
                    for i in range(mid, len(label)):
                        if label[i] == " ":
                            best = i; break
                if best != -1:
                    txt.set_text(label[:best] + "\n" + label[best + 1:])
    else:
        # No empty cells in the grid — place the legend INSIDE the bottom
        # margin of the figure, *below* the supxlabel, in the bottom-RIGHT
        # corner. Stack top-to-bottom is:
        #   panels (y ≥ 0.225) → supxlabel (centered at x=0.5, y≈0.190)
        #   → legend (lower-right anchor at (0.98, 0.030), so the legend's
        #     right edge sits at x=0.98 — hard against the figure right
        #     edge — and it extends leftward to ~x=0.68).
        # The centered supxlabel at x≈0.5 has the entire left half of the
        # bottom band to itself.
        # Bottom-right placement with a hard dark border and explicit
        # horizontal extent. 4-tuple `bbox_to_anchor=(x, y, w, h)` defines
        # an anchor rectangle whose right edge is pinned at x=0.98 and
        # whose width is `bottom_legend_w` (caller-controlled, default
        # 0.36); `mode="expand"` forces the legend to expand horizontally
        # to fill that block, which gives the legend a substantial
        # bottom-right region without ever crossing the centered supxlabel
        # (~x ∈ [0.42, 0.58]) so long as bottom_legend_w stays under ~0.40
        # — the supxlabel sits at y=0.190, the legend extends up only to
        # ~y=0.10 from its y=0.030 bottom, so they don't visually collide
        # vertically either. Caller picks `max_ncol` to lay out entries.
        ncol = min(len(legend_labels), max_ncol)
        bottom_legend_x = 0.98 - bottom_legend_w
        leg = fig.legend(
            legend_handles, legend_labels,
            loc="lower right",
            bbox_to_anchor=(bottom_legend_x, 0.030, bottom_legend_w, 0.18),
            bbox_transform=fig.transFigure,
            mode="expand",
            ncol=ncol,
            frameon=True, framealpha=1.0, edgecolor="#1a1a1a",
            fontsize=_LEGEND_FONTSIZE,
            title=title,
            title_fontsize=_LEGEND_TITLE_FONTSIZE,
            borderpad=1.0, labelspacing=0.8,
            handlelength=2.8, handleheight=1.0,
            handletextpad=0.8, columnspacing=2.4,
        )
        leg.get_title().set_fontweight("bold")
        # Hard border — thick black-ish frame line.
        leg.get_frame().set_linewidth(2.5)
        leg.get_frame().set_edgecolor("#1a1a1a")


def _attach_outside_legend(fig, hax, labels=None, ncol=4, title=None):
    if isinstance(hax, plt.Axes):
        handles, labels = hax.get_legend_handles_labels()
        labels = [l for l in labels if l not in ("run_display", "method_label")]
    else:
        handles = hax
    if not handles:
        return
    kw = dict(loc="lower center", ncol=min(len(labels), ncol), frameon=True,
              framealpha=0.95, edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.10),
              fontsize=_LEGEND_FONTSIZE, title_fontsize=_LEGEND_TITLE_FONTSIZE,
              borderpad=1.2, labelspacing=1.2,
              handlelength=3.0, handletextpad=1.0, columnspacing=2.2)
    if title:
        kw["title"] = title
    fig.legend(handles, labels, **kw)


def _professional_table_axes(ax, pivot: pd.DataFrame, metric: str, title: str):
    ax.axis("off")
    if pivot.empty or pivot.isna().all().all():
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                fontsize=22, style="italic", color="#555555", transform=ax.transAxes)
        ax.set_title(title, fontsize=_SUBPLOT_TITLE_FONTSIZE,
                     fontweight="bold", pad=22)
        return
    rl = [str(r) for r in pivot.index]
    cl = [f"k = {c}" for c in pivot.columns]
    ct = [[format_metric_value(metric, pivot.loc[r, c]) for c in pivot.columns]
          for r in pivot.index]
    nr, nc2 = len(rl), len(cl)
    tbl = ax.table(cellText=ct, rowLabels=rl, colLabels=cl, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(22); tbl.scale(1.2, 3.2)
    for j in range(nc2):
        cell = tbl[0, j]
        cell.set_facecolor(_TBL_HEADER_BG)
        cell.set_text_props(color=_TBL_HEADER_FG, fontweight="bold")
        cell.set_edgecolor("white")
    for i in range(nr):
        rb = _TBL_ROW_ODD if i % 2 == 0 else _TBL_ROW_EVEN
        lc = tbl[i + 1, -1]
        lc.set_facecolor(_TBL_ROWLBL_BG)
        lc.set_text_props(fontweight="bold")
        lc.set_edgecolor("#cccccc")
        for j in range(nc2):
            cell = tbl[i + 1, j]
            cell.set_facecolor(rb); cell.set_edgecolor("#cccccc")
    ax.set_title(title, fontsize=_SUBPLOT_TITLE_FONTSIZE,
                 fontweight="bold", pad=22)


# ── Long-format helpers (DF_LONG_COLUMNS / DF_DELTA_COLUMNS / aggregated) ──

DF_LONG_COLUMNS_PLOT = [
    "model_key", "model_display", "run_key", "run_display",
    "src_lang", "tgt_lang", "direction", "k", "metric", "score", "path",
]
DF_DELTA_COLUMNS_PLOT = [
    "model_key", "model_display", "direction", "k", "metric",
    "delta", "method_label", "on_key", "off_key",
]
DF_AGG_COLUMNS_PLOT = [
    "model_key", "model_display", "run_key", "run_display",
    "k", "metric", "mean_score", "method_label", "reasoning_state",
]
DF_AGG_DELTA_COLUMNS_PLOT = [
    "model_key", "model_display", "k", "metric", "delta", "method_label",
]


def build_df_long_chrfpp(df_master: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose: Aggregate per-sentence chrF++ → file-level mean per (model, run, direction, k) in eval_pipeline's DF_LONG schema.
    Inputs: Master per-sentence DataFrame.
    Outputs: DataFrame with DF_LONG_COLUMNS_PLOT, one row per file, metric = "chrF++".
    """
    agg = (
        df_master.groupby(
            ["model_key", "model_display", "run_key", "run_display",
             "src_lang", "tgt_lang", "direction", "k"],
            as_index=False,
            sort=False,
        )
        .agg(score=("chrfpp_sentence", "mean"))
    )
    agg["metric"] = METRIC_NAME_FOR_PLOTS
    agg["path"] = ""  # not used by plotting helpers
    return agg.reindex(columns=DF_LONG_COLUMNS_PLOT)


def compute_delta_df_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Δ(chrF++ ON − OFF) per (model, direction, k, method)."""
    parts: List[pd.DataFrame] = []
    ic = ["model_key", "model_display", "direction", "k", "metric"]
    for method_key, (on_run_key, off_run_key, method_label) in METHOD_KEY_TO_RUN_KEYS.items():
        on_scores = df[df["run_key"] == on_run_key].set_index(ic)["score"]
        off_scores = df[df["run_key"] == off_run_key].set_index(ic)["score"]
        ds = (on_scores - off_scores).dropna()
        if ds.empty:
            continue
        d = ds.reset_index()
        d.columns = ic + ["delta"]
        d["method_label"] = method_label
        d["on_key"] = on_run_key
        d["off_key"] = off_run_key
        parts.append(d)
    return (
        pd.concat(parts, ignore_index=True)[DF_DELTA_COLUMNS_PLOT]
        if parts else pd.DataFrame(columns=DF_DELTA_COLUMNS_PLOT)
    )


def compute_aggregated_scores_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Mean chrF++ per (model, run, k) — averages over all directions."""
    gc = ["model_key", "model_display", "run_key", "run_display", "k", "metric"]
    da = df.groupby(gc, as_index=False)["score"].mean().rename(columns={"score": "mean_score"})

    run_meta: Dict[str, Tuple[str, str]] = {}
    for method_key, (on_run_key, off_run_key, method_label) in METHOD_KEY_TO_RUN_KEYS.items():
        run_meta[on_run_key] = (method_label, "On")
        run_meta[off_run_key] = (method_label, "Off")

    da["method_label"] = da["run_key"].map(lambda k: run_meta.get(k, ("Unknown", "Unknown"))[0])
    da["reasoning_state"] = da["run_key"].map(lambda k: run_meta.get(k, ("Unknown", "Unknown"))[1])
    return da[DF_AGG_COLUMNS_PLOT]


def compute_aggregated_deltas_plot(da: pd.DataFrame) -> pd.DataFrame:
    """Δ(mean chrF++ ON − OFF) per (model, k, method) — averaged across all directions."""
    parts: List[pd.DataFrame] = []
    ic = ["model_key", "model_display", "k", "metric"]
    for method_key, (on_run_key, off_run_key, method_label) in METHOD_KEY_TO_RUN_KEYS.items():
        on_scores = da[da["run_key"] == on_run_key].set_index(ic)["mean_score"]
        off_scores = da[da["run_key"] == off_run_key].set_index(ic)["mean_score"]
        ds = (on_scores - off_scores).dropna()
        if ds.empty:
            continue
        d = ds.reset_index()
        d.columns = ic + ["delta"]
        d["method_label"] = method_label
        parts.append(d)
    return (
        pd.concat(parts, ignore_index=True)[DF_AGG_DELTA_COLUMNS_PLOT]
        if parts else pd.DataFrame(columns=DF_AGG_DELTA_COLUMNS_PLOT)
    )


# ── Plot routines (lifted from eval_pipeline.py) ───────────────────────────


def save_metric_plot(df: pd.DataFrame, mk: str, dr: str, met: str, op: str) -> None:
    dp = df[(df["model_key"] == mk) & (df["direction"] == dr)
            & (df["metric"] == met) & (~df["score"].isna())].copy()
    dp = _apply_k_plot_filter(dp)
    md = MODELS.get(mk, mk)
    dd = direction_display_from_folder(dr)
    ensure_dir(os.path.dirname(op))
    fig, ax = plt.subplots(figsize=(16, 10))
    if dp.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data found", ha="center", va="center",
                fontsize=26, style="italic")
    else:
        dp = dp.sort_values(["run_display", "k"], kind="stable")
        sns.lineplot(data=dp, x="k", y="score", hue="run_display", style="run_display",
                     markers=True, dashes=False, linewidth=3.0, markersize=12, ax=ax)
        ax.set_xticks(K_LIST_PLOT)
        ax.set_xlabel("Number of Examples (k)", fontsize=_SUPAXIS_FONTSIZE, labelpad=14)
        ax.set_ylabel(met, fontsize=_SUPAXIS_FONTSIZE, labelpad=14)
        _polish_axes(ax)
        ax.tick_params(axis="both", labelsize=_TICK_FONTSIZE)
        h, l = ax.get_legend_handles_labels()
        l = [x for x in l if x != "run_display"]
        if ax.legend_:
            ax.legend_.remove()
        _attach_outside_legend(fig, h, l, ncol=4, title="Method")
    ax.set_title(f"{md} — {dd} — {met}", fontsize=_SUPTITLE_FONTSIZE,
                 fontweight="bold", pad=22)
    fig.tight_layout(); fig.savefig(op, dpi=250, bbox_inches="tight"); plt.close(fig)


def save_metric_table(df: pd.DataFrame, mk: str, dr: str, met: str, op: str) -> None:
    dt = df[(df["model_key"] == mk) & (df["direction"] == dr)
            & (df["metric"] == met)].copy()
    dt = _apply_k_plot_filter(dt)
    piv = dt.pivot_table(index="run_display", columns="k", values="score",
                         aggfunc="first").reindex(columns=K_LIST_PLOT)
    ro = list(dict.fromkeys(dt["run_display"].tolist()))
    if ro:
        piv = piv.reindex(index=ro)
    md = MODELS.get(mk, mk)
    dd = direction_display_from_folder(dr)
    nr = max(1, len(piv.index)); nc = max(1, len(piv.columns))
    ensure_dir(os.path.dirname(op))
    fig, ax = plt.subplots(
        figsize=(max(13.0, 2.6 * nc + 6.0), max(5.5, 1.2 * nr + 4.0))
    )
    _professional_table_axes(ax, piv, met, f"{md} — {dd} — {met}")
    fig.tight_layout(); fig.savefig(op, dpi=250, bbox_inches="tight"); plt.close(fig)


def save_model_metric_superplot(df: pd.DataFrame, mk: str, met: str,
                                 dord: List[str], op: str) -> None:
    ensure_dir(os.path.dirname(op))
    ds = df[(df["model_key"] == mk) & (df["metric"] == met)].copy()
    md = MODELS.get(mk, mk)
    ds = _apply_k_plot_filter(ds)
    avail = [d for d in dord if d in set(ds["direction"].tolist())]
    if not avail:
        avail = sorted(ds["direction"].dropna().unique().tolist())
    n = len(avail)
    if n == 0:
        fig, ax = plt.subplots(figsize=(9, 5)); ax.axis("off")
        ax.text(0.5, 0.5, "No data found", ha="center", va="center", fontsize=22)
        fig.tight_layout(); fig.savefig(op, dpi=250, bbox_inches="tight"); plt.close(fig)
        return
    nr, nc = _superplot_grid(n)
    fig, axes = plt.subplots(nr, nc, figsize=(_SUBPLOT_W * nc, _SUBPLOT_H * nr), squeeze=False)
    af = axes.flatten(); lh = ll = None
    for ax, d in zip(af, avail):
        dd = direction_display_from_folder(d)
        df2 = ds[(ds["direction"] == d) & (~ds["score"].isna())].copy()
        ax.set_title(dd, fontsize=_SUBPLOT_TITLE_FONTSIZE, fontweight="bold", pad=12)
        ax.set_xticks(K_LIST_PLOT)
        ax.tick_params(axis="both", labelsize=_TICK_FONTSIZE)
        if df2.empty:
            _polish_axes(ax)
            ax.text(0.5, 0.5, "No data found", ha="center", va="center",
                    fontsize=22, style="italic", transform=ax.transAxes)
            continue
        sns.lineplot(
            data=df2.sort_values(["run_display", "k"], kind="stable"),
            x="k", y="score",
            hue="run_display", style="run_display",
            markers=True, dashes=False,
            linewidth=3.0, markersize=12, ax=ax, legend=True,
        )
        _polish_axes(ax)
        # seaborn auto-labels the axes from the column names (`x="k"` writes
        # "k" as the x-axis label). Clear them so the figure-level
        # supxlabel / supylabel are the only axis text.
        ax.set_xlabel("")
        ax.set_ylabel("")
        lh, ll = _collect_legend(ax, lh, ll)
        if ax.legend_:
            ax.legend_.remove()
    for ax in af[n:]:
        ax.axis("off")
    fig.suptitle(f"{md} — {met} Across Translation Directions",
                 fontsize=_SUPTITLE_FONTSIZE, fontweight="bold", y=0.995)
    # Use fig.text directly — fig.supxlabel(y=...) gets quietly repositioned
    # by matplotlib's layout system, which is why prior attempts to lift the
    # x-axis label above the legend strip kept failing.
    fig.text(0.5, 0.190, "Number of Examples (k)",
             ha="center", va="center",
             fontsize=_SUPAXIS_FONTSIZE, fontweight="bold")
    fig.text(0.018, 0.588, met,
             ha="center", va="center", rotation="vertical",
             fontsize=_SUPAXIS_FONTSIZE, fontweight="bold")
    # Reserve space at the top (suptitle), bottom (supxlabel), and left
    # (supylabel) so tight_layout doesn't crop them.
    fig.tight_layout(rect=[0.030, 0.225, 1, 0.950])
    _, n_empty = _grid_empty_count(n)
    gap_scale = 1.0 if n_empty <= 1 else 5.0
    _place_figure_legend(fig, af, n, lh, ll, title="Method", gap_width_scale=gap_scale)
    fig.savefig(op, dpi=250, bbox_inches="tight"); plt.close(fig)


def save_delta_superplot(dfd: pd.DataFrame, mk: str, met: str,
                          dord: List[str], op: str) -> None:
    ensure_dir(os.path.dirname(op))
    ds = dfd[(dfd["model_key"] == mk) & (dfd["metric"] == met)].copy()
    md = MODELS.get(mk, mk)
    ds = _apply_k_plot_filter(ds)
    avail = [d for d in dord if d in set(ds["direction"].tolist())]
    if not avail:
        avail = sorted(ds["direction"].dropna().unique().tolist())
    n = len(avail)
    if n == 0:
        fig, ax = plt.subplots(figsize=(9, 5)); ax.axis("off")
        ax.text(0.5, 0.5, "No data found", ha="center", va="center", fontsize=22)
        fig.tight_layout(); fig.savefig(op, dpi=250, bbox_inches="tight"); plt.close(fig)
        return
    nr, nc = _superplot_grid(n)
    fig, axes = plt.subplots(nr, nc, figsize=(_SUBPLOT_W * nc, _SUBPLOT_H * nr), squeeze=False)
    af = axes.flatten(); lh = ll = None
    for ax, d in zip(af, avail):
        dd = direction_display_from_folder(d)
        df2 = ds[ds["direction"] == d].copy()
        ax.set_title(dd, fontsize=_SUBPLOT_TITLE_FONTSIZE, fontweight="bold", pad=12)
        ax.set_xticks(K_LIST_PLOT)
        ax.tick_params(axis="both", labelsize=_TICK_FONTSIZE)
        if df2.empty:
            _polish_axes(ax, axhline_zero=True)
            ax.text(0.5, 0.5, "No data found", ha="center", va="center",
                    fontsize=22, style="italic", transform=ax.transAxes)
            continue
        sns.lineplot(
            data=df2.sort_values(["method_label", "k"], kind="stable"),
            x="k", y="delta",
            hue="method_label", style="method_label",
            markers=True, dashes=False,
            linewidth=3.0, markersize=12, ax=ax, legend=True,
        )
        _polish_axes(ax, axhline_zero=True)
        ax.set_xlabel("")
        ax.set_ylabel("")
        lh, ll = _collect_legend(ax, lh, ll)
        if ax.legend_:
            ax.legend_.remove()
    for ax in af[n:]:
        ax.axis("off")
    fig.suptitle(f"{md} — Δ {met}: Reasoning On vs. Reasoning Off",
                 fontsize=_SUPTITLE_FONTSIZE, fontweight="bold", y=0.995)
    # Use fig.text directly — fig.supxlabel(y=...) gets quietly repositioned
    # by matplotlib's layout system, which is why prior attempts to lift the
    # x-axis label above the legend strip kept failing.
    fig.text(0.5, 0.190, "Number of Examples (k)",
             ha="center", va="center",
             fontsize=_SUPAXIS_FONTSIZE, fontweight="bold")
    fig.text(0.018, 0.588, f"Δ {met} (Reasoning On − Off)",
             ha="center", va="center", rotation="vertical",
             fontsize=_SUPAXIS_FONTSIZE, fontweight="bold")
    fig.tight_layout(rect=[0.030, 0.225, 1, 0.950])
    _, n_empty = _grid_empty_count(n)
    gap_scale = 1.0 if n_empty <= 1 else 2.5
    _place_figure_legend(fig, af, n, lh, ll, title="Method", gap_width_scale=gap_scale)
    fig.savefig(op, dpi=250, bbox_inches="tight"); plt.close(fig)


def save_aggregated_delta_superplot(dad: pd.DataFrame, met: str,
                                     mord: List[str], op: str) -> None:
    ensure_dir(os.path.dirname(op))
    ds = dad[dad["metric"] == met].copy()
    ds = _apply_k_plot_filter(ds)
    models = [m for m in mord if m in ds["model_key"].values]
    if not models:
        models = [m for m in mord if m in MODELS]
    n = len(models)
    if n == 0:
        fig, ax = plt.subplots(figsize=(9, 5)); ax.axis("off")
        ax.text(0.5, 0.5, "No data found", ha="center", va="center", fontsize=22)
        fig.tight_layout(); fig.savefig(op, dpi=250, bbox_inches="tight"); plt.close(fig)
        return
    nr, nc = _superplot_grid(n)
    fig, axes = plt.subplots(nr, nc, figsize=(_SUBPLOT_W * nc, _SUBPLOT_H * nr), squeeze=False)
    af = axes.flatten(); lh = ll = None
    for ax, mk in zip(af, models):
        md = MODELS.get(mk, mk)
        dm = ds[ds["model_key"] == mk].copy()
        ax.set_title(md, fontsize=_SUBPLOT_TITLE_FONTSIZE, fontweight="bold", pad=12)
        ax.set_xticks(K_LIST_PLOT)
        ax.tick_params(axis="both", labelsize=_TICK_FONTSIZE)
        if dm.empty:
            _polish_axes(ax, axhline_zero=True)
            ax.text(0.5, 0.5, "No data found", ha="center", va="center",
                    fontsize=22, style="italic", transform=ax.transAxes)
            continue
        sns.lineplot(
            data=dm.sort_values(["method_label", "k"], kind="stable"),
            x="k", y="delta", hue="method_label", style="method_label",
            markers=True, dashes=False, linewidth=3.0, markersize=12,
            ax=ax, legend=True,
        )
        _polish_axes(ax, axhline_zero=True)
        ax.set_xlabel("")
        ax.set_ylabel("")
        lh, ll = _collect_legend(ax, lh, ll)
        if ax.legend_:
            ax.legend_.remove()
    for ax in af[n:]:
        ax.axis("off")
    fig.suptitle(
        f"Aggregated Δ {met}: Reasoning On vs. Off (Averaged Across All Language Pairs)",
        fontsize=_SUPTITLE_FONTSIZE, fontweight="bold", y=0.995,
    )
    # Use fig.text directly — fig.supxlabel(y=...) gets quietly repositioned
    # by matplotlib's layout system, which is why prior attempts to lift the
    # x-axis label above the legend strip kept failing.
    fig.text(0.5, 0.190, "Number of Examples (k)",
             ha="center", va="center",
             fontsize=_SUPAXIS_FONTSIZE, fontweight="bold")
    fig.text(0.018, 0.588, f"Δ {met}",
             ha="center", va="center", rotation="vertical",
             fontsize=_SUPAXIS_FONTSIZE, fontweight="bold")
    fig.tight_layout(rect=[0.030, 0.225, 1, 0.950])
    # 4 method entries → 2 cols × 2 rows so the legend matches the height
    # of the aggregated-scores legend below.
    _place_figure_legend(fig, af, n, lh, ll, title="Method", max_ncol=2)
    fig.savefig(op, dpi=250, bbox_inches="tight"); plt.close(fig)


def save_aggregated_scores_superplot(da: pd.DataFrame, met: str,
                                       mord: List[str], op: str) -> None:
    ensure_dir(os.path.dirname(op))
    ds = da[da["metric"] == met].copy()
    ds = _apply_k_plot_filter(ds)
    ds["line_label"] = ds["method_label"] + " – Reasoning " + ds["reasoning_state"]
    models = [m for m in mord if m in ds["model_key"].values]
    if not models:
        models = [m for m in mord if m in MODELS]
    n = len(models)
    if n == 0:
        fig, ax = plt.subplots(figsize=(9, 5)); ax.axis("off")
        ax.text(0.5, 0.5, "No data found", ha="center", va="center", fontsize=22)
        fig.tight_layout(); fig.savefig(op, dpi=250, bbox_inches="tight"); plt.close(fig)
        return
    nr, nc = _superplot_grid(n)
    fig, axes = plt.subplots(nr, nc, figsize=(_SUBPLOT_W * nc, _SUBPLOT_H * nr), squeeze=False)
    af = axes.flatten(); lh = ll = None
    rdash = {"On": "", "Off": (4, 2)}
    for ax, mk in zip(af, models):
        md = MODELS.get(mk, mk)
        dm = ds[ds["model_key"] == mk].copy()
        ax.set_title(md, fontsize=_SUBPLOT_TITLE_FONTSIZE, fontweight="bold", pad=12)
        ax.set_xticks(K_LIST_PLOT)
        ax.tick_params(axis="both", labelsize=_TICK_FONTSIZE)
        if dm.empty:
            _polish_axes(ax)
            ax.text(0.5, 0.5, "No data found", ha="center", va="center",
                    fontsize=22, style="italic", transform=ax.transAxes)
            continue
        sns.lineplot(
            data=dm.sort_values(["method_label", "reasoning_state", "k"], kind="stable"),
            x="k", y="mean_score",
            hue="method_label", style="reasoning_state",
            style_order=["On", "Off"],
            markers=True, dashes=rdash,
            linewidth=3.0, markersize=12, ax=ax, legend=True,
        )
        _polish_axes(ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        lh, ll = _collect_legend(
            ax, lh, ll, ek={"method_label", "reasoning_state"},
        )
        if ax.legend_:
            ax.legend_.remove()
    for ax in af[n:]:
        ax.axis("off")
    fig.suptitle(
        f"Aggregated {met} Across All Language Pairs (Reasoning On vs. Off by Method)",
        fontsize=_SUPTITLE_FONTSIZE, fontweight="bold", y=0.995,
    )
    # Use fig.text directly — fig.supxlabel(y=...) gets quietly repositioned
    # by matplotlib's layout system, which is why prior attempts to lift the
    # x-axis label above the legend strip kept failing.
    fig.text(0.5, 0.190, "Number of Examples (k)",
             ha="center", va="center",
             fontsize=_SUPAXIS_FONTSIZE, fontweight="bold")
    fig.text(0.018, 0.588, met,
             ha="center", va="center", rotation="vertical",
             fontsize=_SUPAXIS_FONTSIZE, fontweight="bold")
    fig.tight_layout(rect=[0.030, 0.225, 1, 0.950])
    # 6 entries (4 methods + 2 reasoning states) → 3 cols × 2 rows so the
    # legend matches the height (and now the width) of the aggregated-
    # deltas legend above — both default to bottom_legend_w=0.36.
    _place_figure_legend(fig, af, n, lh, ll, title="Method – Reasoning State", max_ncol=3)
    fig.savefig(op, dpi=250, bbox_inches="tight"); plt.close(fig)


def save_cross_model_comparison_superplot(
    df: pd.DataFrame,
    on_run_key: str,
    off_run_key: str,
    method_label: str,
    met: str,
    dord: List[str],
    op: str,
    st: float,
    restrict_to: Optional[List[str]] = None,
    family_label: Optional[str] = None,
) -> None:
    ensure_dir(os.path.dirname(op))
    sk, lk = _small_and_large_model_keys(st, restrict_to)
    # Defensive: never index MODELS with a filtered-out key.
    sk = [mk for mk in sk if mk in MODELS]
    lk = [mk for mk in lk if mk in MODELS]
    parts: List[pd.DataFrame] = []
    for mk in sk:
        s = df[(df["model_key"] == mk) & (df["run_key"] == on_run_key)
               & (df["metric"] == met) & (~df["score"].isna())].copy()
        s["line_label"] = f"{MODELS[mk]} – Reasoning On"
        parts.append(s)
    for mk in lk:
        s = df[(df["model_key"] == mk) & (df["run_key"] == off_run_key)
               & (df["metric"] == met) & (~df["score"].isna())].copy()
        s["line_label"] = f"{MODELS[mk]} – Reasoning Off"
        parts.append(s)
    dp = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    dp = _apply_k_plot_filter(dp)
    fs = f" ({family_label})" if family_label else ""
    avail = [d for d in dord if not dp.empty and d in set(dp["direction"].tolist())]
    if not avail and not dp.empty:
        avail = sorted(dp["direction"].dropna().unique().tolist())
    n = len(avail)
    if n == 0:
        fig, ax = plt.subplots(figsize=(9, 5)); ax.axis("off")
        ax.text(0.5, 0.5, "No data found", ha="center", va="center", fontsize=22)
        fig.tight_layout(); fig.savefig(op, dpi=250, bbox_inches="tight"); plt.close(fig)
        return
    nr, nc = _superplot_grid(n)
    fig, axes = plt.subplots(nr, nc, figsize=(_SUBPLOT_W * nc, _SUBPLOT_H * nr), squeeze=False)
    af = axes.flatten(); lh = ll = None
    for ax, d in zip(af, avail):
        dd = direction_display_from_folder(d)
        df2 = dp[dp["direction"] == d].copy()
        ax.set_title(dd, fontsize=_SUBPLOT_TITLE_FONTSIZE, fontweight="bold", pad=12)
        ax.set_xticks(K_LIST_PLOT)
        ax.tick_params(axis="both", labelsize=_TICK_FONTSIZE)
        if df2.empty:
            _polish_axes(ax)
            ax.text(0.5, 0.5, "No data found", ha="center", va="center",
                    fontsize=22, style="italic", transform=ax.transAxes)
            continue
        sns.lineplot(
            data=df2.sort_values(["line_label", "k"], kind="stable"),
            x="k", y="score",
            hue="line_label", style="line_label",
            markers=True, dashes=False,
            linewidth=3.0, markersize=12, ax=ax, legend=True,
        )
        _polish_axes(ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        lh, ll = _collect_legend(ax, lh, ll)
        if ax.legend_:
            ax.legend_.remove()
    for ax in af[n:]:
        ax.axis("off")
    fig.suptitle(
        f"{method_label} — {met}: Small Models (Reasoning On) vs. "
        f"Large Models (Reasoning Off){fs}",
        fontsize=_SUPTITLE_FONTSIZE, fontweight="bold", y=0.995,
    )
    # Use fig.text directly — fig.supxlabel(y=...) gets quietly repositioned
    # by matplotlib's layout system, which is why prior attempts to lift the
    # x-axis label above the legend strip kept failing.
    fig.text(0.5, 0.190, "Number of Examples (k)",
             ha="center", va="center",
             fontsize=_SUPAXIS_FONTSIZE, fontweight="bold")
    fig.text(0.018, 0.588, met,
             ha="center", va="center", rotation="vertical",
             fontsize=_SUPAXIS_FONTSIZE, fontweight="bold")
    fig.tight_layout(rect=[0.030, 0.225, 1, 0.950])
    _, n_empty = _grid_empty_count(n)
    gap_scale = 1.0 if n_empty <= 1 else 1.6
    _place_figure_legend(
        fig, af, n, lh, ll,
        title=f"Model – Reasoning State  (threshold: {st}B)",
        gap_width_scale=gap_scale,
    )
    fig.savefig(op, dpi=250, bbox_inches="tight"); plt.close(fig)


def generate_plot_suite(df_master: pd.DataFrame) -> None:
    """
    Purpose: Generate the full eval_pipeline-style plot suite for chrF++ (sentence-mean).
    Inputs: Master per-sentence DataFrame.
    Outputs: PNG files under PLOTS_ROOT/ in the same tree shape eval_pipeline produces.
    """
    if not GENERATE_PLOTS:
        print("\n[Stage 7] GENERATE_PLOTS=False — skipping plot suite.")
        return

    print(f"\n[Stage 7] Generating eval_pipeline-style plots under {PLOTS_ROOT}/ "
          f"(INCLUDE_K0_IN_PLOTS={INCLUDE_K0_IN_PLOTS}).")
    ensure_dir(PLOTS_ROOT)

    # Build the eval_pipeline-shaped DataFrames from our per-sentence master.
    df_long = build_df_long_chrfpp(df_master)
    df_long.to_csv(os.path.join(PLOTS_ROOT, "raw_scores_long_chrfpp_sentence_mean.csv"),
                   index=False)
    df_delta = compute_delta_df_plot(df_long)
    df_agg = compute_aggregated_scores_plot(df_long)
    df_agg_delta = compute_aggregated_deltas_plot(df_agg)

    dord = [direction_folder_name(s, t) for s in SRC_LANGS for t in TGT_LANGS]
    met = METRIC_NAME_FOR_PLOTS
    ms = slugify(met)

    # Per-model × per-direction line plots & tables.
    for mk in MODELS:
        mr = os.path.join(PLOTS_ROOT, slugify(mk))
        ensure_dir(mr)
        for sl in SRC_LANGS:
            for tl in TGT_LANGS:
                dr = direction_folder_name(sl, tl)
                fo = os.path.join(mr, dr)
                ensure_dir(fo)
                save_metric_plot(df_long, mk, dr, met,
                                  os.path.join(fo, f"plot_{ms}.png"))
                save_metric_table(df_long, mk, dr, met,
                                   os.path.join(fo, f"table_{ms}.png"))
        # Per-model superplots.
        save_model_metric_superplot(
            df_long, mk, met, dord,
            os.path.join(mr, f"superplot_{ms}.png"),
        )
        if not df_delta.empty:
            save_delta_superplot(
                df_delta, mk, met, dord,
                os.path.join(mr, f"delta_superplot_{ms}.png"),
            )

    # Cross-model comparison superplots (small ON vs large OFF) per method.
    cr = os.path.join(PLOTS_ROOT, "cross_model_comparison")
    ensure_dir(cr)
    for method_key, (on_run_key, off_run_key, method_label) in METHOD_KEY_TO_RUN_KEYS.items():
        md2 = os.path.join(cr, slugify(method_label))
        ensure_dir(md2)
        save_cross_model_comparison_superplot(
            df_long, on_run_key, off_run_key, method_label, met,
            dord, os.path.join(md2, f"superplot_{ms}.png"),
            MODEL_SIZE_THRESHOLD,
        )
        for fn, fk in MODEL_FAMILIES.items():
            if not fk:  # family fully filtered out (RTRACE_EVAL_MODELS)
                continue
            fd = os.path.join(md2, slugify(fn))
            ensure_dir(fd)
            save_cross_model_comparison_superplot(
                df_long, on_run_key, off_run_key, method_label, met,
                dord, os.path.join(fd, f"superplot_{ms}.png"),
                MODEL_SIZE_THRESHOLD, restrict_to=fk, family_label=fn,
            )

    # Aggregated cross-language superplots (one subplot per model).
    ar = os.path.join(PLOTS_ROOT, "aggregated_cross_language")
    ensure_dir(ar)
    if not df_agg_delta.empty:
        save_aggregated_delta_superplot(
            df_agg_delta, met, MODEL_ORDER,
            os.path.join(ar, f"aggregated_delta_{ms}.png"),
        )
    if not df_agg.empty:
        save_aggregated_scores_superplot(
            df_agg, met, MODEL_ORDER,
            os.path.join(ar, f"aggregated_scores_{ms}.png"),
        )

    print(f"[Stage 7] Plot suite written under {PLOTS_ROOT}/")


# ─────────────────────────────────────────────────────────────────────────────
# Console headline summary
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_corr_row(label: str, row: pd.Series) -> str:
    return (
        f"  {label:<32s} n={int(row['n']):6d}  "
        f"Pearson r={row['pearson_r']:+.4f} (p={row['pearson_p']:.3g})  "
        f"Spearman ρ={row['spearman_r']:+.4f} (p={row['spearman_p']:.3g})"
    )


def _print_anova_family_block(family: str, df_anova: pd.DataFrame) -> None:
    """Print the per-language focal-factor ANOVA summary for one model family."""
    if df_anova is None or df_anova.empty:
        return
    print(f"\n[Stage 4 / {family}] Per-language ANOVA — focal continuous "
          "factors (reasoning_tokens, mean_recall) in reasoning-ON sentence chrF++:")
    focal_terms = ["reasoning_tokens", "mean_recall"]
    sub = df_anova.loc[df_anova["term"].isin(focal_terms)].copy()
    if sub.empty:
        return
    wide = sub.pivot_table(
        index="tgt_lang", columns="term",
        values=["F", "p_value", "partial_eta_sq"],
        aggfunc="first",
    )
    if ("partial_eta_sq", "reasoning_tokens") in wide.columns:
        wide = wide.sort_values(
            ("partial_eta_sq", "reasoning_tokens"), ascending=False,
        )
    print(
        f"  {'tgt_lang':<14s}"
        f"{'F(rt)':>10s} {'p(rt)':>10s} {'η²p(rt)':>10s}  "
        f"{'F(mr)':>10s} {'p(mr)':>10s} {'η²p(mr)':>10s}"
    )
    for tgt_lang, row in wide.iterrows():
        disp = LANG_DISPLAY.get(tgt_lang, tgt_lang)
        def _get(metric: str, term: str) -> float:
            return float(row.get((metric, term), float("nan")))
        print(
            f"  {disp:<14s}"
            f"{_get('F','reasoning_tokens'):>10.3f} "
            f"{_get('p_value','reasoning_tokens'):>10.3g} "
            f"{_get('partial_eta_sq','reasoning_tokens'):>10.4f}  "
            f"{_get('F','mean_recall'):>10.3f} "
            f"{_get('p_value','mean_recall'):>10.3g} "
            f"{_get('partial_eta_sq','mean_recall'):>10.4f}"
        )


def _print_lmm_family_block(family: str, df_lmm: pd.DataFrame) -> None:
    """Print the per-language focal-coefficient LMM summary for one family."""
    if df_lmm is None or df_lmm.empty:
        return
    print(f"\n[Stage 4b / {family}] Per-language LMM — focal continuous "
          "coefficients (reasoning_tokens, mean_recall) in reasoning-ON "
          "sentence chrF++ (random intercept on source_id):")
    focal_terms = ["reasoning_tokens", "mean_recall"]
    sub = df_lmm.loc[df_lmm["term"].isin(focal_terms)].copy()
    if sub.empty:
        return
    wide = sub.pivot_table(
        index="tgt_lang", columns="term",
        values=["coef", "z", "p_value"],
        aggfunc="first",
    )
    if ("coef", "reasoning_tokens") in wide.columns:
        wide = wide.sort_values(
            ("coef", "reasoning_tokens"), ascending=False,
        )
    print(
        f"  {'tgt_lang':<14s}"
        f"{'β(rt)':>12s} {'z(rt)':>10s} {'p(rt)':>10s}  "
        f"{'β(mr)':>12s} {'z(mr)':>10s} {'p(mr)':>10s}"
    )
    for tgt_lang, row in wide.iterrows():
        disp = LANG_DISPLAY.get(tgt_lang, tgt_lang)
        def _get(metric: str, term: str) -> float:
            return float(row.get((metric, term), float("nan")))
        print(
            f"  {disp:<14s}"
            f"{_get('coef','reasoning_tokens'):>+12.4e} "
            f"{_get('z','reasoning_tokens'):>10.3f} "
            f"{_get('p_value','reasoning_tokens'):>10.3g}  "
            f"{_get('coef','mean_recall'):>+12.4e} "
            f"{_get('z','mean_recall'):>10.3f} "
            f"{_get('p_value','mean_recall'):>10.3g}"
        )


def _print_lmm_joint_family_block(family: str, df_joint: pd.DataFrame) -> None:
    """Per-language joint Wald χ² ranking — the term with the largest χ² is
    the dominant predictor for that (family, language) cell.

    Unlike the per-coefficient z² heuristic, this is the proper joint test
    on the term's full coefficient block, accounting for the off-diagonals
    in the fixed-effect covariance matrix.
    """
    if df_joint is None or df_joint.empty:
        return
    print(f"\n[Stage 4b / {family}] Per-language joint Wald χ² — dominant "
          "term per (family, language):")
    print(
        f"  {'tgt_lang':<14s} {'top term':<22s} {'χ²':>10s} {'df':>4s} "
        f"{'p':>10s}   (runner-up: term, χ²)"
    )
    for tgt_lang, sub in df_joint.groupby("tgt_lang"):
        ordered = sub.sort_values("chi2", ascending=False)
        if ordered.empty:
            continue
        top = ordered.iloc[0]
        runner = ordered.iloc[1] if len(ordered) >= 2 else None
        disp = LANG_DISPLAY.get(tgt_lang, tgt_lang)
        runner_str = (
            f"({runner['term']}, χ²={float(runner['chi2']):.1f})"
            if runner is not None else "(—)"
        )
        print(
            f"  {disp:<14s} {top['term']:<22s} "
            f"{float(top['chi2']):>10.2f} {int(top['df']):>4d} "
            f"{float(top['p_value']):>10.3g}   {runner_str}"
        )


def print_headlines(
    df_corr: pd.DataFrame,
    df_anova_by_family: Dict[str, pd.DataFrame],
    df_lmm_by_family: Dict[str, pd.DataFrame],
    df_lmm_joint_by_family: Dict[str, pd.DataFrame],
    df_table3: pd.DataFrame,
    df_paired_bs: Optional[pd.DataFrame] = None,
) -> None:
    print("\n" + "=" * 78)
    print("HEADLINE RESULTS")
    print("=" * 78)

    # Stage 3 — chrfpp × mean_recall, overall + per method.
    headline_pair = "chrfpp × mean_recall"
    df_headline = df_corr.loc[df_corr["pair_name"] == headline_pair]
    if not df_headline.empty:
        print(f"\n{headline_pair} (per-sentence):")
        overall = df_headline.loc[df_headline["axis"] == "overall"]
        if not overall.empty:
            print(_fmt_corr_row("overall", overall.iloc[0]))
        method_rows = df_headline.loc[df_headline["axis"] == "method"]
        for _, row in method_rows.sort_values("group").iterrows():
            print(_fmt_corr_row(f"method={row['group']}", row))

    # Stage 4 — per-language ON-only ANOVA, run separately per family.
    #
    # The research question — "for a given language, which technique factors
    # explain sentence-level chrF++ within reasoning-ON runs?" — is answered
    # by reading the F and partial η² of `reasoning_tokens` and `mean_recall`
    # in each language's ANOVA table. Sort by reasoning_tokens η² so the
    # languages where "spending more thinking tokens" actually predicts
    # quality bubble to the top. Each family is shown separately to avoid
    # pooling Mistral and Qwen reasoning behaviour into a single average.
    #
    # IMPORTANT — η²p (partial η²) is the cross-factor comparison metric.
    # SS values printed in the CSV scale with each predictor's variance and
    # are NOT comparable across categorical/continuous terms or across
    # predictors on different scales (reasoning_tokens 0–30k vs mean_recall
    # 0–100). Use F to test detectability and η²p to compare importance.
    print("\n(η²p is the cross-factor comparison metric; F tests "
          "detectability; SS is scale-dependent and not directly comparable.)")
    for family in MODEL_FAMILIES:
        _print_anova_family_block(family, df_anova_by_family.get(family))

    # Stage 4b — per-language LMM (random intercept on source_id), per family.
    # The LMM is the right primary read because ON rows are NOT independent:
    # the same English sentence appears in every (model, method, k) cell, so
    # the residual within source_id is correlated. The fixed-effect formula
    # matches the ANOVA exactly, so coefficient signs/magnitudes for
    # reasoning_tokens / mean_recall are directly comparable across the two.
    print("\n(LMM β is the per-unit chrF++ change. With ON rows correlated "
          "through source_id, the LMM's z and p are the correct inference; "
          "the ANOVA block above is the conventional SS-decomposition view.)")
    for family in MODEL_FAMILIES:
        _print_lmm_family_block(family, df_lmm_by_family.get(family))

    # Stage 4b joint tests — "which term is the dominant predictor in this
    # (family, language) cell?" The joint Wald χ² is statsmodels' proper
    # linear-restriction test on the full coefficient block of a term,
    # accounting for the within-term covariance matrix. Larger χ² → more
    # variance attributable to that term.
    print("\n(Joint Wald χ² ranks terms by their combined coefficient block "
          "while accounting for the covariance matrix — the inference-correct "
          "version of the earlier Σz² approximation.)")
    for family in MODEL_FAMILIES:
        _print_lmm_joint_family_block(family, df_lmm_joint_by_family.get(family))

    # Stage 6 — Table 3.
    print("\nTable 3 — per-language reasoning-token usage vs Δ(chrF++ ON − OFF):")
    body = df_table3.loc[df_table3["tgt_lang"] != "_SUMMARY_"]
    print(
        f"  {'rank':<5s} {'tgt_lang':<14s} {'tokens_on':>10s} "
        f"{'chrf_on':>8s} {'chrf_off':>9s} {'Δ':>8s}"
    )
    for _, row in body.iterrows():
        print(
            f"  {int(row['rank_by_tokens']):<5d} "
            f"{row['tgt_lang_display']:<14s} "
            f"{float(row['mean_reasoning_tokens_on']):>10.1f} "
            f"{float(row['chrfpp_on']):>8.2f} "
            f"{float(row['chrfpp_off']):>9.2f} "
            f"{float(row['delta_on_minus_off']):>+8.2f}"
        )
    summary = df_table3.loc[df_table3["tgt_lang"] == "_SUMMARY_"]
    if not summary.empty:
        print(f"  {summary.iloc[0]['tgt_lang_display']}")

    # Stage 6b — paired bootstrap aggregated per language (averaged across
    # models / methods / k for the headline). Per-(model, method, direction, k)
    # rows live in the long CSV.
    if df_paired_bs is not None and not df_paired_bs.empty:
        print("\nPaired bootstrap (per-sentence chrF++ ON vs OFF), aggregated per language:")
        per_lang = (
            df_paired_bs.groupby("tgt_lang", as_index=False)
            .agg(
                mean_delta=("delta", "mean"),
                n_cells=("delta", "size"),
                n_sig_005=("significant_p_0_05", "sum"),
                n_sig_001=("significant_p_0_01", "sum"),
                n_better_than_off=("better_than_off", "sum"),
            )
            .sort_values("mean_delta", ascending=False)
        )
        for _, row in per_lang.iterrows():
            disp = LANG_DISPLAY.get(row["tgt_lang"], row["tgt_lang"])
            print(
                f"  {disp:<14s} mean Δ={float(row['mean_delta']):+6.2f}  "
                f"n_cells={int(row['n_cells']):3d}  "
                f"sig@.05={int(row['n_sig_005']):3d}  "
                f"sig@.01={int(row['n_sig_001']):3d}  "
                f"better_than_off={int(row['n_better_than_off']):3d}/{int(row['n_cells']):3d}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Verification — sanity check Stage 1 vs corpus chrF++ for a handful of files
# ─────────────────────────────────────────────────────────────────────────────


def sanity_check_stage1_vs_corpus(df_chrfpp: pd.DataFrame, n_files: int = 5) -> None:
    """
    Purpose: For a few sampled files, print mean(sentence chrF++) vs corpus chrF++. Both should be close but not identical.
    Inputs: Stage-1 DataFrame and the number of files to sample.
    Outputs: Console table.
    """
    if df_chrfpp.empty:
        return
    print(f"\n[Sanity] mean(sentence chrF++) vs corpus chrF++ on {n_files} sampled files:")
    sampled = df_chrfpp.groupby(
        ["model_key", "run_key", "direction", "k"], sort=False
    ).head(1).head(n_files)
    cols_fmt = (
        f"  {'model':<18s} {'run_key':<28s} {'dir':<14s} {'k':>3s} "
        f"{'n':>4s} {'mean(sent)':>10s} {'corpus':>8s} {'|Δ|':>5s}"
    )
    print(cols_fmt)

    for _, head_row in sampled.iterrows():
        model_key = head_row["model_key"]
        run_key = head_row["run_key"]
        direction = head_row["direction"]
        k = int(head_row["k"])
        sub = df_chrfpp.loc[
            (df_chrfpp["model_key"] == model_key)
            & (df_chrfpp["run_key"] == run_key)
            & (df_chrfpp["direction"] == direction)
            & (df_chrfpp["k"] == k)
        ]
        if sub.empty:
            continue
        translation_path = sub.iloc[0]["translation_path"]
        if not isinstance(translation_path, str) or not os.path.exists(translation_path):
            continue

        try:
            predictions = read_jsonl_translations(translation_path)
        except Exception:
            continue
        tgt_lang = sub.iloc[0]["tgt_lang"]
        refs = get_flores_devtest_refs(tgt_lang)
        n = min(len(predictions), len(refs), _apply_limit(len(predictions), EVAL_FIRST_M))
        predictions = predictions[:n]
        refs_capped = refs[:n]

        mean_sentence = float(sub["chrfpp_sentence"].mean())
        corpus_score = float(_CHRFPP.corpus_score(predictions, [refs_capped]).score)
        print(
            f"  {model_key:<18s} {run_key:<28s} {direction:<14s} {k:>3d} "
            f"{n:>4d} {mean_sentence:>10.2f} {corpus_score:>8.2f} "
            f"{abs(mean_sentence - corpus_score):>5.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ensure_dir(OUTPUT_ROOT)
    print(f"[chrfpp_per_sentence_analysis] OUTPUT_ROOT = {OUTPUT_ROOT}")
    print(f"  LOAD_FROM_CSV = {LOAD_FROM_CSV}")

    df_chrfpp = load_or_compute_per_sentence_chrfpp()

    # Diagnostic — show (run_key, k) row counts so it's immediately clear
    # whether edit_dist k=0 etc. made it into the Stage-1 cache via the
    # random fallback. If a (run_key, k) shows 0 rows here, the fallback
    # didn't fire (or the random run's k=0 file isn't on disk) and the
    # plots / bootstrap will be missing that point.
    print("\n[Stage 1 coverage] rows per (run_key, k):")
    print(f"  {'run_key':<32s}" + " ".join(f"{k:>6}" for k in sorted(set(K_LIST))))
    counts = (
        df_chrfpp.groupby(["run_key", "k"]).size()
        .unstack(fill_value=0).reindex(columns=sorted(set(K_LIST)), fill_value=0)
    )
    for rk in sorted(counts.index):
        row = counts.loc[rk]
        print(f"  {rk:<32s}" + " ".join(f"{int(row[k]):>6d}" for k in sorted(set(K_LIST))))

    sanity_check_stage1_vs_corpus(df_chrfpp, n_files=5)

    df_master = load_or_build_master(df_chrfpp)
    df_corr = load_or_compute_correlations(df_master)
    df_anova_by_family = load_or_fit_anova(df_master)
    df_lmm_by_family, df_lmm_joint_by_family = load_or_fit_lmm(df_master)
    df_table3 = load_or_compute_table3(df_master)
    df_paired_bs = load_or_compute_paired_bs(df_master)
    generate_plot_suite(df_master)

    print_headlines(
        df_corr,
        df_anova_by_family,
        df_lmm_by_family,
        df_lmm_joint_by_family,
        df_table3,
        df_paired_bs,
    )
    print(f"\nAll chrF++ per-sentence artefacts saved to: {OUTPUT_ROOT}/")
    print("Done.")


if __name__ == "__main__":
    main()