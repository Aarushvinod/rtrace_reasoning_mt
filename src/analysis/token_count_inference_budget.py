"""
token_count_inference_budget.py
────────────────────────────────
Tokenize the model's outputs (reasoning trace + response) for every translation
file produced by the eval pipeline, using **the exact tokenizer each model
uses at inference time**, and write one CSV row per sentence so per-sample
inference budget can be analyzed in pandas.

Then — using the chrF++ raw scores from `eval_pipeline.py`'s
`raw_scores_long.csv` — compute three single Spearman correlations between
per-cell mean tokens and translation quality, across every
(model × method × language × k) cell in the experimental grid:

  1. response_tokens (Reasoning ON)  ↔  chrF++
     "When reasoning is on, do shorter or longer final translations
      correlate with higher quality?"
  2. response_tokens (Reasoning OFF) ↔  chrF++
     "When reasoning is off, do shorter or longer translations correlate
      with higher quality?"
  3. reasoning_tokens (Reasoning ON) ↔  chrF++
     "Does the length of the reasoning trace itself correlate with the
      quality of the final translation?"

Each correlation is one ρ + p-value across all configurations that satisfy
the reasoning-state filter. A negative ρ means "fewer tokens → higher quality";
positive ρ means "more tokens → higher quality."

Per-model tokenizer (verified against each repo in May 2026):

  ministral_8b     → mistralai/Ministral-3-8B-Reasoning-2512   (Tekken via mistral-common)
  ministral_14b    → mistralai/Ministral-3-14B-Reasoning-2512  (Tekken via mistral-common)
  magistral_small  → mistralai/Magistral-Small-2509            (Tekken via mistral-common)
  qwen3_8b         → Qwen/Qwen3-8B                             (Qwen2TokenizerFast via HF)
  qwen3_14b        → Qwen/Qwen3-14B                            (Qwen2TokenizerFast via HF)
  qwen3_32b        → Qwen/Qwen3-32B                            (Qwen2TokenizerFast via HF)

Behaviour driven by two config flags at the top of this file:
  • LOAD_FROM_CSV — smart caching. Methods already present in the per-sentence
                    CSV are not recomputed; only new methods (e.g. edit_dist
                    when first added) are tokenized and appended.
  • INCLUDE_K0   — when False, k=0 is excluded from token counting AND from
                   the correlation analysis.

Required deps (run in a separate Colab cell once):
  !pip install -q --upgrade "transformers>=4.45" "mistral-common>=1.8.6" pandas huggingface_hub scipy
"""

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.common.dataset_registry import get_dataset, filter_models


# ─────────────────────────────────────────────────────────────────────────────
# Configuration (mirrors prior scripts)
# ─────────────────────────────────────────────────────────────────────────────

LOAD_FROM_CSV: bool = True

# Dataset arm (RTRACE_DATASET: flores | wmt24pp). Every root, language list,
# and k value below derives from the spec, so this script is dataset-agnostic:
# any dataset whose loader speaks the {"dev","devtest"} contract plugs in via
# src/common/dataset_registry.py.
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

MODEL_TOKENIZER_ID: Dict[str, str] = {
    "ministral_8b":    "mistralai/Ministral-3-8B-Reasoning-2512",
    "ministral_14b":   "mistralai/Ministral-3-14B-Reasoning-2512",
    "magistral_small": "mistralai/Magistral-Small-2509",
    "qwen3_8b":        "Qwen/Qwen3-8B",
    "qwen3_14b":       "Qwen/Qwen3-14B",
    "qwen3_32b":       "Qwen/Qwen3-32B",
}

MODEL_TOKENIZER_FAMILY: Dict[str, str] = {
    "ministral_8b":    "mistral",
    "ministral_14b":   "mistral",
    "magistral_small": "mistral",
    "qwen3_8b":        "qwen",
    "qwen3_14b":       "qwen",
    "qwen3_32b":       "qwen",
}

MODEL_FAMILY_ROOT: Dict[str, str] = {
    "ministral_8b":    "Mistral",
    "ministral_14b":   "Mistral",
    "magistral_small": "Mistral",
    "qwen3_8b":        "Qwen",
    "qwen3_14b":       "Qwen",
    "qwen3_32b":       "Qwen",
}


def _model_root_dir_map(mistral_root: str, qwen_root: str) -> Dict[str, str]:
    return {
        "ministral_8b": mistral_root,
        "ministral_14b": mistral_root,
        "magistral_small": mistral_root,
        "qwen3_8b": qwen_root,
        "qwen3_14b": qwen_root,
        "qwen3_32b": qwen_root,
    }


# (root-suffix, method label, filename pattern) per selection method — run
# keys/labels are built from these EXACTLY as the legacy literal lists did,
# so every cached CSV keyed on run_key/method_label stays valid.
_METHODS: List[Tuple[str, str, str]] = [
    ("",           "RRF",           "k{K}_rrf_template11.jsonl"),
    ("_random",    "Random",        "k{K}_random_pool_template11.jsonl"),
    ("_sentinel",  "Sentinel",      "k{K}_pool_sentinel_src_rerank_template11.jsonl"),
    ("_edit_dist", "Edit Distance", "k{K}_edit_dist_template11.jsonl"),
]


def _runs_for_state(state_title: str) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for suffix, method_label, filename_pattern in _METHODS:
        runs.append({
            "key": f"reasoning_{state_title.lower()}{suffix}",
            "label": f"{method_label} Reasoning {state_title}",
            "method_label": method_label,
            "reasoning_state": state_title,
            "suffix": suffix,
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
K_LIST: List[int] = list(DS.k_list)
EVAL_FIRST_M: Optional[int] = 100

INCLUDE_K0: bool = os.environ.get("RTRACE_INCLUDE_K0", "0") == "1"
K_LIST_EFFECTIVE: List[int] = K_LIST if INCLUDE_K0 else [k for k in K_LIST if k > 0]

LANG_DISPLAY: Dict[str, str] = dict(DS.lang_display)

OUTPUT_DIR = os.environ.get("RTRACE_TOKENS_DIR", DS.analysis_dir("eval_token_counts"))
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "per_sentence_token_counts.csv")

# ── Correlation analysis configuration ────────────────────────────────────────
CORRELATION_EVAL_CSV: str = os.environ.get("RTRACE_EVAL_CSV", DS.eval_scores_csv())
CORRELATION_METRIC: str = "chrF++"
CORRELATION_OUTPUT_CSV: str = os.path.join(OUTPUT_DIR, "token_quality_correlations.csv")

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

OUT_COLUMNS: List[str] = [
    "model_key",
    "model_display",
    "tokenizer_id",
    "tokenizer_family",
    "run_key",
    "run_display",
    "reasoning_state",
    "method_label",
    "src_lang",
    "tgt_lang",
    "direction",
    "k",
    "sentence_idx",
    "reasoning_tokens",
    "response_tokens",
    "response_path",
    "reasoning_path",
]


# ─────────────────────────────────────────────────────────────────────────────
# Path / fallback helpers
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


def direction_display_from_folder(folder: str) -> str:
    parts = folder.split("_to_")
    if len(parts) == 2:
        prefix_display = {c.split("_")[0].lower(): n for c, n in LANG_DISPLAY.items()}
        src = prefix_display.get(parts[0], parts[0].capitalize())
        tgt = prefix_display.get(parts[1], parts[1].capitalize())
        return f"{src} to {tgt}"
    return folder


def build_translation_path(
    base_dir: str,
    model_dirname: str,
    direction: str,
    k: int,
    filename_pattern: str,
) -> str:
    return os.path.join(base_dir, model_dirname, direction, filename_pattern.format(K=k))


def build_reasoning_root(model_key: str, suffix: str) -> str:
    # Layout is dataset-dependent (legacy flores: one "all" root per method;
    # wmt24pp: per-method per-state roots) — the registry resolves it. Traces
    # exist for reasoning-ON only, which is the registry's default state.
    family = MODEL_FAMILY_ROOT[model_key]
    return DS.reasoning_trace_root(family, suffix)


def build_reasoning_path(
    model_key: str,
    suffix: str,
    direction: str,
    k: int,
    translation_filename_pattern: str,
) -> str:
    reasoning_filename = translation_filename_pattern.replace(".jsonl", "_reasoning.jsonl")
    root = build_reasoning_root(model_key, suffix)
    return os.path.join(root, model_key, direction, reasoning_filename.format(K=k))


def _build_k0_fallback_map(
    generation_runs: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
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


def _resolve_root_dir(run: Dict[str, Any], model_key: str) -> Optional[str]:
    rd = run["root_dir"]
    if isinstance(rd, str):
        return rd
    if isinstance(rd, dict):
        return rd.get(model_key)
    return None


def print_file_structure(all_runs: List[Dict[str, Any]]) -> None:
    print("\n[File structure resolution & on-disk check]")
    print(f"  Expected per (run, model): {len(SRC_LANGS) * len(TGT_LANGS)} directions × {len(K_LIST_EFFECTIVE)} k-values "
          f"= {len(SRC_LANGS) * len(TGT_LANGS) * len(K_LIST_EFFECTIVE)} files  "
          f"(INCLUDE_K0={INCLUDE_K0}, K_LIST_EFFECTIVE={K_LIST_EFFECTIVE})\n")
    for run in all_runs:
        print(f"  {run['key']}  (label: '{run['label']}', pattern: {run['filename_pattern']})")
        for mk in MODELS:
            root = _resolve_root_dir(run, mk)
            if root is None:
                print(f"    • {mk:<16} → NOT CONFIGURED")
                continue
            dn = resolve_model_dirname(root, mk)
            if dn is None:
                root_exists = os.path.isdir(root)
                print(f"    • {mk:<16} → MODEL DIR NOT FOUND  (root exists: {root_exists}, root: {root})")
                continue
            n_found = 0
            n_expected = 0
            for sl in SRC_LANGS:
                for tl in TGT_LANGS:
                    dr = direction_folder_name(sl, tl)
                    for k in K_LIST_EFFECTIVE:
                        n_expected += 1
                        path = build_translation_path(root, dn, dr, k, run["filename_pattern"])
                        if os.path.exists(path):
                            n_found += 1
            marker = "✓" if n_found == n_expected else ("·" if n_found > 0 else "✗")
            print(f"    {marker} {mk:<16} → {root}/{dn}  ({n_found}/{n_expected} files)")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# JSONL readers
# ─────────────────────────────────────────────────────────────────────────────


def read_jsonl_translations(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line)["translation"])
    return out


def _extract_reasoning_text(obj: Dict[str, Any]) -> Optional[str]:
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


def read_jsonl_reasonings(path: str) -> List[Optional[str]]:
    out: List[Optional[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(_extract_reasoning_text(json.loads(line)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer cache (lazy, per model)
# ─────────────────────────────────────────────────────────────────────────────

_TOKENIZER_CACHE: Dict[str, Callable[[List[str]], List[int]]] = {}


def _make_mistral_batch(model_id: str) -> Callable[[List[str]], List[int]]:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    mt = MistralTokenizer.from_hf_hub(model_id)

    if hasattr(mt, "encode") and callable(getattr(mt, "encode")):
        encode = lambda s: mt.encode(s, bos=False, eos=False)
    else:
        bpe = mt.instruct_tokenizer.tokenizer
        encode = lambda s: bpe.encode(s, bos=False, eos=False)

    def _batch(strings: List[str]) -> List[int]:
        return [len(encode(s)) for s in strings]

    return _batch


def _make_qwen_batch(model_id: str) -> Callable[[List[str]], List[int]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    def _batch(strings: List[str]) -> List[int]:
        if not strings:
            return []
        result = tok(
            strings,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        return [len(ids) for ids in result]

    return _batch


def get_tokenizer_callable(model_key: str) -> Callable[[List[str]], List[int]]:
    if model_key not in _TOKENIZER_CACHE:
        family = MODEL_TOKENIZER_FAMILY[model_key]
        model_id = MODEL_TOKENIZER_ID[model_key]
        print(f"[tokenizer] loading {model_id} ({family}) for {model_key}")
        if family == "mistral":
            _TOKENIZER_CACHE[model_key] = _make_mistral_batch(model_id)
        elif family == "qwen":
            _TOKENIZER_CACHE[model_key] = _make_qwen_batch(model_id)
        else:
            raise ValueError(f"Unknown tokenizer family for {model_key}: {family!r}")
    return _TOKENIZER_CACHE[model_key]


def count_tokens(
    strings: Sequence[Optional[str]],
    batch_callable: Callable[[List[str]], List[int]],
) -> List[Optional[int]]:
    out: List[Optional[int]] = [None] * len(strings)
    nonempty_indices: List[int] = []
    nonempty_strings: List[str] = []
    for i, s in enumerate(strings):
        if s is None:
            out[i] = None
        elif s == "":
            out[i] = 0
        else:
            nonempty_indices.append(i)
            nonempty_strings.append(s)
    if nonempty_strings:
        counts = batch_callable(nonempty_strings)
        for idx, count in zip(nonempty_indices, counts):
            out[idx] = int(count)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main loop  ── supports method-level caching ──
# ─────────────────────────────────────────────────────────────────────────────


def collect_token_counts(
    runs_on: List[Dict[str, Any]],
    runs_off: List[Dict[str, Any]],
    methods_to_compute: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    all_runs = runs_on + runs_off
    k0_fallback = _build_k0_fallback_map(all_runs)

    if methods_to_compute is None:
        compute_method_set = set(r["method_label"] for r in runs_on)
    else:
        compute_method_set = set(methods_to_compute)

    resolved: Dict[Tuple[str, str], Optional[Tuple[str, str]]] = {}
    for run in all_runs:
        for model_key in MODELS:
            root = _resolve_root_dir(run, model_key)
            if root is None:
                resolved[(run["key"], model_key)] = None
                continue
            dirname = resolve_model_dirname(root, model_key)
            resolved[(run["key"], model_key)] = None if dirname is None else (root, dirname)

    print("\n[Translation root resolution]")
    for run in all_runs:
        print(f"  {run['key']}:")
        for model_key in MODELS:
            entry = resolved.get((run["key"], model_key))
            if entry is None:
                root = _resolve_root_dir(run, model_key)
                print(f"    • {model_key} → NOT FOUND (root: {root or 'NO ROOT'})")
            else:
                print(f"    • {model_key} → {entry[0]}/{entry[1]}")

    if k0_fallback and INCLUDE_K0:
        print("\n[k=0 fallback map]")
        for src_key, fb_run in k0_fallback.items():
            print(f"  • {src_key} → {fb_run['key']}  (only when k=0 file missing)")

    rows: List[Dict[str, Any]] = []

    for src_lang in SRC_LANGS:
        for tgt_lang in TGT_LANGS:
            direction = direction_folder_name(src_lang, tgt_lang)
            direction_disp = direction_display_from_folder(direction)

            for model_key, model_display in MODELS.items():
                tokenizer_id = MODEL_TOKENIZER_ID[model_key]
                tokenizer_family = MODEL_TOKENIZER_FAMILY[model_key]

                runs_for_this_model = [
                    r for r in all_runs if r["method_label"] in compute_method_set
                ]
                if not runs_for_this_model:
                    continue
                print(f"\n[{model_display} | {direction_disp}]")

                for run in runs_for_this_model:
                    run_key = run["key"]
                    suffix = run["suffix"]
                    pattern = run["filename_pattern"]
                    state = run["reasoning_state"]
                    method_label = run["method_label"]
                    run_display = run["label"]

                    res = resolved.get((run_key, model_key))

                    for k in K_LIST_EFFECTIVE:
                        translation_path = ""
                        if res is not None:
                            root_dir, model_dirname = res
                            cand = build_translation_path(
                                root_dir, model_dirname, direction, k, pattern
                            )
                            if os.path.exists(cand):
                                translation_path = cand

                        if k == 0 and not translation_path and run_key in k0_fallback:
                            fb_run = k0_fallback[run_key]
                            fb_res = resolved.get((fb_run["key"], model_key))
                            if fb_res is not None:
                                fb_root, fb_dirname = fb_res
                                fb_cand = build_translation_path(
                                    fb_root, fb_dirname, direction, k,
                                    fb_run["filename_pattern"],
                                )
                                if os.path.exists(fb_cand):
                                    translation_path = fb_cand

                        if not translation_path:
                            continue

                        responses = read_jsonl_translations(translation_path)
                        ceiling = _apply_limit(len(responses), EVAL_FIRST_M)
                        responses = responses[:ceiling]

                        reasoning_path = ""
                        reasonings: List[Optional[str]] = [None] * len(responses)

                        if state == "On":
                            cand_reason = build_reasoning_path(
                                model_key, suffix, direction, k, pattern
                            )
                            if os.path.exists(cand_reason):
                                reasoning_path = cand_reason

                            if k == 0 and not reasoning_path and run_key in k0_fallback:
                                fb_run = k0_fallback[run_key]
                                fb_cand_reason = build_reasoning_path(
                                    model_key, fb_run["suffix"], direction, k,
                                    fb_run["filename_pattern"],
                                )
                                if os.path.exists(fb_cand_reason):
                                    reasoning_path = fb_cand_reason

                            if reasoning_path:
                                raw_reasonings = read_jsonl_reasonings(reasoning_path)
                                reasonings = (raw_reasonings + [None] * len(responses))[
                                    : len(responses)
                                ]

                        batch = get_tokenizer_callable(model_key)
                        response_counts = count_tokens(responses, batch)
                        if state == "On" and reasoning_path:
                            reasoning_counts = count_tokens(reasonings, batch)
                        else:
                            reasoning_counts = [None] * len(responses)

                        for i in range(len(responses)):
                            rows.append(
                                {
                                    "model_key": model_key,
                                    "model_display": model_display,
                                    "tokenizer_id": tokenizer_id,
                                    "tokenizer_family": tokenizer_family,
                                    "run_key": run_key,
                                    "run_display": run_display,
                                    "reasoning_state": state,
                                    "method_label": method_label,
                                    "src_lang": src_lang,
                                    "tgt_lang": tgt_lang,
                                    "direction": direction,
                                    "k": k,
                                    "sentence_idx": i,
                                    "reasoning_tokens": reasoning_counts[i],
                                    "response_tokens": (
                                        response_counts[i] if response_counts[i] is not None else 0
                                    ),
                                    "response_path": translation_path,
                                    "reasoning_path": reasoning_path,
                                }
                            )

                        resp_total = sum(c for c in response_counts if c is not None)
                        reason_total = sum(c for c in reasoning_counts if c is not None)
                        n_reason_present = sum(1 for c in reasoning_counts if c is not None)
                        print(
                            f"  {run_key:>26s} k={k:<2d}  "
                            f"sentences={len(responses):3d}  "
                            f"resp_tokens_sum={resp_total:6d}  "
                            f"reason_tokens_sum={reason_total:7d} "
                            f"({n_reason_present} sentences w/ reasoning)"
                        )

    return pd.DataFrame(rows, columns=OUT_COLUMNS)


# ─────────────────────────────────────────────────────────────────────────────
# Correlation analysis: three single ρ values
# ─────────────────────────────────────────────────────────────────────────────


def _spearman_or_nan(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    """Spearman ρ + p-value on aligned pandas Series, NaN-safe."""
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 3:
        return float("nan"), float("nan"), int(len(df))
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan"), float("nan"), int(len(df))
    r, p = spearmanr(a, b)
    return float(r), float(p), int(len(df))


def compute_token_quality_correlations(
    df_tokens: pd.DataFrame,
    eval_csv_path: str,
    metric_label: str,
) -> pd.DataFrame:
    """Compute exactly three Spearman correlations between per-cell mean tokens
    and translation quality. Each correlation is one ρ across all
    (model × method × language × k) cells in the configured grid (subject to
    the reasoning-state filter for that correlation).

    The three correlations:

      1. "response_tokens vs <metric> (Reasoning ON)"
         Unit: each (model, method, language, k) cell whose reasoning_state=On.
         One data point per cell, where:
           x = mean response-token count across that cell's ~100 sentences
           y = the cell's <metric> score (from raw_scores_long.csv)
         Answers: "When reasoning is on, does writing a shorter or longer
         final translation correlate with higher translation quality?"

      2. "response_tokens vs <metric> (Reasoning OFF)"
         Same shape, but only over cells whose reasoning_state=Off.
         Answers: "When reasoning is off, does writing a shorter or longer
         translation correlate with higher quality?"

      3. "reasoning_tokens vs <metric> (Reasoning ON)"
         Unit: each (model, method, language, k) cell whose reasoning_state=On.
         x = mean reasoning-trace-token count across that cell's sentences.
         y = the cell's <metric> score.
         Answers: "Does longer reasoning yield a better translation?"

    Sign reading:
      • negative ρ → fewer tokens correlate with higher quality
      • positive ρ → more tokens correlate with higher quality
    """
    if not os.path.exists(eval_csv_path):
        print(f"\n[Correlation] eval CSV not found at {eval_csv_path}; skipping.")
        return pd.DataFrame(columns=["correlation", "n_cells", "spearman_rho", "p_value"])

    df_eval = pd.read_csv(eval_csv_path)
    df_eval["k"] = df_eval["k"].astype(int)
    df_eval = df_eval[df_eval["metric"] == metric_label].copy()
    if df_eval.empty:
        print(f"\n[Correlation] No rows with metric == {metric_label!r} in {eval_csv_path}; skipping.")
        return pd.DataFrame(columns=["correlation", "n_cells", "spearman_rho", "p_value"])

    df_eval = df_eval[["model_key", "run_key", "direction", "k", "score"]].rename(
        columns={"score": "quality_score"}
    )

    # Per-cell aggregation: each (model_key, run_key, direction, k) cell has
    # ~100 sentence rows; collapse to one mean per cell. Pandas skips NaNs in
    # `.mean()` by default — so reasoning_tokens averages only over sentences
    # that actually have a reasoning trace.
    cell_keys = ["model_key", "run_key", "method_label", "reasoning_state",
                 "tgt_lang", "direction", "k"]
    cells = (
        df_tokens.groupby(cell_keys, dropna=False, observed=False)
        .agg(
            response_tokens_per_cell=("response_tokens", "mean"),
            reasoning_tokens_per_cell=("reasoning_tokens", "mean"),
        )
        .reset_index()
    )

    # Join the cell-level token aggregates against the quality scores. One
    # quality_score per cell.
    cells = cells.merge(df_eval, on=["model_key", "run_key", "direction", "k"], how="left")

    # Split by reasoning state.
    on_cells = cells[cells["reasoning_state"] == "On"]
    off_cells = cells[cells["reasoning_state"] == "Off"]

    # Three single correlations, exactly as requested.
    rho_resp_on, p_resp_on, n_resp_on = _spearman_or_nan(
        on_cells["response_tokens_per_cell"], on_cells["quality_score"]
    )
    rho_resp_off, p_resp_off, n_resp_off = _spearman_or_nan(
        off_cells["response_tokens_per_cell"], off_cells["quality_score"]
    )
    rho_reason_on, p_reason_on, n_reason_on = _spearman_or_nan(
        on_cells["reasoning_tokens_per_cell"], on_cells["quality_score"]
    )

    rows = [
        {
            "correlation": f"response_tokens vs {metric_label} (Reasoning ON)",
            "n_cells": n_resp_on,
            "spearman_rho": rho_resp_on,
            "p_value": p_resp_on,
        },
        {
            "correlation": f"response_tokens vs {metric_label} (Reasoning OFF)",
            "n_cells": n_resp_off,
            "spearman_rho": rho_resp_off,
            "p_value": p_resp_off,
        },
        {
            "correlation": f"reasoning_tokens vs {metric_label} (Reasoning ON)",
            "n_cells": n_reason_on,
            "spearman_rho": rho_reason_on,
            "p_value": p_reason_on,
        },
    ]

    # Side output: the per-cell joined table, in case you want to scatter-plot
    # or run alternative analyses later. Not used for the headline numbers.
    per_cell_path = os.path.join(OUTPUT_DIR, "per_cell_tokens_vs_quality.csv")
    cells.to_csv(per_cell_path, index=False)
    print(f"  per-cell joined table written to: {per_cell_path}")

    return pd.DataFrame(
        rows, columns=["correlation", "n_cells", "spearman_rho", "p_value"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    all_runs = GENERATION_RUNS_REASONING_ON + GENERATION_RUNS_REASONING_OFF

    print_file_structure(all_runs)

    configured_methods = {r["method_label"] for r in GENERATION_RUNS_REASONING_ON}

    if LOAD_FROM_CSV and os.path.exists(OUTPUT_CSV):
        print(f"\n[LOAD_FROM_CSV] Reading existing cache: {OUTPUT_CSV}")
        df_cached = pd.read_csv(OUTPUT_CSV)
        df_cached["k"] = df_cached["k"].astype(int)
        print(f"  Loaded {len(df_cached)} per-sentence rows.")

        cached_methods = (
            set(df_cached["method_label"].unique()) if not df_cached.empty else set()
        )
        missing_methods = configured_methods - cached_methods
        unexpected_methods = cached_methods - configured_methods

        if unexpected_methods:
            print(
                f"  [note] Cache contains {len(unexpected_methods)} method(s) not in current config; "
                f"they will pass through unchanged: {sorted(unexpected_methods)}"
            )

        if missing_methods:
            print(f"\n[Incremental compute] {len(missing_methods)} method(s) missing from cache:")
            for m in sorted(missing_methods):
                print(f"  • {m}")
            print()
            df_new = collect_token_counts(
                GENERATION_RUNS_REASONING_ON,
                GENERATION_RUNS_REASONING_OFF,
                methods_to_compute=missing_methods,
            )
            df = pd.concat([df_cached, df_new], ignore_index=True)
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"\n[Cache updated] {len(df_cached)} cached + {len(df_new)} new = {len(df)} rows.")
            print(f"  Written to: {OUTPUT_CSV}")
        else:
            print("  All configured methods already cached; skipping tokenisation step.")
            df = df_cached
    else:
        df = collect_token_counts(
            GENERATION_RUNS_REASONING_ON,
            GENERATION_RUNS_REASONING_OFF,
            methods_to_compute=None,
        )
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nWrote {len(df)} rows to {OUTPUT_CSV}")

    # Apply the INCLUDE_K0 toggle to the final outputs.
    df = df[df["k"].isin(K_LIST_EFFECTIVE)].reset_index(drop=True)

    # ── Three single-ρ correlations: per-cell tokens vs chrF++ ──
    print(f"\n[Correlation] computing 3 Spearman ρ values across all "
          f"(model × method × language × k) cells, vs {CORRELATION_METRIC} "
          f"from {CORRELATION_EVAL_CSV} …")
    df_corr = compute_token_quality_correlations(
        df,
        eval_csv_path=CORRELATION_EVAL_CSV,
        metric_label=CORRELATION_METRIC,
    )

    if not df_corr.empty:
        df_corr.to_csv(CORRELATION_OUTPUT_CSV, index=False)
        print(f"  correlation table written to: {CORRELATION_OUTPUT_CSV}\n")

        def _fmt(v: float) -> str:
            return "—" if pd.isna(v) else f"{v:+.4f}"

        print("[Correlation] headline values:")
        for _, row in df_corr.iterrows():
            print(
                f"  {row['correlation']:<58} "
                f"n={int(row['n_cells']):<5} "
                f"ρ={_fmt(row['spearman_rho'])}  p={_fmt(row['p_value'])}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()