"""
paired_bootstrap_chrfpp_significance.py
───────────────────────────────────────
Paired bootstrap significance testing for the reasoning ON vs OFF runs in the
attached evaluation pipeline.

What this script does:
  • Reuses the same model keys, run configs, direction naming, and k-values
    from eval_pipeline.py (including the optional k=0 anchor point).
  • Recomputes corpus chrF++ scores using the same SacreBLEU CHRF(word_order=2)
    setup used in the original script.
  • Uses SacreBLEU's paired bootstrap implementation instead of reimplementing
    the test from scratch.
  • Smart CSV cache: loads existing scores/significance, computes only newly
    configured methods (e.g. edit_dist when first added), merges, and persists.
  • Optional k=0 toggle: include or exclude the zero-shot baseline from every
    output (computation and CSVs).
  • Writes CSVs whose key columns align with the original raw score CSVs so the
    significance results can be merged into later table-generation scripts.
  • k=0 fallback: non-random runs reuse the random run's k=0 file (since k=0 is
    method-agnostic — zero in-context examples means the prompt is identical
    regardless of selection method). At k=0, all four methods therefore
    produce the same chrF++ score and the same paired bootstrap result.

Outputs:
  • raw_scores_long_chrfpp.csv
      Same key columns as DF_LONG_COLUMNS from the original script, but only for
      chrF++.
  • paired_bs_significance_long.csv
      One row per paired ON/OFF comparison at fixed model / direction / method / k.
  • raw_scores_long_chrfpp_with_significance.csv
      chrF++ raw-score rows annotated so they can be merged directly when marking
      significance later on.
  • Per-model / per-direction copies of the same CSVs.
  • valid_sentence_counts.json matching the original evaluation policy.

Requirements:
  pip install "sacrebleu>=2.0.0" pandas datasets openpyxl
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from datasets import load_dataset

try:
    from sacrebleu.metrics import CHRF
    from sacrebleu.significance import PairedTest
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This script requires sacrebleu>=2.0.0 because paired significance "
        "testing is implemented there. Install it with: pip install 'sacrebleu>=2.0.0'"
    ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Configuration copied / adapted from eval_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

LOAD_FROM_CSV: bool = True

MODELS: Dict[str, str] = {
    "ministral_8b": "Ministral 8B",
    "ministral_14b": "Ministral 14B",
    "magistral_small": "Magistral 24B",
    "qwen3_8b": "Qwen3 8B",
    "qwen3_14b": "Qwen3 14B",
    "qwen3_32b": "Qwen3 32B",
}


def _model_root_dir_map(mistral_root: str, qwen_root: str) -> Dict[str, str]:
    """Map each model key to the correct root directory."""
    return {
        "ministral_8b": mistral_root,
        "ministral_14b": mistral_root,
        "magistral_small": mistral_root,
        "qwen3_8b": qwen_root,
        "qwen3_14b": qwen_root,
        "qwen3_32b": qwen_root,
    }


GENERATION_RUNS_REASONING_ON: List[Dict[str, Any]] = [
    {
        "key": "reasoning_on",
        "label": "RRF Reasoning On",
        "root_dir": _model_root_dir_map(
            "drive/MyDrive/Mistral_All_Reasoning_On",
            "drive/MyDrive/Qwen_All_Reasoning_On",
        ),
        "filename_pattern": "k{K}_rrf_template11.jsonl",
    },
    {
        "key": "reasoning_on_random",
        "label": "Random Reasoning On",
        "root_dir": _model_root_dir_map(
            "drive/MyDrive/Mistral_All_Reasoning_On_random",
            "drive/MyDrive/Qwen_All_Reasoning_On_random",
        ),
        "filename_pattern": "k{K}_random_pool_template11.jsonl",
    },
    {
        "key": "reasoning_on_sentinel",
        "label": "Sentinel Reasoning On",
        "root_dir": _model_root_dir_map(
            "drive/MyDrive/Mistral_All_Reasoning_On_sentinel",
            "drive/MyDrive/Qwen_All_Reasoning_On_sentinel",
        ),
        "filename_pattern": "k{K}_pool_sentinel_src_rerank_template11.jsonl",
    },
    {
        "key": "reasoning_on_edit_dist",
        "label": "Edit Distance Reasoning On",
        "root_dir": _model_root_dir_map(
            "drive/MyDrive/Mistral_All_Reasoning_On_edit_dist",
            "drive/MyDrive/Qwen_All_Reasoning_On_edit_dist",
        ),
        "filename_pattern": "k{K}_edit_dist_template11.jsonl",
    },
]

GENERATION_RUNS_REASONING_OFF: List[Dict[str, Any]] = [
    {
        "key": "reasoning_off",
        "label": "RRF Reasoning Off",
        "root_dir": _model_root_dir_map(
            "drive/MyDrive/Mistral_All_Reasoning_Off",
            "drive/MyDrive/Qwen_All_Reasoning_Off",
        ),
        "filename_pattern": "k{K}_rrf_template11.jsonl",
    },
    {
        "key": "reasoning_off_random",
        "label": "Random Reasoning Off",
        "root_dir": _model_root_dir_map(
            "drive/MyDrive/Mistral_All_Reasoning_Off_random",
            "drive/MyDrive/Qwen_All_Reasoning_Off_random",
        ),
        "filename_pattern": "k{K}_random_pool_template11.jsonl",
    },
    {
        "key": "reasoning_off_sentinel",
        "label": "Sentinel Reasoning Off",
        "root_dir": _model_root_dir_map(
            "drive/MyDrive/Mistral_All_Reasoning_Off_sentinel",
            "drive/MyDrive/Qwen_All_Reasoning_Off_sentinel",
        ),
        "filename_pattern": "k{K}_pool_sentinel_src_rerank_template11.jsonl",
    },
    {
        "key": "reasoning_off_edit_dist",
        "label": "Edit Distance Reasoning Off",
        "root_dir": _model_root_dir_map(
            "drive/MyDrive/Mistral_All_Reasoning_Off_edit_dist",
            "drive/MyDrive/Qwen_All_Reasoning_Off_edit_dist",
        ),
        "filename_pattern": "k{K}_edit_dist_template11.jsonl",
    },
]

SRC_LANGS: List[str] = ["eng_Latn"]
TGT_LANGS: List[str] = [
    "wol_Latn",
    "swh_Latn",
    "lus_Latn",
    "mni_Beng",
    "tel_Telu",
    "tam_Taml",
    "uzn_Latn",
]
K_LIST: List[int] = [0, 1, 3, 5, 7, 10]
EVAL_FIRST_M: Optional[int] = 100

# Toggle whether the k=0 baseline is included in the bootstrap pipeline.
# False (default): k=0 is excluded from file reading, chrF++ computation,
#                  paired-bootstrap tests, and all output CSVs.
# True:            k=0 is included (original behaviour).
# Note: changing this toggle while LOAD_FROM_CSV is True won't retroactively
# add or drop k=0 rows that are already cached — you can either delete the
# cached CSV to force a full recompute, or rely on the K_LIST_EFFECTIVE
# filter applied at the very end (which drops k=0 from the final outputs
# whenever the toggle is False, regardless of what the cache contains).
INCLUDE_K0: bool = True
K_LIST_EFFECTIVE: List[int] = K_LIST if INCLUDE_K0 else [k for k in K_LIST if k > 0]

# Match the original evaluation behavior by default: for a fixed k, only keep
# sentence indices that are non-empty across ALL runs at that k.
VALID_SENTENCE_POLICY = "all_runs_at_k"  # alternatives: "paired_runs_only"

# SacreBLEU paired bootstrap configuration.
PAIRED_BS_N = 1000
PAIRED_JOBS = 1
SACREBLEU_SEED = "12345"

# Output root for this script.
SIGNIFICANCE_DIR = "drive/MyDrive/eval_plots_paper_final_paired_bs_chrfpp"

LANG_DISPLAY: Dict[str, str] = {
    "eng_Latn": "English",
    "wol_Latn": "Wolof",
    "swh_Latn": "Swahili",
    "lus_Latn": "Mizo",
    "mni_Beng": "Meitei",
    "tel_Telu": "Telugu",
    "tam_Taml": "Tamil",
    "uzn_Latn": "Uzbek",
}

DF_LONG_COLUMNS = [
    "model_key",
    "model_display",
    "run_key",
    "run_display",
    "src_lang",
    "tgt_lang",
    "direction",
    "k",
    "metric",
    "score",
    "path",
]

SIG_LONG_COLUMNS = [
    "model_key",
    "model_display",
    "src_lang",
    "tgt_lang",
    "direction",
    "k",
    "metric",
    "method_label",
    "on_key",
    "on_display",
    "off_key",
    "off_display",
    "score_on",
    "score_off",
    "delta",
    "mean_on",
    "ci_on",
    "mean_off",
    "ci_off",
    "p_value",
    "significant_p_0_05",
    "significant_p_0_01",
    "sig_marker",
    "better_than_off",
    "n_segments",
    "valid_policy",
    "path_on",
    "path_off",
]

ANNOTATED_LONG_COLUMNS = DF_LONG_COLUMNS + [
    "compare_to_run_key",
    "compare_to_run_display",
    "method_label",
    "delta_vs_compare",
    "p_value",
    "significant_p_0_05",
    "significant_p_0_01",
    "sig_marker",
    "better_than_compare",
    "n_segments",
    "valid_policy",
]

METRIC_LABEL = "chrF++"


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────


def ensure_dir(path: str) -> None:
    """Create a directory and its parents if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def slugify(s: str) -> str:
    """Convert arbitrary text into a filesystem-safe slug."""
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s


_PREFIX_DISPLAY: Dict[str, str] = {
    code.split("_")[0].lower(): name for code, name in LANG_DISPLAY.items()
}


def _apply_limit(n: int, limit_m: Optional[int]) -> int:
    """Apply an optional ceiling to a segment count."""
    if limit_m is None:
        return n
    m = int(limit_m)
    return 0 if m <= 0 else min(n, m)


def _resolve_root_dir(run: Dict[str, Any], model_key: str) -> Optional[str]:
    """Resolve a model-specific root directory from a run config."""
    root_dir = run["root_dir"]
    if isinstance(root_dir, str):
        return root_dir
    if isinstance(root_dir, dict):
        return root_dir.get(model_key)
    return None


def _list_immediate_subdirs(parent: str) -> List[str]:
    """List only the immediate child directories of a parent path."""
    try:
        return sorted(
            [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
        )
    except FileNotFoundError:
        return []


def resolve_model_dirname(base_dir: str, model_key: str) -> Optional[str]:
    """Resolve the actual on-disk model folder name for a configured model key."""
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
    """Return the direction folder name used by the original script."""
    return f"{src_lang.split('_')[0].lower()}_to_{tgt_lang.split('_')[0].lower()}"


def direction_display_from_folder(folder: str) -> str:
    """Convert a direction folder slug into a paper-ready display name."""
    parts = folder.split("_to_")
    if len(parts) == 2:
        src = _PREFIX_DISPLAY.get(parts[0], parts[0].capitalize())
        tgt = _PREFIX_DISPLAY.get(parts[1], parts[1].capitalize())
        return f"{src} to {tgt}"
    return folder


def build_translation_path(
    base_dir: str,
    model_dirname: str,
    direction: str,
    k: int,
    filename_pattern: str,
) -> str:
    """Construct the expected JSONL file path for a run / model / direction / k."""
    return os.path.join(base_dir, model_dirname, direction, filename_pattern.format(K=k))


def _method_label_from_on_label(on_label: str) -> str:
    """Extract a short method label such as RRF, Random, Sentinel, or Edit Distance."""
    label = re.sub(r"\s*[Rr]easoning\s*[Oo]n\s*$", "", on_label).strip()
    return label if label else on_label


def _method_label_from_off_label(off_label: str) -> str:
    """Extract a short method label from an OFF run label."""
    label = re.sub(r"\s*[Rr]easoning\s*[Oo]ff\s*$", "", off_label).strip()
    return label if label else off_label


def _build_k0_fallback_map(
    generation_runs: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """For each non-random run, map its key -> the random run with the same
    reasoning state. Used only as a k=0 fallback, since k=0 is method-agnostic
    (zero in-context examples means the prompt is identical regardless of
    selection method). Returns {non_random_run_key: random_run_dict}.
    """
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


def print_file_structure(all_runs: List[Dict[str, Any]]) -> None:
    """Print resolved root + model directories and a per-(run, model) summary
    of how many of the configured (direction × k) translation files actually
    exist on disk. Runs unconditionally at startup so the user can sanity-check
    folder layout even when the cache short-circuits the main evaluation loop.
    """
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
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────


def read_jsonl_translations(path: str) -> List[str]:
    """Read the 'translation' field from a JSONL file into a list."""
    preds: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line)["translation"])
    return preds


def load_flores_devtest(src_lang: str, tgt_lang: str) -> Tuple[List[str], List[str]]:
    """Load aligned FLORES devtest sources and references for one direction.

    Uses the Muennighoff/flores200 mirror (same schema as facebook/flores —
    `sentence` field, `devtest` split — but actively maintained on HF Hub and
    free of the cp1252/extract-archive issues the legacy loader hits on Colab
    + datasets>=3.0)."""
    ds_src = load_dataset("Muennighoff/flores200", src_lang, trust_remote_code=True)
    ds_tgt = load_dataset("Muennighoff/flores200", tgt_lang, trust_remote_code=True)
    sources = [ex["sentence"] for ex in ds_src["devtest"]]
    targets = [ex["sentence"] for ex in ds_tgt["devtest"]]
    return sources, targets


def _is_empty_translation(text: Optional[str]) -> bool:
    """Return True if a prediction is missing or the empty string."""
    return text is None or text == ""


def find_valid_indices(all_pred_lists: List[List[str]], ceiling: int) -> List[int]:
    """Find the sentence indices that are non-empty in every provided list."""
    if not all_pred_lists:
        return list(range(ceiling))
    valid = set(range(ceiling))
    for preds in all_pred_lists:
        for i in range(ceiling):
            if i >= len(preds) or _is_empty_translation(preds[i]):
                valid.discard(i)
    return sorted(valid)


# ─────────────────────────────────────────────────────────────────────────────
# chrF++ scoring and SacreBLEU paired bootstrap
# ─────────────────────────────────────────────────────────────────────────────


def compute_chrfpp(targets: Sequence[str], predictions: Sequence[str]) -> float:
    """Compute corpus chrF++ with the same CHRF(word_order=2) setup as the source script."""
    metric = CHRF(word_order=2)
    return float(metric.corpus_score(list(predictions), [list(targets)]).score)


def run_sacrebleu_paired_bs(
    refs: Sequence[str],
    preds_off: Sequence[str],
    preds_on: Sequence[str],
    n_samples: int,
    n_jobs: int,
) -> Dict[str, float]:
    """Run SacreBLEU's paired bootstrap test for one OFF-vs-ON comparison."""
    os.environ.setdefault("SACREBLEU_SEED", SACREBLEU_SEED)

    metric = CHRF(word_order=2, references=[list(refs)])
    test = PairedTest(
        named_systems=[
            ("Baseline: OFF", list(preds_off)),
            ("ON", list(preds_on)),
        ],
        metrics={"CHRF": metric},
        references=None,
        test_type="bs",
        n_samples=n_samples,
        n_jobs=n_jobs,
    )
    _, scores = test()

    metric_key = next(k for k in scores.keys() if k != "System")
    baseline_result = scores[metric_key][0]
    candidate_result = scores[metric_key][1]

    return {
        "score_off": float(baseline_result.score),
        "score_on": float(candidate_result.score),
        "mean_off": float(baseline_result.mean) if baseline_result.mean is not None else float("nan"),
        "ci_off": float(baseline_result.ci) if baseline_result.ci is not None else float("nan"),
        "mean_on": float(candidate_result.mean) if candidate_result.mean is not None else float("nan"),
        "ci_on": float(candidate_result.ci) if candidate_result.ci is not None else float("nan"),
        "p_value": float(candidate_result.p_value) if candidate_result.p_value is not None else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────


def append_raw_score_row(
    rows: List[Dict[str, Any]],
    model_key: str,
    model_display: str,
    run_key: str,
    run_display: str,
    src_lang: str,
    tgt_lang: str,
    direction: str,
    k: int,
    score: float,
    path: str,
) -> None:
    """Append one chrF++ raw-score row using the original DF_LONG schema."""
    rows.append(
        {
            "model_key": model_key,
            "model_display": model_display,
            "run_key": run_key,
            "run_display": run_display,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "direction": direction,
            "k": k,
            "metric": METRIC_LABEL,
            "score": score,
            "path": path,
        }
    )


def build_annotated_long_df(
    df_raw: pd.DataFrame,
    df_sig: pd.DataFrame,
) -> pd.DataFrame:
    """Annotate ON-system raw chrF++ rows with paired significance information."""
    if df_raw.empty:
        return pd.DataFrame(columns=ANNOTATED_LONG_COLUMNS)

    df_ann = df_raw.copy()
    for col in [
        "compare_to_run_key",
        "compare_to_run_display",
        "method_label",
        "delta_vs_compare",
        "p_value",
        "significant_p_0_05",
        "significant_p_0_01",
        "sig_marker",
        "better_than_compare",
        "n_segments",
        "valid_policy",
    ]:
        df_ann[col] = pd.NA

    if df_sig.empty:
        return df_ann.reindex(columns=ANNOTATED_LONG_COLUMNS)

    sig_for_merge = df_sig.rename(
        columns={
            "on_key": "run_key",
            "on_display": "run_display",
            "off_key": "compare_to_run_key",
            "off_display": "compare_to_run_display",
            "delta": "delta_vs_compare",
            "better_than_off": "better_than_compare",
        }
    )[
        [
            "model_key",
            "src_lang",
            "tgt_lang",
            "direction",
            "k",
            "metric",
            "run_key",
            "run_display",
            "compare_to_run_key",
            "compare_to_run_display",
            "method_label",
            "delta_vs_compare",
            "p_value",
            "significant_p_0_05",
            "significant_p_0_01",
            "sig_marker",
            "better_than_compare",
            "n_segments",
            "valid_policy",
        ]
    ]

    merge_keys = [
        "model_key",
        "src_lang",
        "tgt_lang",
        "direction",
        "k",
        "metric",
        "run_key",
        "run_display",
    ]

    df_merged = df_ann.drop(
        columns=[
            "compare_to_run_key",
            "compare_to_run_display",
            "method_label",
            "delta_vs_compare",
            "p_value",
            "significant_p_0_05",
            "significant_p_0_01",
            "sig_marker",
            "better_than_compare",
            "n_segments",
            "valid_policy",
        ]
    ).merge(sig_for_merge, on=merge_keys, how="left")

    return df_merged.reindex(columns=ANNOTATED_LONG_COLUMNS)


def export_direction_tables(
    df_raw: pd.DataFrame,
    df_sig: pd.DataFrame,
    df_annotated: pd.DataFrame,
    out_dir: str,
) -> None:
    """Export per-direction CSV files mirroring the original folder structure."""
    ensure_dir(out_dir)
    df_raw.to_csv(os.path.join(out_dir, "raw_scores_long_chrfpp.csv"), index=False)
    df_sig.to_csv(os.path.join(out_dir, "paired_bs_significance_long.csv"), index=False)
    df_annotated.to_csv(
        os.path.join(out_dir, "raw_scores_long_chrfpp_with_significance.csv"),
        index=False,
    )


def export_global_tables(
    df_raw: pd.DataFrame,
    df_sig: pd.DataFrame,
    df_annotated: pd.DataFrame,
    out_root: str,
) -> None:
    """Export global CSVs across every model, direction, method, and k."""
    ensure_dir(out_root)

    df_raw = df_raw.reindex(columns=DF_LONG_COLUMNS).sort_values(
        ["model_key", "direction", "run_key", "k"], kind="stable"
    )
    df_sig = df_sig.reindex(columns=SIG_LONG_COLUMNS).sort_values(
        ["model_key", "direction", "method_label", "k"], kind="stable"
    )
    df_annotated = df_annotated.reindex(columns=ANNOTATED_LONG_COLUMNS).sort_values(
        ["model_key", "direction", "run_key", "k"], kind="stable"
    )

    df_raw.to_csv(os.path.join(out_root, "raw_scores_long_chrfpp.csv"), index=False)
    df_sig.to_csv(os.path.join(out_root, "paired_bs_significance_long.csv"), index=False)
    df_annotated.to_csv(
        os.path.join(out_root, "raw_scores_long_chrfpp_with_significance.csv"),
        index=False,
    )


def evaluate_significance_all(
    runs_on: List[Dict[str, Any]],
    runs_off: List[Dict[str, Any]],
    methods_to_compute: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate raw chrF++ and paired bootstrap significance for the configured
    comparisons.

    Parameters
    ----------
    runs_on, runs_off
        The full lists of reasoning-ON and reasoning-OFF run configs. Files for
        every run in these lists are read at every k in K_LIST_EFFECTIVE so the
        per-(model, direction, k) valid-sentence policy stays consistent.
    methods_to_compute
        If provided, only run chrF++ scoring + paired bootstrap on the named
        methods. The remaining runs are still *read* from disk for valid-index
        computation. If None, every method in `runs_on` is computed.

    Returns
    -------
    df_raw : raw chrF++ rows (DF_LONG_COLUMNS schema)
    df_sig : paired-bootstrap significance rows (SIG_LONG_COLUMNS schema)
    """
    flores_cache: Dict[Tuple[str, str], Tuple[List[str], List[str]]] = {}
    all_runs = runs_on + runs_off

    method_label_by_run_key: Dict[str, str] = {}
    for r in runs_on:
        method_label_by_run_key[r["key"]] = _method_label_from_on_label(r["label"])
    for r in runs_off:
        method_label_by_run_key[r["key"]] = _method_label_from_off_label(r["label"])

    if methods_to_compute is None:
        compute_method_set = set(method_label_by_run_key.values())
    else:
        compute_method_set = set(methods_to_compute)

    raw_rows: List[Dict[str, Any]] = []
    sig_rows: List[Dict[str, Any]] = []

    resolved: Dict[Tuple[str, str], Optional[Tuple[str, str]]] = {}
    for run in all_runs:
        for model_key in MODELS:
            root = _resolve_root_dir(run, model_key)
            if root is None:
                resolved[(run["key"], model_key)] = None
                continue
            dirname = resolve_model_dirname(root, model_key)
            resolved[(run["key"], model_key)] = None if dirname is None else (root, dirname)

    print("\n[Model directory resolution]")
    for run in all_runs:
        print(f"  {run['key']}:")
        for model_key in MODELS:
            entry = resolved.get((run["key"], model_key))
            if entry is None:
                root = _resolve_root_dir(run, model_key)
                root_str = root if root else "NO ROOT CONFIGURED"
                print(f"    • {model_key} → NOT FOUND (root: {root_str})")
            else:
                print(f"    • {model_key} → {entry[0]}/{entry[1]}")

    # Build the k=0 fallback map once: non-random runs reuse the random run's
    # k=0 file when their own k=0 file is missing (k=0 is method-agnostic).
    k0_fallback = _build_k0_fallback_map(all_runs)
    if k0_fallback:
        print("\n[k=0 fallback map]")
        for src_key, fb_run in k0_fallback.items():
            print(f"  • {src_key} → {fb_run['key']}  (only when k=0 file missing)")

    for src_lang in SRC_LANGS:
        for tgt_lang in TGT_LANGS:
            direction = direction_folder_name(src_lang, tgt_lang)
            direction_disp = direction_display_from_folder(direction)

            if (src_lang, tgt_lang) not in flores_cache:
                flores_cache[(src_lang, tgt_lang)] = load_flores_devtest(src_lang, tgt_lang)
            sources_full, targets_full = flores_cache[(src_lang, tgt_lang)]

            ceiling = _apply_limit(len(sources_full), EVAL_FIRST_M)
            targets_ceil = targets_full[:ceiling]

            for model_key, model_display in MODELS.items():
                file_registry: Dict[Tuple[str, int], Dict[str, Any]] = {}
                preds_by_k: Dict[int, List[List[str]]] = {k: [] for k in K_LIST_EFFECTIVE}

                # Read files for every run × k so the valid-index policy can
                # use the full run set even when only a subset of methods is
                # being recomputed.
                for run in all_runs:
                    run_key = run["key"]
                    resolved_entry = resolved.get((run_key, model_key))
                    for k in K_LIST_EFFECTIVE:
                        reg_entry: Dict[str, Any] = {"path": "", "preds": None}
                        if resolved_entry is not None:
                            root_dir, model_dirname = resolved_entry
                            path = build_translation_path(
                                root_dir,
                                model_dirname,
                                direction,
                                k,
                                run["filename_pattern"],
                            )
                            if os.path.exists(path):
                                preds_full = read_jsonl_translations(path)
                                preds_ceil = preds_full[:ceiling]
                                reg_entry["path"] = path
                                reg_entry["preds"] = preds_ceil
                                preds_by_k[k].append(preds_ceil)

                        # ── k=0 fallback: pull from the random run if this
                        # run lacks its own k=0 file.
                        if k == 0 and reg_entry["preds"] is None and run_key in k0_fallback:
                            fb_run = k0_fallback[run_key]
                            fb_resolved = resolved.get((fb_run["key"], model_key))
                            if fb_resolved is not None:
                                fb_root_dir, fb_model_dirname = fb_resolved
                                fb_path = build_translation_path(
                                    fb_root_dir,
                                    fb_model_dirname,
                                    direction,
                                    k,
                                    fb_run["filename_pattern"],
                                )
                                if os.path.exists(fb_path):
                                    fb_preds_full = read_jsonl_translations(fb_path)
                                    fb_preds_ceil = fb_preds_full[:ceiling]
                                    reg_entry["path"] = fb_path
                                    reg_entry["preds"] = fb_preds_ceil
                                    preds_by_k[k].append(fb_preds_ceil)
                        # ──────────────────────────────────────────────────

                        file_registry[(run_key, k)] = reg_entry

                valid_indices_by_k: Dict[int, List[int]] = {}
                for k in K_LIST_EFFECTIVE:
                    if VALID_SENTENCE_POLICY == "all_runs_at_k":
                        valid_indices_by_k[k] = find_valid_indices(preds_by_k[k], ceiling)
                    else:
                        valid_indices_by_k[k] = list(range(ceiling))

                count_out_dir = os.path.join(SIGNIFICANCE_DIR, slugify(model_key), direction)
                ensure_dir(count_out_dir)
                per_k_counts: Dict[str, Any] = {}

                print(
                    f"\n[{model_display} | {direction_disp}] Valid sentences per k "
                    f"({VALID_SENTENCE_POLICY}):"
                )
                for k in K_LIST_EFFECTIVE:
                    n_valid = len(valid_indices_by_k[k])
                    print(f"    k={k:2d} → {n_valid} / {ceiling} sentences")
                    per_k_counts[str(k)] = {
                        "valid": n_valid,
                        "total_considered": ceiling,
                        "policy": VALID_SENTENCE_POLICY,
                    }

                with open(
                    os.path.join(count_out_dir, "valid_sentence_counts.json"),
                    "w",
                    encoding="utf-8",
                ) as fh:
                    json.dump(
                        {
                            "model_key": model_key,
                            "model_display": model_display,
                            "src_lang": src_lang,
                            "tgt_lang": tgt_lang,
                            "direction": direction,
                            "direction_display": direction_disp,
                            "per_k": per_k_counts,
                        },
                        fh,
                        indent=2,
                    )

                # Raw chrF++ rows — only for runs whose method is being recomputed.
                for run in all_runs:
                    run_key = run["key"]
                    if method_label_by_run_key[run_key] not in compute_method_set:
                        continue
                    for k in K_LIST_EFFECTIVE:
                        entry = file_registry[(run_key, k)]
                        preds = entry["preds"]
                        path = entry["path"]
                        if preds is None:
                            continue

                        valid_indices = valid_indices_by_k[k]
                        if VALID_SENTENCE_POLICY == "paired_runs_only":
                            valid_indices = find_valid_indices([preds], min(ceiling, len(preds)))

                        if not valid_indices:
                            continue

                        refs_filtered = [targets_ceil[i] for i in valid_indices]
                        preds_filtered = [preds[i] for i in valid_indices]
                        score = compute_chrfpp(refs_filtered, preds_filtered)
                        append_raw_score_row(
                            raw_rows,
                            model_key=model_key,
                            model_display=model_display,
                            run_key=run_key,
                            run_display=run["label"],
                            src_lang=src_lang,
                            tgt_lang=tgt_lang,
                            direction=direction,
                            k=k,
                            score=score,
                            path=path,
                        )

                # Paired bootstrap: only for ON/OFF pairs whose method is being recomputed.
                for on_run, off_run in zip(runs_on, runs_off):
                    method_label = _method_label_from_on_label(on_run["label"])
                    if method_label not in compute_method_set:
                        continue
                    for k in K_LIST_EFFECTIVE:
                        on_entry = file_registry[(on_run["key"], k)]
                        off_entry = file_registry[(off_run["key"], k)]
                        preds_on = on_entry["preds"]
                        preds_off = off_entry["preds"]
                        if preds_on is None or preds_off is None:
                            continue

                        if VALID_SENTENCE_POLICY == "all_runs_at_k":
                            valid_indices = valid_indices_by_k[k]
                        else:
                            pair_ceiling = min(ceiling, len(preds_on), len(preds_off))
                            valid_indices = find_valid_indices(
                                [preds_on[:pair_ceiling], preds_off[:pair_ceiling]],
                                pair_ceiling,
                            )

                        if not valid_indices:
                            continue

                        refs_filtered = [targets_ceil[i] for i in valid_indices]
                        preds_on_filtered = [preds_on[i] for i in valid_indices]
                        preds_off_filtered = [preds_off[i] for i in valid_indices]

                        result = run_sacrebleu_paired_bs(
                            refs=refs_filtered,
                            preds_off=preds_off_filtered,
                            preds_on=preds_on_filtered,
                            n_samples=PAIRED_BS_N,
                            n_jobs=PAIRED_JOBS,
                        )

                        delta = result["score_on"] - result["score_off"]
                        p_value = result["p_value"]
                        significant_0_01 = pd.notna(p_value) and float(p_value) < 0.01
                        significant_0_05 = pd.notna(p_value) and float(p_value) < 0.05
                        sig_marker = "**" if significant_0_01 else "*" if significant_0_05 else ""

                        print(
                            f"  [{model_display} | {direction_disp} | {method_label} | k={k}] "
                            f"OFF={result['score_off']:.2f} ON={result['score_on']:.2f} "
                            f"Δ={delta:+.2f} p={p_value:.4f} sig='{sig_marker}'"
                        )

                        sig_rows.append(
                            {
                                "model_key": model_key,
                                "model_display": model_display,
                                "src_lang": src_lang,
                                "tgt_lang": tgt_lang,
                                "direction": direction,
                                "k": k,
                                "metric": METRIC_LABEL,
                                "method_label": method_label,
                                "on_key": on_run["key"],
                                "on_display": on_run["label"],
                                "off_key": off_run["key"],
                                "off_display": off_run["label"],
                                "score_on": result["score_on"],
                                "score_off": result["score_off"],
                                "delta": delta,
                                "mean_on": result["mean_on"],
                                "ci_on": result["ci_on"],
                                "mean_off": result["mean_off"],
                                "ci_off": result["ci_off"],
                                "p_value": p_value,
                                "significant_p_0_05": bool(significant_0_05),
                                "significant_p_0_01": bool(significant_0_01),
                                "sig_marker": sig_marker,
                                "better_than_off": bool(delta > 0),
                                "n_segments": len(valid_indices),
                                "valid_policy": VALID_SENTENCE_POLICY,
                                "path_on": on_entry["path"],
                                "path_off": off_entry["path"],
                            }
                        )

    df_raw = pd.DataFrame(raw_rows, columns=DF_LONG_COLUMNS)
    df_sig = pd.DataFrame(sig_rows, columns=SIG_LONG_COLUMNS)
    return df_raw, df_sig


def main() -> None:
    """Run the full chrF++ paired-bootstrap significance pipeline and export CSVs.

    Behaviour driven by two config flags at the top of this file:
      • LOAD_FROM_CSV  — when True and both raw/sig CSVs exist, load them,
                         compute only methods that are missing from the cache,
                         concatenate, and persist the union.
      • INCLUDE_K0     — when False, k=0 is excluded from every step
                         (file reading, scoring, bootstrap, and final CSVs).
    """
    ensure_dir(SIGNIFICANCE_DIR)
    raw_csv = os.path.join(SIGNIFICANCE_DIR, "raw_scores_long_chrfpp.csv")
    sig_csv = os.path.join(SIGNIFICANCE_DIR, "paired_bs_significance_long.csv")

    all_runs = GENERATION_RUNS_REASONING_ON + GENERATION_RUNS_REASONING_OFF
    print_file_structure(all_runs)

    configured_methods = {
        _method_label_from_on_label(r["label"]) for r in GENERATION_RUNS_REASONING_ON
    }

    if LOAD_FROM_CSV and os.path.exists(raw_csv) and os.path.exists(sig_csv):
        print(f"\n[LOAD_FROM_CSV] Reading existing caches:")
        print(f"  {raw_csv}")
        print(f"  {sig_csv}")
        df_raw_cached = pd.read_csv(raw_csv)
        df_sig_cached = pd.read_csv(sig_csv)
        df_raw_cached["k"] = df_raw_cached["k"].astype(int)
        df_sig_cached["k"] = df_sig_cached["k"].astype(int)
        df_raw_cached["score"] = pd.to_numeric(df_raw_cached["score"], errors="coerce")
        print(f"  Loaded {len(df_raw_cached)} raw rows + {len(df_sig_cached)} significance rows.")

        cached_methods = (
            set(df_sig_cached["method_label"].unique()) if not df_sig_cached.empty else set()
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

            df_raw_new, df_sig_new = evaluate_significance_all(
                GENERATION_RUNS_REASONING_ON,
                GENERATION_RUNS_REASONING_OFF,
                methods_to_compute=missing_methods,
            )
            df_raw = pd.concat([df_raw_cached, df_raw_new], ignore_index=True)
            df_sig = pd.concat([df_sig_cached, df_sig_new], ignore_index=True)
            print(
                f"\n[Cache updated]\n"
                f"  Raw: {len(df_raw_cached)} cached + {len(df_raw_new)} new = {len(df_raw)} rows.\n"
                f"  Sig: {len(df_sig_cached)} cached + {len(df_sig_new)} new = {len(df_sig)} rows."
            )
        else:
            print("  All configured methods already cached; skipping the bootstrap step.")
            df_raw = df_raw_cached
            df_sig = df_sig_cached
    else:
        df_raw, df_sig = evaluate_significance_all(
            GENERATION_RUNS_REASONING_ON,
            GENERATION_RUNS_REASONING_OFF,
            methods_to_compute=None,
        )

    # Apply the INCLUDE_K0 toggle to the final outputs. This filters cached
    # rows too, so flipping INCLUDE_K0 immediately reflects in what gets
    # written even when LOAD_FROM_CSV is True.
    df_raw = df_raw[df_raw["k"].isin(K_LIST_EFFECTIVE)].reset_index(drop=True)
    df_sig = df_sig[df_sig["k"].isin(K_LIST_EFFECTIVE)].reset_index(drop=True)

    df_ann = build_annotated_long_df(df_raw, df_sig)

    # Global tables first (crash safety: if per-direction export fails the
    # global state is already on disk).
    export_global_tables(df_raw, df_sig, df_ann, SIGNIFICANCE_DIR)

    # Per-direction tables using the complete combined data.
    for model_key in MODELS:
        for src_lang in SRC_LANGS:
            for tgt_lang in TGT_LANGS:
                direction = direction_folder_name(src_lang, tgt_lang)
                df_raw_dir = df_raw[
                    (df_raw["model_key"] == model_key) & (df_raw["direction"] == direction)
                ].copy()
                df_sig_dir = df_sig[
                    (df_sig["model_key"] == model_key) & (df_sig["direction"] == direction)
                ].copy()
                df_ann_dir = build_annotated_long_df(df_raw_dir, df_sig_dir)
                out_dir = os.path.join(SIGNIFICANCE_DIR, slugify(model_key), direction)
                ensure_dir(out_dir)
                export_direction_tables(df_raw_dir, df_sig_dir, df_ann_dir, out_dir)

    print(f"\nAll paired-bootstrap artefacts saved to: {SIGNIFICANCE_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()