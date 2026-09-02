"""
dataset_registry.py
───────────────────
Single source of truth for every dataset arm the analysis scripts run on.

Standardized dataset contract — a dataset is fully described by:
  • a sentence loader  load_sentences(lang) -> {"dev": [...], "devtest": [...]}
    ("dev" = the selection pool, "devtest" = the fixed test set that every
    translation JSONL is row-aligned to: line i ↔ devtest sentence i),
  • a target-language list (source is English in every current arm),
  • a naming prefix shared by all of the arm's run roots (e.g. "WMT24PP_"),
  • the k-grid the arm was generated with.

Every analysis script resolves its arm through get_dataset() (RTRACE_DATASET
env var, default "flores") and derives ALL paths from the spec — so adding a
new dataset is one DatasetSpec entry here, provided its loader speaks the
{"dev","devtest"} contract.

Path conventions (out_base = RTRACE_OUT_BASE, default per-arm):
  translations      {out_base}/{prefix}{Family}_All_Reasoning_{State}{suffix}
  reasoning traces  flores : {out_base}/reasoning_traces_{Family}_all{suffix}
                    wmt24pp: {out_base}/{prefix}reasoning_traces_{Family}_{method}_{state}
  analysis outputs  {out_base}/{prefix}<analysis-name>
The flores arm keeps its legacy Colab layout (and drive/MyDrive default
out_base) so Drive-mounted reruns of the July results still work unchanged.

Inputs:  RTRACE_DATASET / RTRACE_OUT_BASE / RTRACE_EVAL_MODELS env vars.
Outputs: DatasetSpec instances via get_dataset(); filter_models() helper.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# suffix used in run roots ("" | "_random" | ...) → short method name used in
# the WMT-era per-method reasoning-trace roots.
_SUFFIX_TO_METHOD: Dict[str, str] = {
    "": "rrf",
    "_random": "random",
    "_sentinel": "sentinel",
    "_edit_dist": "edit_dist",
}


@dataclass(frozen=True)
class DatasetSpec:
    """
    Purpose: Describe one dataset arm (languages, k-grid, naming, loader).
    Inputs: constructor fields below.
    Outputs: path builders + sentence loader honouring the standard contract.
    """
    key: str                      # "flores" | "wmt24pp"
    prefix: str                   # root-name prefix, e.g. "WMT24PP_"
    src_langs: Tuple[str, ...]
    tgt_langs: Tuple[str, ...]
    k_list: Tuple[int, ...]       # k values the arm was generated with
    lang_display: Dict[str, str] = field(default_factory=dict)
    default_out_base: str = "runs"
    eval_plots_dirname: str = ""  # directory (under out_base) holding raw_scores_long.csv
    trace_layout: str = "family_method_state"  # or legacy "family_all_suffix"
    emb_dirname: str = ""         # retrieval-embedding matrices directory (under out_base)

    def out_base(self) -> str:
        return os.environ.get("RTRACE_OUT_BASE", self.default_out_base)

    def load_sentences(self, lang: str) -> Dict[str, List[str]]:
        """Standard loader contract: {"dev": [...], "devtest": [...]}."""
        # Lazy imports keep this module cheap to import from any venv.
        if self.key == "wmt24pp":
            from src.data.wmt24pp import load_wmt24pp_sentences
            return load_wmt24pp_sentences(lang)
        from src.retrieval.retrieval_helpers import load_flores_sentences
        return load_flores_sentences(lang)

    def generation_root(self, family: str, state_title: str, suffix: str) -> str:
        """Translation root for (family, On|Off, method suffix)."""
        return os.path.join(
            self.out_base(), f"{self.prefix}{family}_All_Reasoning_{state_title}{suffix}"
        )

    def reasoning_trace_root(self, family: str, suffix: str, state: str = "on") -> str:
        """Reasoning-trace root for (family, method suffix); traces exist for ON only."""
        if self.trace_layout == "family_all_suffix":
            return os.path.join(self.out_base(), f"reasoning_traces_{family}_all{suffix}")
        method = _SUFFIX_TO_METHOD[suffix]
        return os.path.join(
            self.out_base(), f"{self.prefix}reasoning_traces_{family}_{method}_{state}"
        )

    def analysis_dir(self, name: str) -> str:
        """Output directory for one analysis stage, e.g. eval_token_counts."""
        return os.path.join(self.out_base(), f"{self.prefix}{name}")

    def eval_scores_csv(self) -> str:
        """eval_pipeline's raw_scores_long.csv for this arm."""
        return os.path.join(self.out_base(), self.eval_plots_dirname, "raw_scores_long.csv")

    def emb_root(self) -> str:
        """Retrieval-embedding matrices root (RTRACE_EMB_ROOT overrides)."""
        return os.environ.get("RTRACE_EMB_ROOT", os.path.join(self.out_base(), self.emb_dirname))


DATASETS: Dict[str, DatasetSpec] = {
    "flores": DatasetSpec(
        key="flores",
        prefix="",
        src_langs=("eng_Latn",),
        tgt_langs=(
            "wol_Latn", "swh_Latn", "lus_Latn", "mni_Beng",
            "tel_Telu", "tam_Taml", "uzn_Latn",
        ),
        k_list=(0, 1, 3, 5, 7, 10),
        lang_display={
            "eng_Latn": "English", "wol_Latn": "Wolof", "swh_Latn": "Swahili",
            "lus_Latn": "Mizo", "mni_Beng": "Meitei", "tel_Telu": "Telugu",
            "tam_Taml": "Tamil", "uzn_Latn": "Uzbek",
        },
        default_out_base="drive/MyDrive",
        eval_plots_dirname="eval_plots_paper_initial",
        trace_layout="family_all_suffix",
        emb_dirname="flores_embeddings",
    ),
    "wmt24pp": DatasetSpec(
        key="wmt24pp",
        prefix="WMT24PP_",
        src_langs=("eng_Latn",),
        tgt_langs=("cat_Latn", "zul_Latn", "mal_Mlym", "slk_Latn", "isl_Latn"),
        k_list=(1, 3, 5, 7, 10),  # no zero-shot arm in WMT24++
        lang_display={
            "eng_Latn": "English", "cat_Latn": "Catalan", "zul_Latn": "Zulu",
            "mal_Mlym": "Malayalam", "slk_Latn": "Slovak", "isl_Latn": "Icelandic",
        },
        default_out_base="runs",
        eval_plots_dirname="WMT24PP_eval_plots",
        trace_layout="family_method_state",
        emb_dirname="wmt24pp_embeddings",
    ),
}


def get_dataset() -> DatasetSpec:
    """
    Purpose: Resolve the active dataset arm from RTRACE_DATASET.
    Inputs: RTRACE_DATASET env ("flores" default).
    Outputs: the DatasetSpec.
    """
    key = os.environ.get("RTRACE_DATASET", "flores").lower()
    if key not in DATASETS:
        raise ValueError(f"RTRACE_DATASET must be one of {sorted(DATASETS)}, got {key!r}")
    return DATASETS[key]


def filter_models(models: Dict[str, str]) -> Dict[str, str]:
    """
    Purpose: Restrict a model_key→display map via RTRACE_EVAL_MODELS
             (csv of model keys; same contract as eval_pipeline.py).
    Inputs: full models dict.
    Outputs: filtered dict (order follows the env list); unchanged when unset.
    """
    raw = os.environ.get("RTRACE_EVAL_MODELS", "").strip()
    if not raw:
        return models
    keep = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [k for k in keep if k not in models]
    if unknown:
        raise ValueError(f"Unknown model key(s) in RTRACE_EVAL_MODELS: {unknown} (known: {sorted(models)})")
    return {k: models[k] for k in keep}
