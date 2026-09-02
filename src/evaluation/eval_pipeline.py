"""
eval_pipeline.py
────────────────
Full evaluation pipeline for Mistral & Qwen translation runs with:
  • Six models (three Mistral, three Qwen)
  • Two index-aligned run lists (reasoning ON / reasoning OFF)
  • Per-model root directories (Mistral and Qwen live in separate trees)
  • Combined score plots (all methods, legend outside the axes)
  • Delta superplots  Δ(reasoning on − reasoning off) per method pair
  • Aggregated cross-language delta superplots (one subplot per model,
    averaged across all translation directions)
  • Aggregated cross-language raw-score superplots (eight lines per subplot:
    four methods × two reasoning states)
  • Cross-model comparison superplots (all models, Mistral-only, Qwen-only)
  • Professionally formatted table images (paper-ready, no heatmap)
  • Spearman correlation matrices across all metrics (raw + reasoning deltas)
  • Optional k=0 toggle: include or exclude the zero-shot baseline from plots
  • Human-readable direction titles  ("English to Wolof", not "eng_to_wol")
  • Publication-quality figure formatting throughout — paper-ready text sizes
  • Smart CSV cache: loads existing scores, computes only newly-configured
    runs (e.g. when edit_dist is added later), merges, and persists.
  • k=0 fallback: non-random runs reuse the random run's k=0 file (since
    k=0 is method-agnostic — zero in-context examples means the prompt
    is identical regardless of selection method).
"""

import os, json, re, math
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from scipy.stats import spearmanr
from comet import download_model, load_from_checkpoint
import fasttext
from huggingface_hub import hf_hub_download
from src.common.plots import _collect_legend
from src.common.fasttext_compat import _patch_fasttext_for_numpy2

# ─────────────────────────────────────────────────────────────────────────────
# Compatibility shim: fasttext + NumPy 2.x
# ─────────────────────────────────────────────────────────────────────────────
_patch_fasttext_for_numpy2()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
LOAD_FROM_CSV = True

# ── Dataset arm: flores (notebook-era Drive defaults) or wmt24pp (cluster
#    runs/ roots). RTRACE_DATASET selects generation roots, target languages,
#    k grid, references and the output dir; RTRACE_EVAL_MODELS (csv of model
#    keys) restricts scoring to a subset — e.g. the finished Qwen runs while
#    the Mistral grid is still generating.
DATASET = os.environ.get("RTRACE_DATASET", "flores").lower()
if DATASET not in ("flores", "wmt24pp"):
    raise ValueError(f"RTRACE_DATASET must be 'flores' or 'wmt24pp', got {DATASET!r}")
OUT_BASE = os.environ.get("RTRACE_OUT_BASE", "runs")
if DATASET == "wmt24pp":
    _MISTRAL_ROOT = f"{OUT_BASE}/WMT24PP_Mistral_All_Reasoning_"
    _QWEN_ROOT = f"{OUT_BASE}/WMT24PP_Qwen_All_Reasoning_"
else:
    _MISTRAL_ROOT = "drive/MyDrive/Mistral_All_Reasoning_"
    _QWEN_ROOT = "drive/MyDrive/Qwen_All_Reasoning_"

MODELS: Dict[str, str] = {
    "ministral_8b": "Ministral 8B",
    "ministral_14b": "Ministral 14B",
    "magistral_small": "Magistral 24B",
    "qwen3_8b": "Qwen3 8B",
    "qwen3_14b": "Qwen3 14B",
    "qwen3_32b": "Qwen3 32B",
}

MODEL_ORDER: List[str] = list(MODELS.keys())

MODEL_FAMILIES: Dict[str, List[str]] = {
    "Mistral": ["ministral_8b", "ministral_14b", "magistral_small"],
    "Qwen":    ["qwen3_8b",     "qwen3_14b",     "qwen3_32b"],
}

# Partial-grid evaluation: RTRACE_EVAL_MODELS=qwen3_8b,qwen3_14b,qwen3_32b
# scores only those models; family plots collapse to the families present.
_EVAL_MODELS_ENV = os.environ.get("RTRACE_EVAL_MODELS", "").strip()
if _EVAL_MODELS_ENV:
    _keep = {m.strip() for m in _EVAL_MODELS_ENV.split(",") if m.strip()}
    _unknown = _keep - set(MODELS)
    if _unknown:
        raise ValueError(f"RTRACE_EVAL_MODELS contains unknown keys: {sorted(_unknown)}")
    MODELS = {k: v for k, v in MODELS.items() if k in _keep}
    MODEL_ORDER = list(MODELS.keys())
    MODEL_FAMILIES = {f: [m for m in ms if m in MODELS] for f, ms in MODEL_FAMILIES.items()}
    MODEL_FAMILIES = {f: ms for f, ms in MODEL_FAMILIES.items() if ms}

MODEL_SIZE_THRESHOLD: float = 14.0


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
        "root_dir": _model_root_dir_map(_MISTRAL_ROOT + "On", _QWEN_ROOT + "On"),
        "filename_pattern": "k{K}_rrf_template11.jsonl",
    },
    {
        "key": "reasoning_on_random",
        "label": "Random Reasoning On",
        "root_dir": _model_root_dir_map(_MISTRAL_ROOT + "On_random", _QWEN_ROOT + "On_random"),
        "filename_pattern": "k{K}_random_pool_template11.jsonl",
    },
    {
        "key": "reasoning_on_sentinel",
        "label": "Sentinel Reasoning On",
        "root_dir": _model_root_dir_map(_MISTRAL_ROOT + "On_sentinel", _QWEN_ROOT + "On_sentinel"),
        "filename_pattern": "k{K}_pool_sentinel_src_rerank_template11.jsonl",
    },
    {
        "key": "reasoning_on_edit_dist",
        "label": "Edit Distance Reasoning On",
        "root_dir": _model_root_dir_map(_MISTRAL_ROOT + "On_edit_dist", _QWEN_ROOT + "On_edit_dist"),
        "filename_pattern": "k{K}_edit_dist_template11.jsonl",
    },
]

GENERATION_RUNS_REASONING_OFF: List[Dict[str, Any]] = [
    {
        "key": "reasoning_off",
        "label": "RRF Reasoning Off",
        "root_dir": _model_root_dir_map(_MISTRAL_ROOT + "Off", _QWEN_ROOT + "Off"),
        "filename_pattern": "k{K}_rrf_template11.jsonl",
    },
    {
        "key": "reasoning_off_random",
        "label": "Random Reasoning Off",
        "root_dir": _model_root_dir_map(_MISTRAL_ROOT + "Off_random", _QWEN_ROOT + "Off_random"),
        "filename_pattern": "k{K}_random_pool_template11.jsonl",
    },
    {
        "key": "reasoning_off_sentinel",
        "label": "Sentinel Reasoning Off",
        "root_dir": _model_root_dir_map(_MISTRAL_ROOT + "Off_sentinel", _QWEN_ROOT + "Off_sentinel"),
        "filename_pattern": "k{K}_pool_sentinel_src_rerank_template11.jsonl",
    },
    {
        "key": "reasoning_off_edit_dist",
        "label": "Edit Distance Reasoning Off",
        "root_dir": _model_root_dir_map(_MISTRAL_ROOT + "Off_edit_dist", _QWEN_ROOT + "Off_edit_dist"),
        "filename_pattern": "k{K}_edit_dist_template11.jsonl",
    },
]

SRC_LANGS: List[str] = ["eng_Latn"]
if DATASET == "wmt24pp":
    TGT_LANGS: List[str] = ["cat_Latn", "zul_Latn", "mal_Mlym", "slk_Latn", "isl_Latn"]
    # The WMT24++ generation grid ran K_LIST=1,3,5,7,10 — no k=0 files exist,
    # so the zero-shot column is omitted rather than filled with NaNs.
    K_LIST: List[int] = [1, 3, 5, 7, 10]
else:
    TGT_LANGS = [
        "wol_Latn",
        "swh_Latn",
        "lus_Latn",
        "mni_Beng",
        "tel_Telu",
        "tam_Taml",
        "uzn_Latn",
    ]
    K_LIST = [0, 1, 3, 5, 7, 10]

# Toggle whether the k=0 baseline shows up in every plot/table.
# False (default): k=0 is excluded from line plots, delta plots, superplots,
#                  and table figures. Raw CSV / Excel exports and the
#                  correlation matrices are unaffected (correlations already
#                  exclude k=0 by design; CSV/Excel exports still write every
#                  k value so the cache stays complete).
# True:            k=0 is plotted everywhere (original behaviour).
INCLUDE_K0_IN_PLOTS: bool = False
K_LIST_PLOT: List[int] = K_LIST if INCLUDE_K0_IN_PLOTS else [k for k in K_LIST if k > 0]

EVAL_FIRST_M: Optional[int] = 100

PLOTS_DIR = os.environ.get(
    "RTRACE_EVAL_PLOTS_DIR",
    f"{OUT_BASE}/WMT24PP_eval_plots" if DATASET == "wmt24pp" else "drive/MyDrive/eval_plots_paper_initial",
)
LANG_DISPLAY: Dict[str, str] = {
    "eng_Latn": "English", "wol_Latn": "Wolof", "swh_Latn": "Swahili",
    "lus_Latn": "Mizo", "mni_Beng": "Meitei", "tel_Telu": "Telugu",
    "tam_Taml": "Tamil", "uzn_Latn": "Uzbek",
    # WMT24++ arm
    "cat_Latn": "Catalan", "zul_Latn": "Zulu", "mal_Mlym": "Malayalam",
    "slk_Latn": "Slovak", "isl_Latn": "Icelandic",
}

COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"
COMET_BATCH_SIZE = 64; COMET_GPUS = 1; EMPTY_AS_ZERO = True
FASTTEXT_LID_REPO = "facebook/fasttext-language-identification"
FASTTEXT_LID_FILENAME = "model.bin"
FILL_MISSING_AS_NAN = True

DF_LONG_COLUMNS = ["model_key","model_display","run_key","run_display","src_lang","tgt_lang","direction","k","metric","score","path"]
DF_DELTA_COLUMNS = ["model_key","model_display","direction","k","metric","delta","method_label","on_key","off_key"]
DF_AGG_COLUMNS = ["model_key","model_display","run_key","run_display","k","metric","mean_score","method_label","reasoning_state"]
DF_AGG_DELTA_COLUMNS = ["model_key","model_display","k","metric","delta","method_label"]
METRICS: List[str] = ["laCOMET", "COMET", "BLEU_corpus", "BLEU_sentavg", "chrF++"]

bleu = BLEU(tokenize="flores200"); bleu_sent = BLEU(tokenize="flores200", effective_order=True); chrfpp = CHRF(word_order=2)

# ─────────────────────────────────────────────────────────────────────────────
# Global aesthetics  ── all text bumped ~45 % for paper-ready readability ──
# ─────────────────────────────────────────────────────────────────────────────
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
    "lines.linewidth":    3.4,
    "lines.markersize":   13,
    "axes.titlepad":      14,
    "axes.labelpad":      10,
})
_TBL_HEADER_BG = "#2c3e50"; _TBL_HEADER_FG = "white"
_TBL_ROW_ODD = "#f0f4f8"; _TBL_ROW_EVEN = "white"; _TBL_ROWLBL_BG = "#dce8f5"

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def slugify(s):
    s = re.sub(r"\s+", "_", s.strip()); return re.sub(r"[^A-Za-z0-9_\-]+", "", s)

_PREFIX_DISPLAY = {c.split("_")[0].lower(): n for c, n in LANG_DISPLAY.items()}
def lang_display(fc): return LANG_DISPLAY.get(fc, _PREFIX_DISPLAY.get(fc.split("_")[0].lower(), fc))

def parse_model_size(dn):
    m = re.search(r"([\d]+(?:\.\d+)?)\s*[Bb]", dn); return float(m.group(1)) if m else 0.0

def _small_and_large_model_keys(threshold, restrict_to=None):
    small, large = [], []
    for mk in (restrict_to if restrict_to else list(MODELS.keys())):
        (small if parse_model_size(MODELS.get(mk, mk)) < threshold else large).append(mk)
    return small, large

def _resolve_root_dir(run, mk):
    rd = run["root_dir"]
    if isinstance(rd, str): return rd
    if isinstance(rd, dict): return rd.get(mk)
    return None

def direction_folder_name(s, t): return f"{s.split('_')[0].lower()}_to_{t.split('_')[0].lower()}"
def direction_display_name(s, t): return f"{lang_display(s)} to {lang_display(t)}"

def direction_display_from_folder(f):
    p = f.split("_to_")
    return f"{_PREFIX_DISPLAY.get(p[0],p[0].capitalize())} to {_PREFIX_DISPLAY.get(p[1],p[1].capitalize())}" if len(p)==2 else f

def _method_label_from_on_label(l): l2 = re.sub(r"\s*[Rr]easoning\s*[Oo]n\s*$","",l).strip(); return l2 or l
def _method_label_from_off_label(l): l2 = re.sub(r"\s*[Rr]easoning\s*[Oo]ff\s*$","",l).strip(); return l2 or l
def metric_precision(m): return 4 if m in {"laCOMET","COMET"} else 2

def format_metric_value(m, v):
    if v is None: return "—"
    try:
        if math.isnan(float(v)): return "—"
    except: return "—"
    return f"{float(v):.{metric_precision(m)}f}"


def _build_k0_fallback_map(generation_runs):
    """For each non-random run, map its key -> the random run with the same
    reasoning state. Used only as a k=0 fallback, since k=0 is method-agnostic
    (zero in-context examples means the prompt is identical regardless of
    selection method). Returns {non_random_run_key: random_run_dict}."""
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

# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
def read_jsonl_translations(path):
    preds = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: preds.append(json.loads(line)["translation"])
    return preds

def load_flores_devtest(sl, tl):
    """Load FLORES-200 devtest sentences for one source / one target language
    from the Muennighoff/flores200 mirror (same schema as facebook/flores —
    `sentence` field, `devtest` split — but actively maintained on HF Hub
    and free of the cp1252/extract-archive issues the legacy facebook/flores
    loader hits on Colab + datasets>=3.0)."""
    ds = load_dataset("Muennighoff/flores200", sl, trust_remote_code=True)
    dt = load_dataset("Muennighoff/flores200", tl, trust_remote_code=True)
    return [e["sentence"] for e in ds["devtest"]], [e["sentence"] for e in dt["devtest"]]

def load_reference_sentences(sl, tl):
    """Dataset-arm dispatch: FLORES devtest, or the fixed 100-sentence
    WMT24++ test split (identical to what generation translated — the
    committed data/wmt24pp_split.json ordering)."""
    if DATASET == "wmt24pp":
        from src.data.wmt24pp import load_wmt24pp_sentences
        return load_wmt24pp_sentences(sl)["devtest"], load_wmt24pp_sentences(tl)["devtest"]
    return load_flores_devtest(sl, tl)

def load_comet_and_identifier():
    mp = download_model(COMET_MODEL_NAME); cm = load_from_checkpoint(mp)
    ip = hf_hub_download(repo_id=FASTTEXT_LID_REPO, filename=FASTTEXT_LID_FILENAME)
    return cm, fasttext.load_model(ip)

def _apply_limit(n, lm):
    if lm is None: return n
    m = int(lm); return 0 if m<=0 else min(n,m)

def _list_immediate_subdirs(p):
    try: return sorted([d for d in os.listdir(p) if os.path.isdir(os.path.join(p,d))])
    except FileNotFoundError: return []

def resolve_model_dirname(bd, mk):
    if os.path.isdir(os.path.join(bd,mk)): return mk
    nm = mk.lower(); ms = [d for d in _list_immediate_subdirs(bd) if d.lower()==nm]
    if len(ms)==1: return ms[0]
    if len(ms)>1: return sorted(ms)[0]
    return None

def build_translation_path(bd, md, dr, k, fp): return os.path.join(bd, md, dr, fp.format(K=k))


def print_file_structure(all_runs):
    """Print resolved root + model directories and a per-(run, model)
    summary of how many of the configured (direction × k) translation
    files actually exist on disk. Runs unconditionally at startup so the
    user can sanity-check folder layout even when the cache short-circuits
    the main evaluation loop.
    """
    print("\n[File structure resolution & on-disk check]")
    print(f"  Expected per (run, model): {len(SRC_LANGS) * len(TGT_LANGS)} directions × {len(K_LIST)} k-values "
          f"= {len(SRC_LANGS) * len(TGT_LANGS) * len(K_LIST)} files\n")
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
                    for k in K_LIST:
                        n_expected += 1
                        path = build_translation_path(root, dn, dr, k, run["filename_pattern"])
                        if os.path.exists(path):
                            n_found += 1
            marker = "✓" if n_found == n_expected else ("·" if n_found > 0 else "✗")
            print(f"    {marker} {mk:<16} → {root}/{dn}  ({n_found}/{n_expected} files)")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Valid-sentence filtering
# ─────────────────────────────────────────────────────────────────────────────
def _is_empty_translation(t): return t is None or t == ""

def find_valid_indices(all_pred_lists, ceiling):
    if not all_pred_lists: return list(range(ceiling))
    valid = set(range(ceiling))
    for preds in all_pred_lists:
        for i in range(ceiling):
            if i >= len(preds) or _is_empty_translation(preds[i]): valid.discard(i)
    return sorted(valid)

# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_comet(cm, src, tgt, pred, bs, gpus, lm=None):
    n = _apply_limit(min(len(src),len(tgt),len(pred)),lm)
    if n==0: return float("nan")
    d = [{"src":src[i],"mt":pred[i],"ref":tgt[i]} for i in range(n)]
    return float(np.mean(np.asarray(cm.predict(d,batch_size=bs,gpus=gpus).scores)[:n]))

def compute_lacomet(cm, ident, src, tgt, pred, tlk, bs, gpus, lm=None):
    n = _apply_limit(min(len(src),len(tgt),len(pred)),lm)
    if n==0: return float("nan")
    src,tgt,pred = src[:n],tgt[:n],pred[:n]
    mask = np.zeros(n,dtype=np.float32)
    for i,p in enumerate(pred):
        if EMPTY_AS_ZERO and not p.strip(): continue
        lb,_ = ident.predict(p.split("\n")[0])
        lb = lb[0] if isinstance(lb,(list,tuple)) else lb
        mask[i] = 1.0 if tlk in lb else 0.0
    d = [{"src":src[i],"mt":pred[i],"ref":tgt[i]} for i in range(n)]
    return float(np.mean(np.asarray(cm.predict(d,batch_size=bs,gpus=gpus).scores)[:n]*mask))

def compute_bleu_corpus_and_sentavg(tgt, pred, lm=None):
    n = _apply_limit(min(len(tgt),len(pred)),lm)
    if n==0: return float("nan"),float("nan")
    c = float(bleu.corpus_score(pred[:n],[tgt[:n]]).score)
    ss = [float(bleu_sent.sentence_score(pred[i],[tgt[i]]).score) for i in range(n)]
    return c, float(np.mean(ss)) if ss else float("nan")

def compute_chrfpp(tgt, pred, lm=None):
    n = _apply_limit(min(len(tgt),len(pred)),lm)
    if n==0: return float("nan")
    return float(chrfpp.corpus_score(pred[:n],[tgt[:n]]).score)

def compute_metric_bundle(cm, ident, src, tgt, pred, tl, bs, gpus, lm):
    lac = compute_lacomet(cm,ident,src,tgt,pred,tl,bs,gpus,lm)
    cv = compute_comet(cm,src,tgt,pred,bs,gpus,lm)
    bc,bs2 = compute_bleu_corpus_and_sentavg(tgt,pred,lm)
    ch = compute_chrfpp(tgt,pred,lm)
    return {"laCOMET":lac,"COMET":cv,"BLEU_corpus":bc,"BLEU_sentavg":bs2,"chrF++":ch}

# ─────────────────────────────────────────────────────────────────────────────
# Data accumulation
# ─────────────────────────────────────────────────────────────────────────────
def append_result_rows(rows, mk, md, rk, rd, sl, tl, dr, k, ms, path):
    for mn in METRICS:
        rows.append({"model_key":mk,"model_display":md,"run_key":rk,"run_display":rd,
                      "src_lang":sl,"tgt_lang":tl,"direction":dr,"k":k,"metric":mn,
                      "score":float(ms.get(mn,float("nan"))),"path":path})

def evaluate_all(generation_runs, src_langs, tgt_langs, k_list, eval_first_m, comet_batch_size, comet_gpus):
    comet_model, identifier = load_comet_and_identifier()
    rows = []; flores_cache = {}
    resolved = {}
    for run in generation_runs:
        for mk in MODELS:
            root = _resolve_root_dir(run,mk)
            if root is None: resolved[(run["key"],mk)]=None; continue
            dn = resolve_model_dirname(root,mk)
            resolved[(run["key"],mk)] = (root,dn) if dn else None
    print("\n[Model directory resolution]")
    for run in generation_runs:
        print(f"  {run['key']}:")
        for mk in MODELS:
            res = resolved.get((run["key"],mk))
            if res is None: print(f"    • {mk} → NOT FOUND")
            else: print(f"    • {mk} → {res[0]}/{res[1]}")
    print()

    k0_fallback = _build_k0_fallback_map(generation_runs)
    if k0_fallback:
        print("[k=0 fallback map]")
        for src_key, fb_run in k0_fallback.items():
            print(f"  • {src_key} → {fb_run['key']}  (only when k=0 file missing)")
        print()

    for sl in src_langs:
        for tl in tgt_langs:
            dr = direction_folder_name(sl,tl); dd = direction_display_from_folder(dr)
            if (sl,tl) not in flores_cache: flores_cache[(sl,tl)] = load_reference_sentences(sl,tl)
            sf,tf = flores_cache[(sl,tl)]
            ceil = _apply_limit(len(sf),eval_first_m); sc=sf[:ceil]; tc=tf[:ceil]
            for mk,md in MODELS.items():
                fr = {}; pbk = {k:[] for k in k_list}
                for run in generation_runs:
                    rk=run["key"]; res=resolved.get((rk,mk))
                    for k in k_list:
                        entry={"path":"","preds":None}
                        if res is not None:
                            rd2,mdn=res; cand=build_translation_path(rd2,mdn,dr,k,run["filename_pattern"])
                            if os.path.exists(cand):
                                pf=read_jsonl_translations(cand); pc=pf[:ceil]
                                entry["path"]=cand; entry["preds"]=pc; pbk[k].append(pc)

                        if k == 0 and entry["preds"] is None and rk in k0_fallback:
                            fb_run = k0_fallback[rk]
                            fb_res = resolved.get((fb_run["key"], mk))
                            if fb_res is not None:
                                fb_rd, fb_mdn = fb_res
                                fb_cand = build_translation_path(
                                    fb_rd, fb_mdn, dr, k, fb_run["filename_pattern"]
                                )
                                if os.path.exists(fb_cand):
                                    pf = read_jsonl_translations(fb_cand)
                                    pc = pf[:ceil]
                                    entry["path"] = fb_cand
                                    entry["preds"] = pc
                                    pbk[k].append(pc)

                        fr[(rk,k)]=entry
                vibk = {k:find_valid_indices(pbk[k],ceil) for k in k_list}
                cod=os.path.join(PLOTS_DIR,slugify(mk),dr); ensure_dir(cod)
                print(f"\n[{md} | {dd}]  Valid sentences per k:")
                pkc = {}
                for k in k_list:
                    nv=len(vibk[k]); print(f"    k={k:2d}  →  {nv} / {ceil}"); pkc[str(k)]={"valid":nv,"total_considered":ceil}
                with open(os.path.join(cod,"valid_sentence_counts.json"),"w",encoding="utf-8") as fh:
                    json.dump({"model_key":mk,"model_display":md,"direction":dr,"direction_display":dd,"per_k":pkc},fh,indent=2)
                for run in generation_runs:
                    rk=run["key"]; rdsp=run["label"]
                    for k in k_list:
                        ms2={m:float("nan") for m in METRICS}; entry=fr[(rk,k)]
                        path=entry["path"]; pc=entry["preds"]; vi=vibk[k]; nv=len(vi)
                        if pc is not None and nv>0:
                            sf2=[sc[i] for i in vi]; tf2=[tc[i] for i in vi]; pf2=[pc[i] for i in vi]
                            ms2=compute_metric_bundle(comet_model,identifier,sf2,tf2,pf2,tl,comet_batch_size,comet_gpus,None)
                            print(f"  [{rdsp}] k={k} | laCOMET={ms2['laCOMET']:.4f} | COMET={ms2['COMET']:.4f}"
                                  f" | BLEU={ms2['BLEU_corpus']:.2f} | BLEU_sentavg={ms2['BLEU_sentavg']:.2f} | chrF++={ms2['chrF++']:.2f}")
                        if path or FILL_MISSING_AS_NAN:
                            append_result_rows(rows,mk,md,rk,rdsp,sl,tl,dr,k,ms2,path)
    return pd.DataFrame(rows, columns=DF_LONG_COLUMNS)

# ─────────────────────────────────────────────────────────────────────────────
# Delta / aggregated computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_delta_df(df, ron, roff):
    ic=["model_key","model_display","direction","k","metric"]; parts=[]
    for on,off in zip(ron,roff):
        os2=df[df["run_key"]==on["key"]].set_index(ic)["score"]
        fs=df[df["run_key"]==off["key"]].set_index(ic)["score"]
        ds=(os2-fs).dropna()
        if ds.empty: continue
        d=ds.reset_index(); d.columns=ic+["delta"]
        d["method_label"]=_method_label_from_on_label(on["label"]); d["on_key"]=on["key"]; d["off_key"]=off["key"]
        parts.append(d)
    return pd.concat(parts,ignore_index=True)[DF_DELTA_COLUMNS] if parts else pd.DataFrame(columns=DF_DELTA_COLUMNS)

def compute_aggregated_scores(df, ron, roff):
    gc=["model_key","model_display","run_key","run_display","k","metric"]
    da=df.groupby(gc,as_index=False)["score"].mean().rename(columns={"score":"mean_score"})
    ri={}
    for r in ron: ri[r["key"]]=(_method_label_from_on_label(r["label"]),"On")
    for r in roff: ri[r["key"]]=(_method_label_from_off_label(r["label"]),"Off")
    da["method_label"]=da["run_key"].map(lambda k:ri.get(k,("Unknown","Unknown"))[0])
    da["reasoning_state"]=da["run_key"].map(lambda k:ri.get(k,("Unknown","Unknown"))[1])
    return da[DF_AGG_COLUMNS]

def compute_aggregated_deltas(da, ron, roff):
    ic=["model_key","model_display","k","metric"]; parts=[]
    for on,off in zip(ron,roff):
        os2=da[da["run_key"]==on["key"]].set_index(ic)["mean_score"]
        fs=da[da["run_key"]==off["key"]].set_index(ic)["mean_score"]
        ds=(os2-fs).dropna()
        if ds.empty: continue
        d=ds.reset_index(); d.columns=ic+["delta"]; d["method_label"]=_method_label_from_on_label(on["label"])
        parts.append(d)
    return pd.concat(parts,ignore_index=True)[DF_AGG_DELTA_COLUMNS] if parts else pd.DataFrame(columns=DF_AGG_DELTA_COLUMNS)

# ─────────────────────────────────────────────────────────────────────────────
# Figure / legend helpers
# ─────────────────────────────────────────────────────────────────────────────
def _superplot_grid(n):
    nc = min(3,n); return math.ceil(n/nc), nc

def _place_figure_legend(fig, axes_flat, n_used, legend_handles, legend_labels,
                         title="Method", max_ncol=4, gap_width_scale=1.0):
    if not legend_handles or not legend_labels:
        return

    n_total = len(axes_flat)
    n_empty = n_total - n_used

    fig.canvas.draw()
    occ_bboxes = [axes_flat[i].get_position() for i in range(n_used)]

    if n_empty >= 1:
        empty_bboxes = [axes_flat[i].get_position() for i in range(n_used, n_total)]

        emp_x0 = min(b.x0 for b in empty_bboxes)
        emp_x1 = max(b.x1 for b in empty_bboxes)
        emp_y0 = min(b.y0 for b in empty_bboxes)
        emp_y1 = max(b.y1 for b in empty_bboxes)
        emp_w = emp_x1 - emp_x0
        emp_h = emp_y1 - emp_y0

        pad_x = emp_w * 0.05
        pad_y = emp_h * 0.07
        leg_x0 = emp_x0 + pad_x
        leg_w = emp_w - 2 * pad_x
        leg_y0 = emp_y0 + pad_y
        leg_h = emp_h - 2 * pad_y

        # The legend stays INSIDE the empty cell(s): a tall two-column box
        # with compact spacing (plus the label-wrapping pass below) fits the
        # gap, so it can never spill over neighbouring subplots. The old
        # gap_width_scale widening is therefore capped to the empty region —
        # callers still pass it, but it no longer grows past the gap.
        if gap_width_scale != 1.0:
            new_leg_w = min(leg_w * gap_width_scale, emp_w - 2 * pad_x)
            leg_x0 = max(emp_x0 + pad_x, (emp_x0 + emp_x1 - new_leg_w) / 2.0)
            leg_w = new_leg_w

        fig_w_in, fig_h_in = fig.get_size_inches()
        leg_w_in = leg_w * fig_w_in
        leg_h_in = leg_h * fig_h_in
        aspect = leg_w_in / leg_h_in if leg_h_in > 0 else 1.0
        n_entries = len(legend_labels)
        if n_entries >= 5:
            # Many entries: tall-and-narrow beats wide — two columns.
            ncol = 2 if aspect >= 0.8 else 1
        elif aspect >= 1.4:
            ncol = min(n_entries, max_ncol)
        else:
            ncol = 1

        leg = fig.legend(
            legend_handles, legend_labels,
            loc="center",
            ncol=ncol,
            bbox_to_anchor=(leg_x0, leg_y0, leg_w, leg_h),
            bbox_transform=fig.transFigure,
            frameon=True, framealpha=0.97, edgecolor="#999999",
            fontsize=19, title=title, title_fontsize=21,
            borderpad=1.0, labelspacing=1.05,
            handlelength=2.2, handleheight=1.5,
            handletextpad=0.8, columnspacing=1.5,
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
                        best = i
                        break
                if best == -1:
                    for i in range(mid, len(label)):
                        if label[i] == " ":
                            best = i
                            break
                if best != -1:
                    txt.set_text(label[:best] + "\n" + label[best + 1:])

    else:
        all_x0 = min(b.x0 for b in occ_bboxes)
        all_x1 = max(b.x1 for b in occ_bboxes)
        span_w = all_x1 - all_x0
        single_w = occ_bboxes[0].x1 - occ_bboxes[0].x0
        # Narrower anchor box; with mode="expand" removed below, the legend
        # sizes itself to its content instead of stretching to fill this box.
        leg_w = max(span_w * 0.34, single_w * 0.9)
        leg_cx = (all_x0 + all_x1) / 2.0
        leg_x0 = leg_cx - leg_w / 2.0

        row_bottoms = sorted({round(b.y0, 4) for b in occ_bboxes}, reverse=True)
        lowest_y0 = min(b.y0 for b in occ_bboxes)

        if len(row_bottoms) >= 2:
            row_gap = row_bottoms[-2] - row_bottoms[-1]
            spacing = row_gap * 0.20
            leg_h = row_gap * 0.60
        else:
            subplot_h = max(b.y1 for b in occ_bboxes) - lowest_y0
            spacing = subplot_h * 0.24
            leg_h = subplot_h * 0.40

        leg_top = lowest_y0 - spacing
        leg_y = leg_top - leg_h

        ncol = min(len(legend_labels), max_ncol)
        leg = fig.legend(
            legend_handles, legend_labels,
            loc="upper center",
            ncol=ncol,
            bbox_to_anchor=(leg_x0, leg_y, leg_w, leg_h),
            bbox_transform=fig.transFigure,
            frameon=True, framealpha=0.97, edgecolor="#999999",
            fontsize=19, title=title, title_fontsize=21,
            borderpad=1.1, labelspacing=1.0,
            handlelength=2.4, handleheight=1.6,
            handletextpad=0.85, columnspacing=1.6,
        )
        leg.get_title().set_fontweight("bold")


def _attach_outside_legend(fig, hax, labels=None, ncol=4, title=None):
    if isinstance(hax, plt.Axes):
        handles, labels = hax.get_legend_handles_labels()
        labels = [l for l in labels if l not in ("run_display","method_label")]
    else: handles = hax
    if not handles: return
    kw = dict(loc="lower center", ncol=min(len(labels),ncol), frameon=True,
              framealpha=0.95, edgecolor="#cccccc", bbox_to_anchor=(0.5,-0.10),
              fontsize=19, title_fontsize=21,
              borderpad=1.2, labelspacing=1.2,
              handlelength=3.0, handletextpad=1.0, columnspacing=2.2)
    if title: kw["title"]=title
    fig.legend(handles, labels, **kw)


def _professional_table_axes(ax, pivot, metric, title):
    ax.axis("off")
    if pivot.empty or pivot.isna().all().all():
        ax.text(0.5,0.5,"No data available",ha="center",va="center",
                fontsize=18,style="italic",color="#555555",transform=ax.transAxes)
        ax.set_title(title,fontsize=24,fontweight="bold",pad=18); return
    rl=[str(r) for r in pivot.index]; cl=[f"k = {c}" for c in pivot.columns]
    ct=[[format_metric_value(metric,pivot.loc[r,c]) for c in pivot.columns] for r in pivot.index]
    nr,nc2=len(rl),len(cl)
    tbl=ax.table(cellText=ct,rowLabels=rl,colLabels=cl,loc="center",cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(19); tbl.scale(1.2,3.0)
    for j in range(nc2):
        cell=tbl[0,j]; cell.set_facecolor(_TBL_HEADER_BG); cell.set_text_props(color=_TBL_HEADER_FG,fontweight="bold"); cell.set_edgecolor("white")
    for i in range(nr):
        rb=_TBL_ROW_ODD if i%2==0 else _TBL_ROW_EVEN
        lc=tbl[i+1,-1]; lc.set_facecolor(_TBL_ROWLBL_BG); lc.set_text_props(fontweight="bold"); lc.set_edgecolor("#cccccc")
        for j in range(nc2): cell=tbl[i+1,j]; cell.set_facecolor(rb); cell.set_edgecolor("#cccccc")
    ax.set_title(title,fontsize=24,fontweight="bold",pad=18)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-metric Spearman correlation matrices  (raw scores + deltas, k>0 only)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_spearman_correlation_matrix(df_wide, metrics):
    """Compute pairwise Spearman ρ and p-values across rows of `df_wide`,
    using only rows where every listed metric column is non-NaN."""
    clean = df_wide[metrics].dropna()
    n = len(metrics)
    rho = np.full((n, n), np.nan, dtype=np.float64)
    pval = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        rho[i, i] = 1.0
        pval[i, i] = 0.0
        for j in range(i + 1, n):
            if len(clean) < 3:
                continue
            r, p = spearmanr(clean[metrics[i]], clean[metrics[j]])
            rho[i, j] = rho[j, i] = float(r)
            pval[i, j] = pval[j, i] = float(p)
    rho_df = pd.DataFrame(rho, index=metrics, columns=metrics)
    pval_df = pd.DataFrame(pval, index=metrics, columns=metrics)
    return rho_df, pval_df, len(clean)


def _save_correlation_heatmap(rho_df, pval_df, n_cells, title, out_path):
    """Render a Spearman-ρ correlation matrix as a paper-ready heatmap PNG.

    Diverging RdBu_r colormap centred at 0; cells annotated with ρ to three
    decimals plus a star marker for the p-value (`*` < 0.05, `**` < 0.01,
    `***` < 0.001). Footer records `n`."""
    from matplotlib.colors import LinearSegmentedColormap

    ensure_dir(os.path.dirname(out_path))
    metrics = list(rho_df.index)
    n_m = len(metrics)

    fig, ax = plt.subplots(figsize=(max(9.0, 1.8 * n_m + 4.5), max(8.0, 1.6 * n_m + 4.0)))
    base = plt.get_cmap("RdBu_r")
    cm = LinearSegmentedColormap.from_list("RdBu_r_clipped", base(np.linspace(0.05, 0.95, 256)))
    cm.set_bad(color="lightgray")

    mat = rho_df.values.astype(np.float64)
    masked = np.ma.array(mat, mask=np.isnan(mat))
    im = ax.imshow(masked, cmap=cm, vmin=-1.0, vmax=1.0)

    ax.set_xticks(range(n_m))
    ax.set_yticks(range(n_m))
    ax.set_xticklabels(metrics, rotation=35, ha="right", fontsize=20)
    ax.set_yticklabels(metrics, fontsize=20)
    ax.set_title(title, fontsize=24, fontweight="bold", pad=18)

    for i in range(n_m):
        for j in range(n_m):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=19, color="black"); continue
            p = float(pval_df.iat[i, j])
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            shade = abs(v)
            txt_color = "white" if shade > 0.55 else "black"
            ax.text(j, i, f"{v:+.3f}{star}", ha="center", va="center",
                    fontsize=21, fontweight="bold", color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", fontsize=22, labelpad=10)
    cbar.ax.tick_params(labelsize=18)

    fig.text(0.5, 0.02, f"n = {n_cells:,} cells   (* p<0.05, ** p<0.01, *** p<0.001)",
             ha="center", fontsize=18, style="italic")

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def generate_correlation_matrices(df, dfd, out_root, metrics):
    """Compute and persist two cross-metric Spearman correlation matrices
    plus their heatmaps:

      • spearman_raw_scores_*.{csv,png}
            One row per (model, direction, k>0, run_key) cell.
            Answers: "do the metrics agree on which configurations produce
            higher-quality translations?"

      • spearman_deltas_*.{csv,png}
            One row per (model, direction, k>0, method) cell with
            value = score reasoning ON − score reasoning OFF.
            Answers: "do the metrics agree on the sign and magnitude of
            the reasoning effect at each operating point?"

    `k=0` rows are excluded from both matrices."""
    cor_dir = os.path.join(out_root, "metric_correlations")
    ensure_dir(cor_dir)

    # ── Matrix 1: raw per-config scores, k > 0 ────────────────────────────
    raw = df[df["k"] > 0].copy()
    raw_wide = raw.pivot_table(
        index=["model_key", "direction", "k", "run_key"],
        columns="metric",
        values="score",
        aggfunc="first",
    ).reindex(columns=metrics)

    raw_rho, raw_pval, raw_n = _compute_spearman_correlation_matrix(raw_wide, metrics)
    raw_rho.to_csv(os.path.join(cor_dir, "spearman_raw_scores_rho.csv"))
    raw_pval.to_csv(os.path.join(cor_dir, "spearman_raw_scores_pvalue.csv"))
    _save_correlation_heatmap(
        raw_rho, raw_pval, raw_n,
        "Spearman Correlation of Raw Metric Scores\n"
        "(per (model × direction × k>0 × run) cell)",
        os.path.join(cor_dir, "spearman_raw_scores_heatmap.png"),
    )

    # ── Matrix 2: reasoning-on minus reasoning-off deltas, k > 0 ──────────
    delta = dfd[dfd["k"] > 0].copy()
    delta_wide = delta.pivot_table(
        index=["model_key", "direction", "k", "method_label"],
        columns="metric",
        values="delta",
        aggfunc="first",
    ).reindex(columns=metrics)

    delta_rho, delta_pval, delta_n = _compute_spearman_correlation_matrix(delta_wide, metrics)
    delta_rho.to_csv(os.path.join(cor_dir, "spearman_deltas_rho.csv"))
    delta_pval.to_csv(os.path.join(cor_dir, "spearman_deltas_pvalue.csv"))
    _save_correlation_heatmap(
        delta_rho, delta_pval, delta_n,
        "Spearman Correlation of Reasoning On − Off Deltas\n"
        "(per (model × direction × k>0 × method) cell)",
        os.path.join(cor_dir, "spearman_deltas_heatmap.png"),
    )

    print(f"\n[Metric correlation matrices written]")
    print(f"  Raw scores: n={raw_n:,} cells   →  {cor_dir}/spearman_raw_scores_*")
    print(f"  Deltas:     n={delta_n:,} cells   →  {cor_dir}/spearman_deltas_*")


# ─────────────────────────────────────────────────────────────────────────────
# k=0 plot-filter helper  ── respects INCLUDE_K0_IN_PLOTS toggle ──
# ─────────────────────────────────────────────────────────────────────────────
def _apply_k_plot_filter(df_in):
    """Drop k=0 rows when INCLUDE_K0_IN_PLOTS is False; pass-through otherwise.
    Used by every plotting function so the toggle takes effect uniformly."""
    if INCLUDE_K0_IN_PLOTS:
        return df_in
    if "k" not in df_in.columns:
        return df_in
    return df_in[df_in["k"] > 0]


# ─────────────────────────────────────────────────────────────────────────────
# Per-direction artefacts
# ─────────────────────────────────────────────────────────────────────────────
def save_metric_plot(df, mk, dr, met, op):
    dp=df[(df["model_key"]==mk)&(df["direction"]==dr)&(df["metric"]==met)&(~df["score"].isna())].copy()
    dp = _apply_k_plot_filter(dp)
    md=MODELS.get(mk,mk); dd=direction_display_from_folder(dr)
    ensure_dir(os.path.dirname(op)); fig,ax=plt.subplots(figsize=(16,10))
    if dp.empty:
        ax.axis("off")
        ax.text(0.5,0.5,"No data found",ha="center",va="center",fontsize=23,style="italic")
    else:
        dp=dp.sort_values(["run_display","k"],kind="stable")
        sns.lineplot(data=dp,x="k",y="score",hue="run_display",style="run_display",
                     markers=True,dashes=False,linewidth=3.6,markersize=13,ax=ax)
        ax.set_xticks(K_LIST_PLOT)
        ax.set_xlabel("Number of Examples (k)",fontsize=22,labelpad=12)
        ax.set_ylabel(met,fontsize=22,labelpad=12)
        ax.grid(True,linestyle="--",alpha=0.45); ax.tick_params(axis="both",labelsize=19)
        h,l=ax.get_legend_handles_labels(); l=[x for x in l if x!="run_display"]
        if ax.legend_: ax.legend_.remove()
        _attach_outside_legend(fig,h,l,ncol=4,title="Method")
    ax.set_title(f"{md} — {dd} — {met}",fontsize=26,fontweight="bold",pad=18)
    fig.tight_layout(); fig.savefig(op,dpi=250,bbox_inches="tight"); plt.close(fig)

def save_metric_table(df, mk, dr, met, op):
    dt=df[(df["model_key"]==mk)&(df["direction"]==dr)&(df["metric"]==met)].copy()
    dt = _apply_k_plot_filter(dt)
    piv=dt.pivot_table(index="run_display",columns="k",values="score",aggfunc="first").reindex(columns=K_LIST_PLOT)
    ro=list(dict.fromkeys(dt["run_display"].tolist()))
    if ro: piv=piv.reindex(index=ro)
    md=MODELS.get(mk,mk); dd=direction_display_from_folder(dr)
    nr=max(1,len(piv.index)); nc=max(1,len(piv.columns))
    ensure_dir(os.path.dirname(op))
    fig,ax=plt.subplots(figsize=(max(13.0, 2.6*nc + 6.0), max(5.5, 1.2*nr + 4.0)))
    _professional_table_axes(ax,piv,met,f"{md} — {dd} — {met}")
    fig.tight_layout(); fig.savefig(op,dpi=250,bbox_inches="tight"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Super-plots (per-model, one subplot per language)
# ─────────────────────────────────────────────────────────────────────────────
def save_model_metric_superplot(df, mk, met, dord, op):
    ensure_dir(os.path.dirname(op))
    ds=df[(df["model_key"]==mk)&(df["metric"]==met)].copy(); md=MODELS.get(mk,mk)
    ds = _apply_k_plot_filter(ds)
    avail=[d for d in dord if d in set(ds["direction"].tolist())]
    if not avail: avail=sorted(ds["direction"].dropna().unique().tolist())
    n=len(avail)
    if n==0:
        fig,ax=plt.subplots(figsize=(9,5)); ax.axis("off")
        ax.text(0.5,0.5,"No data found",ha="center",va="center",fontsize=22)
        fig.tight_layout(); fig.savefig(op,dpi=250,bbox_inches="tight"); plt.close(fig); return
    nr,nc=_superplot_grid(n)
    fig,axes=plt.subplots(nr,nc,figsize=(12*nc,8.5*nr),squeeze=False)
    af=axes.flatten(); lh=ll=None
    for ax,d in zip(af,avail):
        dd=direction_display_from_folder(d); df2=ds[(ds["direction"]==d)&(~ds["score"].isna())].copy()
        ax.set_title(dd,fontsize=25,fontweight="bold",pad=14)
        ax.set_xlabel("Number of Examples (k)",fontsize=22,labelpad=12)
        ax.set_ylabel(met,fontsize=22,labelpad=12)
        ax.set_xticks(K_LIST_PLOT); ax.tick_params(axis="both",labelsize=19); ax.grid(True,linestyle="--",alpha=0.45)
        if df2.empty:
            ax.text(0.5,0.5,"No data found",ha="center",va="center",fontsize=19,style="italic",transform=ax.transAxes); continue
        sns.lineplot(data=df2.sort_values(["run_display","k"],kind="stable"),x="k",y="score",
                     hue="run_display",style="run_display",markers=True,dashes=False,
                     linewidth=3.2,markersize=13,ax=ax,legend=True)
        lh,ll=_collect_legend(ax,lh,ll)
        if ax.legend_: ax.legend_.remove()
    for ax in af[n:]: ax.axis("off")
    fig.suptitle(f"{md} — {met} Across Translation Directions",fontsize=30,fontweight="bold",y=0.995)
    fig.tight_layout(rect=[0,0,1,0.955])
    _place_figure_legend(fig,af,n,lh,ll,title="Method", gap_width_scale=5.0)
    fig.savefig(op,dpi=250,bbox_inches="tight"); plt.close(fig)

def save_delta_superplot(dfd, mk, met, dord, op):
    ensure_dir(os.path.dirname(op))
    ds=dfd[(dfd["model_key"]==mk)&(dfd["metric"]==met)].copy(); md=MODELS.get(mk,mk)
    ds = _apply_k_plot_filter(ds)
    avail=[d for d in dord if d in set(ds["direction"].tolist())]
    if not avail: avail=sorted(ds["direction"].dropna().unique().tolist())
    n=len(avail)
    if n==0:
        fig,ax=plt.subplots(figsize=(9,5)); ax.axis("off")
        ax.text(0.5,0.5,"No data found",ha="center",va="center",fontsize=22)
        fig.tight_layout(); fig.savefig(op,dpi=250,bbox_inches="tight"); plt.close(fig); return
    nr,nc=_superplot_grid(n)
    fig,axes=plt.subplots(nr,nc,figsize=(12*nc,8.5*nr),squeeze=False)
    af=axes.flatten(); lh=ll=None
    for ax,d in zip(af,avail):
        dd=direction_display_from_folder(d); df2=ds[ds["direction"]==d].copy()
        ax.set_title(dd,fontsize=25,fontweight="bold",pad=14)
        ax.set_xlabel("Number of Examples (k)",fontsize=22,labelpad=12)
        ax.set_ylabel(f"Δ {met} (Reasoning On − Off)",fontsize=22,labelpad=12)
        ax.set_xticks(K_LIST_PLOT); ax.tick_params(axis="both",labelsize=19)
        ax.axhline(0,color="#777777",linewidth=1.6,linestyle="--",zorder=1); ax.grid(True,linestyle="--",alpha=0.4)
        if df2.empty:
            ax.text(0.5,0.5,"No data found",ha="center",va="center",fontsize=19,style="italic",transform=ax.transAxes); continue
        sns.lineplot(data=df2.sort_values(["method_label","k"],kind="stable"),x="k",y="delta",
                     hue="method_label",style="method_label",markers=True,dashes=False,
                     linewidth=3.2,markersize=13,ax=ax,legend=True)
        lh,ll=_collect_legend(ax,lh,ll)
        if ax.legend_: ax.legend_.remove()
    for ax in af[n:]: ax.axis("off")
    fig.suptitle(f"{md} — Δ {met}: Reasoning On vs. Reasoning Off",fontsize=30,fontweight="bold",y=0.995)
    fig.tight_layout(rect=[0,0,1,0.955])
    _place_figure_legend(fig,af,n,lh,ll,title="Method", gap_width_scale=2.5)
    fig.savefig(op,dpi=250,bbox_inches="tight"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated cross-language superplots
# ─────────────────────────────────────────────────────────────────────────────
def save_aggregated_delta_superplot(dad, met, mord, op):
    ensure_dir(os.path.dirname(op)); ds=dad[dad["metric"]==met].copy()
    ds = _apply_k_plot_filter(ds)
    models=[m for m in mord if m in ds["model_key"].values]
    if not models: models=[m for m in mord if m in MODELS]
    n=len(models)
    if n==0:
        fig,ax=plt.subplots(figsize=(9,5)); ax.axis("off")
        ax.text(0.5,0.5,"No data found",ha="center",va="center",fontsize=22)
        fig.tight_layout(); fig.savefig(op,dpi=250,bbox_inches="tight"); plt.close(fig); return
    nr,nc=_superplot_grid(n)
    fig,axes=plt.subplots(nr,nc,figsize=(12*nc,8.5*nr),squeeze=False)
    af=axes.flatten(); lh=ll=None
    for ax,mk in zip(af,models):
        md=MODELS.get(mk,mk); dm=ds[ds["model_key"]==mk].copy()
        ax.set_title(md,fontsize=25,fontweight="bold",pad=14)
        ax.set_xlabel("Number of Examples (k)",fontsize=22,labelpad=12)
        ax.set_ylabel(f"Δ {met}",fontsize=22,labelpad=12)
        ax.set_xticks(K_LIST_PLOT); ax.tick_params(axis="both",labelsize=19)
        ax.axhline(0,color="#777777",linewidth=1.6,linestyle="--",zorder=1); ax.grid(True,linestyle="--",alpha=0.4)
        if dm.empty:
            ax.text(0.5,0.5,"No data found",ha="center",va="center",fontsize=19,style="italic",transform=ax.transAxes); continue
        sns.lineplot(data=dm.sort_values(["method_label","k"],kind="stable"),x="k",y="delta",
                     hue="method_label",style="method_label",markers=True,dashes=False,
                     linewidth=3.2,markersize=13,ax=ax,legend=True)
        lh,ll=_collect_legend(ax,lh,ll)
        if ax.legend_: ax.legend_.remove()
    for ax in af[n:]: ax.axis("off")
    fig.suptitle(f"Aggregated Δ {met}: Reasoning On vs. Off (Averaged Across All Language Pairs)",
                 fontsize=30,fontweight="bold",y=0.995)
    fig.tight_layout(rect=[0,0,1,0.955])
    _place_figure_legend(fig,af,n,lh,ll,title="Method", max_ncol=4)
    fig.savefig(op,dpi=250,bbox_inches="tight"); plt.close(fig)

def save_aggregated_scores_superplot(da, met, mord, op):
    ensure_dir(os.path.dirname(op)); ds=da[da["metric"]==met].copy()
    ds = _apply_k_plot_filter(ds)
    ds["line_label"]=ds["method_label"]+" – Reasoning "+ds["reasoning_state"]
    models=[m for m in mord if m in ds["model_key"].values]
    if not models: models=[m for m in mord if m in MODELS]
    n=len(models)
    if n==0:
        fig,ax=plt.subplots(figsize=(9,5)); ax.axis("off")
        ax.text(0.5,0.5,"No data found",ha="center",va="center",fontsize=22)
        fig.tight_layout(); fig.savefig(op,dpi=250,bbox_inches="tight"); plt.close(fig); return
    nr,nc=_superplot_grid(n)
    fig,axes=plt.subplots(nr,nc,figsize=(12*nc,8.5*nr),squeeze=False)
    af=axes.flatten(); lh=ll=None; rdash={"On":"","Off":(4,2)}
    for ax,mk in zip(af,models):
        md=MODELS.get(mk,mk); dm=ds[ds["model_key"]==mk].copy()
        ax.set_title(md,fontsize=25,fontweight="bold",pad=14)
        ax.set_xlabel("Number of Examples (k)",fontsize=22,labelpad=12)
        ax.set_ylabel(met,fontsize=22,labelpad=12)
        ax.set_xticks(K_LIST_PLOT); ax.tick_params(axis="both",labelsize=19); ax.grid(True,linestyle="--",alpha=0.45)
        if dm.empty:
            ax.text(0.5,0.5,"No data found",ha="center",va="center",fontsize=19,style="italic",transform=ax.transAxes); continue
        sns.lineplot(data=dm.sort_values(["method_label","reasoning_state","k"],kind="stable"),
                     x="k",y="mean_score",hue="method_label",style="reasoning_state",style_order=["On","Off"],
                     markers=True,dashes=rdash,linewidth=3.2,markersize=13,ax=ax,legend=True)
        lh,ll=_collect_legend(ax,lh,ll,ek={"method_label","reasoning_state"})
        if ax.legend_: ax.legend_.remove()
    for ax in af[n:]: ax.axis("off")
    fig.suptitle(f"Aggregated {met} Across All Language Pairs (Reasoning On vs. Off by Method)",
                 fontsize=30,fontweight="bold",y=0.995)
    fig.tight_layout(rect=[0,0,1,0.955])
    _place_figure_legend(fig,af,n,lh,ll,title="Method – Reasoning State", max_ncol=4)
    fig.savefig(op,dpi=250,bbox_inches="tight"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-model comparison superplots
# ─────────────────────────────────────────────────────────────────────────────
def save_cross_model_comparison_superplot(df, monr, moffr, met, dord, op, st, restrict_to=None, family_label=None):
    ensure_dir(os.path.dirname(op))
    sk,lk=_small_and_large_model_keys(st,restrict_to); onk=monr["key"]; offk=moffr["key"]
    parts=[]
    for mk in sk:
        s=df[(df["model_key"]==mk)&(df["run_key"]==onk)&(df["metric"]==met)&(~df["score"].isna())].copy()
        s["line_label"]=f"{MODELS[mk]} – Reasoning On"; parts.append(s)
    for mk in lk:
        s=df[(df["model_key"]==mk)&(df["run_key"]==offk)&(df["metric"]==met)&(~df["score"].isna())].copy()
        s["line_label"]=f"{MODELS[mk]} – Reasoning Off"; parts.append(s)
    dp=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    dp = _apply_k_plot_filter(dp)
    ml=_method_label_from_on_label(monr["label"]); fs=f" ({family_label})" if family_label else ""
    avail=[d for d in dord if not dp.empty and d in set(dp["direction"].tolist())]
    if not avail and not dp.empty: avail=sorted(dp["direction"].dropna().unique().tolist())
    n=len(avail)
    if n==0:
        fig,ax=plt.subplots(figsize=(9,5)); ax.axis("off")
        ax.text(0.5,0.5,"No data found",ha="center",va="center",fontsize=22)
        fig.tight_layout(); fig.savefig(op,dpi=250,bbox_inches="tight"); plt.close(fig); return
    nr,nc=_superplot_grid(n)
    fig,axes=plt.subplots(nr,nc,figsize=(12*nc,8.5*nr),squeeze=False)
    af=axes.flatten(); lh=ll=None
    for ax,d in zip(af,avail):
        dd=direction_display_from_folder(d); df2=dp[dp["direction"]==d].copy()
        ax.set_title(dd,fontsize=25,fontweight="bold",pad=14)
        ax.set_xlabel("Number of Examples (k)",fontsize=22,labelpad=12)
        ax.set_ylabel(met,fontsize=22,labelpad=12)
        ax.set_xticks(K_LIST_PLOT); ax.tick_params(axis="both",labelsize=19); ax.grid(True,linestyle="--",alpha=0.45)
        if df2.empty:
            ax.text(0.5,0.5,"No data found",ha="center",va="center",fontsize=19,style="italic",transform=ax.transAxes); continue
        sns.lineplot(data=df2.sort_values(["line_label","k"],kind="stable"),x="k",y="score",
                     hue="line_label",style="line_label",markers=True,dashes=False,
                     linewidth=3.2,markersize=13,ax=ax,legend=True)
        lh,ll=_collect_legend(ax,lh,ll)
        if ax.legend_: ax.legend_.remove()
    for ax in af[n:]: ax.axis("off")
    fig.suptitle(f"{ml} — {met}: Small Models (Reasoning On) vs. Large Models (Reasoning Off){fs}",
                 fontsize=30,fontweight="bold",y=0.995)
    fig.tight_layout(rect=[0,0,1,0.955])
    _place_figure_legend(fig,af,n,lh,ll,title=f"Model – Reasoning State  (threshold: {st}B)")
    fig.savefig(op,dpi=250,bbox_inches="tight"); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────
def export_direction_raw_files(df, mk, dr, od):
    ensure_dir(od)
    ds=df[(df["model_key"]==mk)&(df["direction"]==dr)].copy().sort_values(["run_key","metric","k"],kind="stable")
    ds.to_csv(os.path.join(od,"raw_scores_long.csv"),index=False)
    ds.to_excel(os.path.join(od,"raw_scores_long.xlsx"),index=False)
    piv=ds.pivot_table(index=["run_key","run_display","k"],columns="metric",values="score",aggfunc="first").sort_index()
    piv.to_excel(os.path.join(od,"raw_scores_pivot.xlsx"))

def export_global_tables(df, od):
    ensure_dir(od); df=df.reindex(columns=DF_LONG_COLUMNS)
    ds=df.sort_values(["model_key","direction","run_key","k","metric"],kind="stable")
    ds.to_csv(os.path.join(od,"raw_scores_long.csv"),index=False)
    ds.to_excel(os.path.join(od,"raw_scores_long.xlsx"),index=False)
    with pd.ExcelWriter(os.path.join(od,"raw_scores_by_model.xlsx"),engine="openpyxl") as w:
        for mk,dm in ds.groupby("model_key",sort=True):
            piv=dm.pivot_table(index=["direction","run_key","run_display","k"],columns="metric",values="score",aggfunc="first").sort_index()
            piv.to_excel(w,sheet_name=slugify(mk)[:31])
        c=ds.pivot_table(index=["model_key","model_display","direction","run_key","run_display","k"],columns="metric",values="score",aggfunc="first").sort_index()
        c.to_excel(w,sheet_name="ALL_models_compact")


# ─────────────────────────────────────────────────────────────────────────────
# Master output orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_outputs(df, dfd, da, dad, out_root, src_langs, tgt_langs):
    dord=[direction_folder_name(s,t) for s in src_langs for t in tgt_langs]
    for mk in MODELS:
        mr=os.path.join(out_root,slugify(mk)); ensure_dir(mr)
        for sl in src_langs:
            for tl in tgt_langs:
                dr=direction_folder_name(sl,tl); fo=os.path.join(mr,dr); ensure_dir(fo)
                export_direction_raw_files(df,mk,dr,fo)
                for met in METRICS:
                    ms=slugify(met)
                    save_metric_plot(df,mk,dr,met,os.path.join(fo,f"plot_{ms}.png"))
                    save_metric_table(df,mk,dr,met,os.path.join(fo,f"table_{ms}.png"))
        for met in METRICS:
            ms=slugify(met)
            save_model_metric_superplot(df,mk,met,dord,os.path.join(mr,f"superplot_{ms}.png"))
            if not dfd.empty: save_delta_superplot(dfd,mk,met,dord,os.path.join(mr,f"delta_superplot_{ms}.png"))

    cr=os.path.join(out_root,"cross_model_comparison"); ensure_dir(cr)
    for onr,offr in zip(GENERATION_RUNS_REASONING_ON,GENERATION_RUNS_REASONING_OFF):
        ml=_method_label_from_on_label(onr["label"]); md2=os.path.join(cr,slugify(ml)); ensure_dir(md2)
        for met in METRICS:
            ms=slugify(met)
            save_cross_model_comparison_superplot(df,onr,offr,met,dord,os.path.join(md2,f"superplot_{ms}.png"),MODEL_SIZE_THRESHOLD)
            for fn,fk in MODEL_FAMILIES.items():
                fd=os.path.join(md2,slugify(fn)); ensure_dir(fd)
                save_cross_model_comparison_superplot(df,onr,offr,met,dord,os.path.join(fd,f"superplot_{ms}.png"),MODEL_SIZE_THRESHOLD,restrict_to=fk,family_label=fn)

    ar=os.path.join(out_root,"aggregated_cross_language"); ensure_dir(ar)
    for met in METRICS:
        ms=slugify(met)
        if not dad.empty: save_aggregated_delta_superplot(dad,met,MODEL_ORDER,os.path.join(ar,f"aggregated_delta_{ms}.png"))
        if not da.empty: save_aggregated_scores_superplot(da,met,MODEL_ORDER,os.path.join(ar,f"aggregated_scores_{ms}.png"))
    export_global_tables(df,out_root)

    # ── Cross-metric Spearman correlation matrices (raw + reasoning deltas,
    #    k > 0 only) — written under <out_root>/metric_correlations/ ──
    generate_correlation_matrices(df, dfd, out_root, METRICS)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ensure_dir(PLOTS_DIR)
    csv_path = os.path.join(PLOTS_DIR, "raw_scores_long.csv")
    all_runs = GENERATION_RUNS_REASONING_ON + GENERATION_RUNS_REASONING_OFF

    print_file_structure(all_runs)

    if LOAD_FROM_CSV and os.path.exists(csv_path):
        print(f"\n[LOAD_FROM_CSV] Reading existing cache: {csv_path}")
        df_cached = pd.read_csv(csv_path)
        df_cached["k"] = df_cached["k"].astype(int)
        df_cached["score"] = pd.to_numeric(df_cached["score"], errors="coerce")
        print(f"  Loaded {len(df_cached)} rows.")

        cached_run_keys = set(df_cached["run_key"].unique())
        all_run_keys = set(r["key"] for r in all_runs)
        missing_run_keys = all_run_keys - cached_run_keys
        unexpected_run_keys = cached_run_keys - all_run_keys

        if unexpected_run_keys:
            print(f"  [note] Cache contains {len(unexpected_run_keys)} run(s) not in current config; "
                  f"they will pass through unchanged: {sorted(unexpected_run_keys)}")

        if missing_run_keys:
            runs_to_compute = [r for r in all_runs if r["key"] in missing_run_keys]
            print(f"\n[Incremental compute] {len(runs_to_compute)} run(s) missing from cache:")
            for r in runs_to_compute:
                print(f"  • {r['key']:<32} ({r['label']})")
            print()

            df_new = evaluate_all(
                runs_to_compute, SRC_LANGS, TGT_LANGS, K_LIST,
                EVAL_FIRST_M, COMET_BATCH_SIZE, COMET_GPUS,
            )
            df = pd.concat([df_cached, df_new], ignore_index=True)
            df.to_csv(csv_path, index=False)
            print(f"\n[Cache updated] {len(df_cached)} cached + {len(df_new)} new = {len(df)} rows.")
            print(f"  Written to: {csv_path}")
        else:
            print("  All configured runs already cached; skipping the metric-computation step.")
            df = df_cached
    else:
        df = evaluate_all(
            all_runs, SRC_LANGS, TGT_LANGS, K_LIST,
            EVAL_FIRST_M, COMET_BATCH_SIZE, COMET_GPUS,
        )

    dfd = compute_delta_df(df, GENERATION_RUNS_REASONING_ON, GENERATION_RUNS_REASONING_OFF)
    da = compute_aggregated_scores(df, GENERATION_RUNS_REASONING_ON, GENERATION_RUNS_REASONING_OFF)
    dad = compute_aggregated_deltas(da, GENERATION_RUNS_REASONING_ON, GENERATION_RUNS_REASONING_OFF)
    generate_all_outputs(df, dfd, da, dad, PLOTS_DIR, SRC_LANGS, TGT_LANGS)
    print(f"\nAll artefacts saved to: {PLOTS_DIR}/\nDone.")

if __name__ == "__main__":
    main()