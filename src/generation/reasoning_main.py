"""
reasoning_main.py
─────────────────
Entrypoint for reasoning-model generation runs. Loads FLORES devtest, builds
few-shot demonstrations via the configured ensemble method (RRF / random /
sentinel / edit_dist), boots a vLLM server for each requested model in
sequence via `vllm_server.VLLMServerProcess`, and streams reasoning +
translation into per-direction JSONL files.

Inputs:  SRC_LANG_CODES, TGT_LANG_CODES, K_LIST, EMB_METHODS, ENSEMBLE_METHOD,
         PROMPT_HEADER, and the vLLM sampling knobs (MAX_NEW_TOKENS,
         TEMPERATURE, TOP_P, TOP_K, MIN_P) configured at the top of this file.
Outputs: JSONL translation files under GEN_ROOT and reasoning traces under
         REASONING_ROOT.
"""

# =========================
# Cell 3 — Reasoning models main (config + main only)
# =========================

import os
import torch
import numpy as np

from typing import Dict, List, Tuple, Optional

from openai import OpenAI

from src.retrieval.retrieval_helpers import (
    _apply_devtest_limit,
    _ensure_dir,
    _plot_matrix_png,
    build_fragmentshot_pools,
    compute_pairwise_overlap_and_meanrank,
    ensemble_topk_dispatch,
    load_flores_sentences,
    load_retrieval_embeddings,
    load_sentinel_src_scores,
    parse_ensemble_method,
    topk_cosine_indices_and_scores,
    topk_cosine_indices_from_pools,
)
from src.generation.vllm_server import (
    VLLMServerProcess,
    VLLMServerSpec,
    get_served_model_id,
    load_system_prompt,
    run_openai_for_direction_streaming,
)
from src.data.wmt24pp import WMT_TGT_LANGS, load_wmt24pp_sentences

HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

OPENAI_API_KEY = "EMPTY"

# ── Env-driven run configuration ─────────────────────────────────────────────
# Every knob a SLURM job needs is settable via RTRACE_* environment variables,
# with defaults preserving the original notebook values. One sbatch job =
# one (model, method, reasoning state, dataset) cell of the experiment grid.

# DATASET: "flores" (dev pool = 997, devtest test = first 100) or "wmt24pp"
# (fixed committed split: pool = 860, test = 100; see data/wmt24pp_split.json).
DATASET = os.environ.get("RTRACE_DATASET", "flores").lower()
if DATASET not in ("flores", "wmt24pp"):
    raise ValueError(f"RTRACE_DATASET must be 'flores' or 'wmt24pp', got {DATASET!r}")

# DATASET_LOADER: returns {"dev": [...], "devtest": [...]} for a lang code —
# identical contract for both datasets, so main() below is dataset-agnostic.
DATASET_LOADER = load_wmt24pp_sentences if DATASET == "wmt24pp" else load_flores_sentences

_DEFAULT_EMB_ROOT = "wmt24pp_embeddings" if DATASET == "wmt24pp" else "drive/MyDrive/flores_embeddings"
EMB_ROOT = os.environ.get("RTRACE_EMB_ROOT", _DEFAULT_EMB_ROOT)
BM25_METHOD_NAME = "bm25"

SRC_LANG_CODES: List[str] = ["eng_Latn"]

_FLORES_TGT_LANGS: List[str] = [
    "wol_Latn",
    "swh_Latn",
    "lus_Latn",
    "mni_Beng",
    "tel_Telu",
    "tam_Taml",
    "uzn_Latn",
]
_DEFAULT_TGTS = WMT_TGT_LANGS if DATASET == "wmt24pp" else _FLORES_TGT_LANGS
TGT_LANG_CODES: List[str] = [
    s.strip() for s in os.environ.get("RTRACE_TGT_LANGS", ",".join(_DEFAULT_TGTS)).split(",") if s.strip()
]

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

K_LIST = [int(k) for k in os.environ.get("RTRACE_K_LIST", "1,3,5,7,10").split(",")]
K_LIST = sorted(set(int(k) for k in K_LIST))
K_MAX = max(K_LIST) if K_LIST else 0

M_PER_MODEL = K_MAX

EMB_METHODS = ["cohere", "sonar", "labse", "MiniLM", "e5"]
# The SLURM scripts pass short method names (random, sentinel) but both the
# dispatcher (ensemble_topk_dispatch) and the eval pipeline's filename
# patterns (k{K}_random_pool_template11.jsonl, k{K}_pool_sentinel_src_rerank_
# template11.jsonl) speak the internal vocabulary — normalize here so the
# dispatch succeeds AND output filenames match what eval globs for.
_METHOD_ALIASES = {"random": "random_pool", "sentinel": "pool_sentinel_src_rerank"}
ENSEMBLE_METHOD = os.environ.get("RTRACE_METHOD", "edit_dist")
ENSEMBLE_METHOD = _METHOD_ALIASES.get(ENSEMBLE_METHOD, ENSEMBLE_METHOD)
RRF_K0 = 60

FRAGMENTSHOT_MAX_FRAGMENT_SIZE = 5
FRAGMENTSHOT_OVERLAPS = False

DEVTEST_N: Optional[int] = 100

# ── Reasoning state comes first: sampling + budget defaults depend on it.
#    on → ENABLE_REASONING + reasoning traces logged; off → plain decoding.
_REASONING_STATE = os.environ.get("RTRACE_REASONING", "off").lower()
if _REASONING_STATE not in ("on", "off"):
    raise ValueError(f"RTRACE_REASONING must be 'on' or 'off', got {_REASONING_STATE!r}")
_REASONING_ON = _REASONING_STATE == "on"

# ── Decoding budgets are sized for WMT24++'s paragraph-length segments
#    (far longer than FLORES: the worst observed k=10 prompt measured 7,937
#    input tokens, and the old OFF budget of 256 silently truncated long
#    Malayalam translations). Both states reserve the SAME 12,288 tokens of
#    input headroom — prompts are identical across states, so a prompt must
#    fit (or fail loudly) in both, never one-sided:
#      ON : 16384 new tokens in a 28672 window (trace + translation)
#      OFF:  2048 new tokens in a 14336 window (longest segment ~800 EN
#            tokens x ~2 target-token expansion fits comfortably)
#    No per-request clamping — an overlong prompt fails the job loudly.
#    Sampling follows each family's official card: Qwen3 thinking 0.6/0.95,
#    non-thinking 0.7/0.8 (defaults below); Mistral via sbatch overrides
#    (0.7/0.95 family-wide, ministral_14b at temperature 1.0), both states.
#    Traces exceeding the ON budget are truncated -> empty translation ->
#    caught by the retry loop below.
REQUEST_BATCH_SIZE = 32
MAX_NEW_TOKENS = int(os.environ.get("RTRACE_MAX_NEW_TOKENS", "16384" if _REASONING_ON else "2048"))
TEMPERATURE = float(os.environ.get("RTRACE_TEMPERATURE", "0.6" if _REASONING_ON else "0.7"))
TOP_P = float(os.environ.get("RTRACE_TOP_P", "0.95" if _REASONING_ON else "0.8"))
TOP_K = 20
MIN_P = 0
MAX_MODEL_LEN = int(os.environ.get("RTRACE_MAX_MODEL_LEN", "28672" if _REASONING_ON else "14336"))
STOP_SEQUENCES = []

PROMPT_HEADER = (
    'You are an expert translator who translates sentences from any specified src language to any specified tgt language.'
    ' You should reason over the demonstration sentences provided to you below, using them as a guide to accurately translate the final sentence.'
    ' Your final response should be the translation of the final untranslated src sentence to the tgt language with no other words or characters accompanying the translation.\n'
)
# SYSTEM_PROMPT: when True, the model's own SYSTEM_PROMPT.txt is fetched from
# its HF repo and prepended (Mistral's reasoning recipe). Resolved per-model
# inside main(): defaults to True for the Mistral family when reasoning is on,
# False otherwise; RTRACE_SYSTEM_PROMPT=0/1 forces it globally.
_SYSTEM_PROMPT_ENV = os.environ.get("RTRACE_SYSTEM_PROMPT", "")

GEN_ROOT = os.environ.get("RTRACE_GEN_ROOT", "drive/MyDrive/Qwen_All_Reasoning_Off_edit_dist")
REASONING_ROOT = os.environ.get("RTRACE_REASONING_ROOT", "drive/MyDrive/reasoning_traces_Qwen_dummy")
ANALYSIS_DIR = os.environ.get("RTRACE_ANALYSIS_DIR", "retrieval_overlap_analysis")

# -----------------------------------------------------------------------
# When True:  sentences that already have a non-empty translation in the
#             output JSONL are skipped; their stored translation and
#             reasoning are preserved as-is.  Only sentences whose stored
#             translation is None or "" are (re-)translated, and for those
#             the reasoning trace is always overwritten with the new one.
# When False: original behaviour — every sentence is (re-)translated and
#             every output file is overwritten from scratch.
# -----------------------------------------------------------------------
SKIP_EXISTING_TRANSLATIONS: bool = True

# -----------------------------------------------------------------------
# Number of additional attempts to make when the model returns an empty
# translation.  The initial attempt does not count as a retry, so a value
# of 3 means up to 4 total calls per sentence.  Set to 0 to disable
# retries entirely.
# -----------------------------------------------------------------------
MAX_TRANSLATION_RETRIES: int = 7

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
TOPK_SIM_CHUNK = 256

VLLM_SERVER_HOST = "127.0.0.1"
# RTRACE_PORT_BASE: overridable so (a) several single-GPU jobs sharing a node
# and (b) several per-GPU server processes inside one language-parallel job
# never race for the same port. SLURM scripts derive it from the job id.
VLLM_PORT_BASE = int(os.environ.get("RTRACE_PORT_BASE", "8000"))
# First boot on a fresh cache can legitimately take >20 min for the 24B/32B
# models (weight download over NFS + safetensors load + torch.compile), so the
# readiness timeout must comfortably exceed that — 600s killed healthy boots.
VLLM_SERVER_START_TIMEOUT_S = int(os.environ.get("RTRACE_SERVER_START_TIMEOUT_S", "2400"))
VLLM_SERVER_POLL_INTERVAL_S = 1.0
VLLM_SERVER_LOG_DIR = "vllm_server_logs"

LOG_REASONING_TRACES = _REASONING_ON
ENABLE_REASONING = _REASONING_ON

# Fixed random-selection artifact, one file PER DATASET (rows are indices
# into that dataset's eng-side pool; pool sizes differ, so sharing a file
# across datasets would silently misindex or fail validation). The wmt24pp
# file is committed to the repo (seed 12345, scripts/make_random_pool_
# selections.py) so every concurrent job LOADS the same fixed selection
# instead of racing to create one.
RANDOM_SELECTION_FILEPATHS: List[str] = [
    os.environ.get(
        "RTRACE_RANDOM_POOL",
        f"data/random_pool_selections/{DATASET}_eng_random_pool.json",
    ),
]

# MODEL_REGISTRY: every model in the study → (HF model id, reasoning parser,
# family). Parser names follow vLLM's --reasoning-parser; ids match the
# tokenizer registry in token_count_inference_budget.py.
MODEL_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    "qwen3_8b":        ("Qwen/Qwen3-8B",  "qwen3",   "qwen"),
    "qwen3_14b":       ("Qwen/Qwen3-14B", "qwen3",   "qwen"),
    "qwen3_32b":       ("Qwen/Qwen3-32B", "qwen3",   "qwen"),
    "ministral_8b":    ("mistralai/Ministral-3-8B-Reasoning-2512",  "mistral", "mistral"),
    "ministral_14b":   ("mistralai/Ministral-3-14B-Reasoning-2512", "mistral", "mistral"),
    "magistral_small": ("mistralai/Magistral-Small-2509",           "mistral", "mistral"),
}

# RTRACE_MODELS: comma-separated model keys to run in this job (default: all).
_SELECTED_MODELS = [
    s.strip() for s in os.environ.get("RTRACE_MODELS", ",".join(MODEL_REGISTRY)).split(",") if s.strip()
]
for _mk in _SELECTED_MODELS:
    if _mk not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key in RTRACE_MODELS: {_mk!r} (known: {sorted(MODEL_REGISTRY)})")

reasoning_model_specs: List[Tuple[str, str, Optional[str]]] = [
    (mk, MODEL_REGISTRY[mk][0], MODEL_REGISTRY[mk][1]) for mk in _SELECTED_MODELS
]


def main() -> None:
    """
    Purpose: Run retrieval+ensembling+generation pipeline for reasoning models across multiple src->tgt directions.
    Inputs: config variables in this cell and helper functions from Cell 1.
    Outputs: JSONL generations and JSONL reasoning traces.
    """
    use_fragment_shot, ensemble_submethod = parse_ensemble_method(ENSEMBLE_METHOD)

    if M_PER_MODEL < K_MAX:
        raise ValueError(f"M_PER_MODEL must be >= K_MAX (got {M_PER_MODEL} < {K_MAX}).")

    if ensemble_submethod == "random_pool" and len(RANDOM_SELECTION_FILEPATHS) != len(SRC_LANG_CODES):
        raise ValueError(
            f"RANDOM_SELECTION_FILEPATHS must have the same length as SRC_LANG_CODES when using random_pool "
            f"(got {len(RANDOM_SELECTION_FILEPATHS)} and {len(SRC_LANG_CODES)})."
        )

    tp = max(1, torch.cuda.device_count())

    jobs: List[Dict[str, object]] = []

    for src_idx, src_lang in enumerate(SRC_LANG_CODES):
        src_name = LANG_NAME.get(src_lang, src_lang)
        src_dir = LANG_DIRNAME.get(src_lang, src_lang)

        if ensemble_submethod == "random_pool":
            random_selection_path = RANDOM_SELECTION_FILEPATHS[src_idx]
        else:
            random_selection_path = None

        src_data = DATASET_LOADER(src_lang)
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
            for m in EMB_METHODS:
                dev_emb, devtest_emb_full = emb_full[m]
                devtest_emb = devtest_emb_full[:n_devtest, :]
                idx, _ = topk_cosine_indices_and_scores(
                    dev_emb,
                    devtest_emb,
                    M_PER_MODEL,
                    device=DEVICE,
                    torch_dtype=TORCH_DTYPE,
                    chunk=TOPK_SIM_CHUNK,
                )
                per_method_idx_m_full[m] = idx

        for tgt_lang in TGT_LANG_CODES:
            if tgt_lang == src_lang:
                continue

            tgt_name = LANG_NAME.get
            tgt_name = LANG_NAME.get(tgt_lang, tgt_lang)
            tgt_dir = LANG_DIRNAME.get(tgt_lang, tgt_lang)

            tgt_data = DATASET_LOADER(tgt_lang)
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
                for m in EMB_METHODS:
                    dev_emb, devtest_emb_full = emb_full[m]
                    devtest_emb = devtest_emb_full[:n_devtest, :]
                    idx = topk_cosine_indices_from_pools(
                        dev_emb,
                        devtest_emb,
                        pools,
                        M_PER_MODEL,
                        device=DEVICE,
                        torch_dtype=TORCH_DTYPE,
                    )
                    per_method_idx_m[m] = idx
            else:
                per_method_idx_m = per_method_idx_m_full

            overlap_mat, meanrank_mat, method_labels = compute_pairwise_overlap_and_meanrank(per_method_idx_m, K_MAX)
            _ensure_dir(ANALYSIS_DIR)
            overlap_png = os.path.join(ANALYSIS_DIR, f"{src_dir}_to_{tgt_dir}_overlap_k{K_MAX}_{ENSEMBLE_METHOD}.png")
            meanrank_png = os.path.join(ANALYSIS_DIR, f"{src_dir}_to_{tgt_dir}_meanrank_k{K_MAX}_{ENSEMBLE_METHOD}.png")
            _plot_matrix_png(overlap_mat, method_labels, f"Avg overlap (Top-{K_MAX}) {src_dir}->{tgt_dir}", overlap_png, ".2f", vmin=0, vmax=K_MAX)
            _plot_matrix_png(meanrank_mat, method_labels, f"Avg mean-rank (Top-{K_MAX}) {src_dir}->{tgt_dir}", meanrank_png, ".2f", vmin=1, vmax=K_MAX)

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

            direction_dir = f"{src_dir}_to_{tgt_dir}"
            jobs.append(
                {
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "src_name": src_name,
                    "tgt_name": tgt_name,
                    "direction_dir": direction_dir,
                    "src_dev": src_dev,
                    "src_devtest": src_devtest,
                    "tgt_dev": tgt_dev,
                    "final_indices_by_k": final_indices_by_k,
                }
            )

    for i, (llm_key, model_id, reasoning_parser) in enumerate(reasoning_model_specs):
        port = int(VLLM_PORT_BASE) + int(i)

        spec = VLLMServerSpec(
            model_key=llm_key,
            model_id=model_id,
            port=port,
            reasoning_parser= None if not ENABLE_REASONING else reasoning_parser,
            tensor_parallel_size=tp,
            gpu_memory_utilization=0.90,
            max_model_len=MAX_MODEL_LEN,
        )

        log_reasoning_traces_this_model = bool(LOG_REASONING_TRACES) and (reasoning_parser is not None)

        # Mistral's reasoning recipe injects the model's own SYSTEM_PROMPT.txt
        # (with its [THINK] scaffold); Qwen3 needs no system prompt. Env var
        # RTRACE_SYSTEM_PROMPT=0/1 overrides the family default globally.
        family = MODEL_REGISTRY[llm_key][2]
        if _SYSTEM_PROMPT_ENV != "":
            use_system_prompt = bool(int(_SYSTEM_PROMPT_ENV))
        else:
            use_system_prompt = (family == "mistral") and ENABLE_REASONING
        system_prompt = load_system_prompt(model_id, 'SYSTEM_PROMPT.txt') if use_system_prompt else None

        extra_body = {"top_k": TOP_K, "min_p": MIN_P}
        if reasoning_parser == "qwen3":
            extra_body["chat_template_kwargs"] = {"enable_thinking": bool(ENABLE_REASONING)}
        elif reasoning_parser != "mistral":
            extra_body["chat_template_kwargs"] = {"thinking": bool(ENABLE_REASONING)}

        with VLLMServerProcess(
            spec,
            host=VLLM_SERVER_HOST,
            log_dir=VLLM_SERVER_LOG_DIR,
            start_timeout_s=VLLM_SERVER_START_TIMEOUT_S,
            poll_interval_s=VLLM_SERVER_POLL_INTERVAL_S,
            openai_api_key=OPENAI_API_KEY,
        ) as server:
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=server.base_url)
            served_model_id = get_served_model_id(client)

            for job in jobs:
                direction_dir = str(job["direction_dir"])
                src_lang = str(job["src_lang"])
                tgt_lang = str(job["tgt_lang"])
                src_name = str(job["src_name"])
                tgt_name = str(job["tgt_name"])
                src_dev = job["src_dev"]  # type: ignore[assignment]
                src_devtest = job["src_devtest"]  # type: ignore[assignment]
                tgt_dev = job["tgt_dev"]  # type: ignore[assignment]
                final_indices_by_k = job["final_indices_by_k"]  # type: ignore[assignment]

                for k in K_LIST:
                    out_dir = os.path.join(GEN_ROOT, llm_key, direction_dir)
                    out_path = os.path.join(out_dir, f"k{k}_{ENSEMBLE_METHOD}_template11.jsonl")

                    r_dir = os.path.join(REASONING_ROOT, llm_key, direction_dir)
                    r_path = os.path.join(r_dir, f"k{k}_{ENSEMBLE_METHOD}_template11_reasoning.jsonl")

                    run_openai_for_direction_streaming(
                        client,
                        served_model_id=served_model_id,
                        model_key=llm_key,
                        src_lang_code=src_lang,
                        tgt_lang_code=tgt_lang,
                        k=k,
                        src_dev=src_dev,
                        src_devtest=src_devtest,
                        tgt_dev=tgt_dev,
                        dev_indices_per_devtest=final_indices_by_k[k],
                        out_jsonl_path=out_path,
                        out_reasoning_jsonl_path=r_path,
                        request_batch_size=REQUEST_BATCH_SIZE,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_new_tokens=MAX_NEW_TOKENS,
                        stop_sequences=STOP_SEQUENCES,
                        extra_body=extra_body,
                        log_reasoning_traces=log_reasoning_traces_this_model,
                        prompt_header=PROMPT_HEADER,
                        src_name=src_name,
                        tgt_name=tgt_name,
                        system_prompt=system_prompt,
                        skip_existing_translations=SKIP_EXISTING_TRANSLATIONS,
                        max_retries=MAX_TRANSLATION_RETRIES,
                    )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()