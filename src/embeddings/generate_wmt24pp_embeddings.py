"""
generate_wmt24pp_embeddings.py
──────────────────────────────
Compute retrieval embeddings for the WMT24++ experimental setup, mirroring the
FLORES embedding layout exactly so `load_retrieval_embeddings` /
`load_sentinel_src_scores` work unchanged:

    <WMT_EMB_ROOT>/<method>/eng_Latn/{dev,devtest}.bin

"dev" = the 860-sentence selection pool, "devtest" = the fixed 100-sentence
test set (both from data/wmt24pp_split.json). Only the ENGLISH source side is
embedded — retrieval is source-side in this pipeline, and WMT24++ English
sources are identical across language pairs (verified at split generation),
so one embedding set serves all covered targets.

Every output file is skipped if it already exists, so a preempted SLURM job
resumes without recomputation.

Inputs:  data/wmt24pp_split.json; RTRACE_WMT_EMB_ROOT (default
         "wmt24pp_embeddings"); COHERE_API_KEY env var for the cohere method.
Outputs: dev/devtest .bin files per method (sonar, labse, e5, cohere, MiniLM,
         bm25) plus sentinel_src difficulty scores.
"""

import os

import numpy as np
import torch

from src.data.wmt24pp import load_wmt24pp_sentences
from src.embeddings.generate_flores_embeddings import (
    BM25_NUM_WORKERS,
    SENTINEL_BATCH_SIZE,
    SENTINEL_GPUS,
    SENTINEL_METHOD_NAME,
    SENTINEL_MODEL_NAME,
    bm25_score_matrix,
    compute_sentinel_src_scores,
    embed_cohere_multilingual_v3,
    embed_labse,
    embed_minilm,
    embed_multilingual_e5,
    embed_sonar,
    ensure_dir,
    load_sentinel_src_model,
    save_bin,
)

# WMT_EMB_ROOT: output root for WMT24++ embeddings (mirrors flores_embeddings layout).
WMT_EMB_ROOT = os.environ.get("RTRACE_WMT_EMB_ROOT", "wmt24pp_embeddings")
# SRC_LANG: the only embedded language — retrieval is English-source-side.
SRC_LANG = "eng_Latn"
# RTRACE_EMB_METHODS: csv filter over {sonar,labse,e5,cohere,MiniLM,bm25,
# sentinel_src}. Exists because the SONAR stack (fairseq2n, numpy>=2.2) and
# the Sentinel stack (unbabel-comet, numpy<2) cannot share one virtualenv —
# the SLURM job runs this script twice: dense+bm25 methods in .venv-emb, then
# sentinel_src alone in .venv-sent. Default runs everything (single-env case).
_SELECTED = [
    s.strip() for s in os.environ.get(
        "RTRACE_EMB_METHODS", "sonar,labse,e5,cohere,MiniLM,bm25,sentinel_src"
    ).split(",") if s.strip()
]
# MODELS_TO_RUN: same method set as the FLORES script, filtered by the env var.
MODELS_TO_RUN = [m for m in ["sonar", "labse", "e5", "cohere", "MiniLM", "bm25"] if m in _SELECTED]
# RUN_SENTINEL: whether this pass computes SENTINEL source-difficulty scores.
RUN_SENTINEL = SENTINEL_METHOD_NAME in _SELECTED or "sentinel" in _SELECTED


def _out_path(method: str, split: str) -> str:
    """
    Purpose: Build the output .bin path for a method/split under WMT_EMB_ROOT.
    Inputs: method name string, split name string ("dev" or "devtest").
    Outputs: full file path string (parent dir created).
    """
    d = os.path.join(WMT_EMB_ROOT, method, SRC_LANG)
    ensure_dir(d)
    return os.path.join(d, f"{split}.bin")


def main() -> None:
    """
    Purpose: Embed the WMT24++ English pool/test sets with every retrieval method, resumably.
    Inputs: module constants and the fixed split via load_wmt24pp_sentences.
    Outputs: .bin files under WMT_EMB_ROOT; skips any file that already exists.
    """
    data = load_wmt24pp_sentences(SRC_LANG)
    dev_sents = data["dev"]
    devtest_sents = data["devtest"]
    print(f"[wmt24pp] pool={len(dev_sents)} test={len(devtest_sents)} -> {WMT_EMB_ROOT}")
    print(f"[wmt24pp] methods this pass: {MODELS_TO_RUN}  sentinel={RUN_SENTINEL}")

    model_fns = {
        "labse": lambda sents, lang=None: embed_labse(sents),
        "cohere": lambda sents, lang=None: embed_cohere_multilingual_v3(sents),
        "e5": lambda sents, lang=None: embed_multilingual_e5(sents),
        "sonar": lambda sents, lang: embed_sonar(sents, lang),
        "MiniLM": lambda sents, lang=None: embed_minilm(sents),
    }

    for method in MODELS_TO_RUN:
        dev_path = _out_path(method, "dev")
        devtest_path = _out_path(method, "devtest")
        if os.path.exists(dev_path) and os.path.exists(devtest_path):
            print(f"[skip] {method}: both splits already on disk.")
            continue

        if method == "bm25":
            emb_dev = np.eye(len(dev_sents), dtype=np.float32)
            emb_devtest = bm25_score_matrix(
                dev_sentences=dev_sents,
                query_sentences=devtest_sents,
                num_workers=BM25_NUM_WORKERS,
            ).astype(np.float32)
            save_bin(dev_path, emb_dev)
            save_bin(devtest_path, emb_devtest)
            print(f"[ok] bm25 saved | dev={emb_dev.shape} devtest={emb_devtest.shape}")
        else:
            fn = model_fns[method]
            for split, sents in (("dev", dev_sents), ("devtest", devtest_sents)):
                path = _out_path(method, split)
                if os.path.exists(path):
                    print(f"[skip] {method}/{split} already on disk.")
                    continue
                emb = fn(sents, lang=SRC_LANG)
                save_bin(path, emb)
                print(f"[ok] {method} saved split={split} shape={emb.shape}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not RUN_SENTINEL:
        print("[wmt24pp] sentinel_src not selected for this pass; done.")
        return

    sent_dev_path = _out_path(SENTINEL_METHOD_NAME, "dev")
    sent_devtest_path = _out_path(SENTINEL_METHOD_NAME, "devtest")
    if os.path.exists(sent_dev_path) and os.path.exists(sent_devtest_path):
        print(f"[skip] {SENTINEL_METHOD_NAME}: both splits already on disk.")
    else:
        sentinel_model = load_sentinel_src_model(SENTINEL_MODEL_NAME)
        sentinel_dev = compute_sentinel_src_scores(
            dev_sents, model=sentinel_model,
            batch_size=SENTINEL_BATCH_SIZE, gpus=SENTINEL_GPUS,
        )
        sentinel_devtest = compute_sentinel_src_scores(
            devtest_sents, model=sentinel_model,
            batch_size=SENTINEL_BATCH_SIZE, gpus=SENTINEL_GPUS,
        )
        save_bin(sent_dev_path, sentinel_dev)
        save_bin(sent_devtest_path, sentinel_devtest)
        print(f"[ok] {SENTINEL_METHOD_NAME} saved | dev={sentinel_dev.shape} devtest={sentinel_devtest.shape}")
        del sentinel_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
