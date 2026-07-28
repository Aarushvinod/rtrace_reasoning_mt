"""
generate_flores_embeddings.py
─────────────────────────────
Compute FLORES-200 embeddings for every configured target language across every
supported retrieval method (SONAR, LaBSE, mE5, Cohere multilingual v3, MiniLM,
BM25) plus SENTINEL source-difficulty scores, and persist them to disk under
`flores_embeddings/<method>/<lang>/{dev,devtest}.bin`.

Inputs:  MODELS_TO_RUN, LANG_LIST, SPLITS, and OUT_DIR configured at the top
         of this file; requires a Cohere API key in the COHERE_API_KEY env var
         for the Cohere method and network access for HuggingFace model
         downloads.
Outputs: per-method / per-language binary embedding matrices, plus a
         cosine-similarity histogram PNG (dev vs devtest) rendered by
         `cosine_histogram_dev_vs_devtest`.
"""

import os
from typing import Dict, Iterable, List, Literal, Optional

import numpy as np
import torch
from datasets import load_dataset

# LANG_LIST: FLORES language codes to generate embeddings for (may include English itself).
LANG_LIST = ["eng_Latn"]
# SRC_LANG_DEFAULT: Default “source” language (kept for compatibility; not required for multilingual models).
SRC_LANG_DEFAULT = "eng_Latn"
# SPLITS: FLORES splits to embed (dev + devtest).
SPLITS = ["dev", "devtest"]
# OUT_DIR: Root directory where embeddings are written (NEW STRUCTURE).
OUT_DIR = "flores_embeddings"
# BATCH_SIZE: Batch size for local embedding model inference.
BATCH_SIZE = 64
# COHERE_BATCH_SIZE: Batch size for Cohere embed API calls.
COHERE_BATCH_SIZE = 96
# DEVICE: Torch device used by models that support GPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODELS_TO_RUN: List of embedding methods to compute and save.
MODELS_TO_RUN = ["sonar", "labse", "e5", "cohere", "MiniLM", "bm25"]
# BIN_SIZE: Histogram bin width for cosine similarity distribution.
BIN_SIZE = 0.01
# SIM_BLOCK_ROWS: Number of dev rows per block when forming similarity blocks.
SIM_BLOCK_ROWS = 512
# HIST_OUT_NAME: Output filename for the histogram plot image.
HIST_OUT_NAME = "cosine_similarity_distributions_dev_vs_devtest.png"
# COHERE_API_KEY_ENV: Environment variable name that stores the Cohere API key.
COHERE_API_KEY_ENV = "COHERE_API_KEY"
# BM25_NUM_WORKERS: Number of processes for BM25 scoring.
BM25_NUM_WORKERS = 8
# SENTINEL_METHOD_NAME: Folder name used to save SENTINEL source-difficulty scores.
SENTINEL_METHOD_NAME = "sentinel_src"
# SENTINEL_MODEL_NAME: Hugging Face model id for the SENTINEL source-difficulty model.
SENTINEL_MODEL_NAME = "Prosho/sentinel-src-25"
# SENTINEL_BATCH_SIZE: Batch size for SENTINEL scoring.
SENTINEL_BATCH_SIZE = 32
# SENTINEL_GPUS: Number of GPUs passed to SENTINEL prediction.
SENTINEL_GPUS = 1

SplitOut = Literal["dev", "devtest"]


def ensure_dir(path: str) -> None:
    """
    Purpose: Ensure a directory exists.
    Inputs: path string for the directory to create.
    Outputs: None.
    """
    os.makedirs(path, exist_ok=True)


def iter_batches(items: List[str], batch_size: int) -> Iterable[List[str]]:
    """
    Purpose: Yield successive batches from a list.
    Inputs: items list and batch_size integer.
    Outputs: Iterator over list batches.
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def save_bin(path: str, mat: np.ndarray) -> None:
    """
    Purpose: Save a float32 matrix as a raw .bin file.
    Inputs: output file path and matrix array.
    Outputs: None.
    """
    mat = np.asarray(mat, dtype=np.float32)
    mat.tofile(path)


def model_output_dir(model_name: str, lang_code: str) -> str:
    """
    Purpose: Create and return the output directory for a model + language.
    Inputs: model_name string, lang_code string.
    Outputs: directory path string.
    """
    d = os.path.join(OUT_DIR, model_name, lang_code)
    ensure_dir(d)
    return d


def output_path(model_name: str, lang_code: str, split: SplitOut) -> str:
    """
    Purpose: Build the output file path for a model/language/split.
    Inputs: model_name, lang_code, and split label.
    Outputs: full .bin output path string.
    """
    fname = f"{split}.bin"
    return os.path.join(model_output_dir(model_name, lang_code), fname)


def load_flores(lang: str) -> Dict[str, List[str]]:
    """
    Purpose: Load FLORES dev and devtest sentence lists for a language.
    Inputs: lang FLORES language code string.
    Outputs: dict with keys 'dev' and 'devtest' holding sentence lists.
    """
    ds = load_dataset("facebook/flores", lang, trust_remote_code=True)
    dev = [ex["sentence"] for ex in ds["dev"]]
    devtest = [ex["sentence"] for ex in ds["devtest"]]
    return {"dev": dev, "devtest": devtest}


def _mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """
    Purpose: Mean-pool token embeddings using an attention mask.
    Inputs: last_hidden tensor and attn_mask tensor.
    Outputs: pooled tensor of shape (batch, hidden_dim).
    """
    mask = attn_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _row_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Purpose: L2-normalize rows of a tensor.
    Inputs: tensor x and small epsilon.
    Outputs: row-normalized tensor.
    """
    return x / (x.norm(dim=1, keepdim=True).clamp_min(eps))


def cosine_histogram_dev_vs_devtest(
    emb_dev: np.ndarray,
    emb_devtest: np.ndarray,
    bin_edges: np.ndarray,
    device: str = DEVICE,
    block_rows: int = SIM_BLOCK_ROWS,
) -> np.ndarray:
    """
    Purpose: Histogram cosine similarities for all pairs in dev x devtest without storing the full matrix.
    Inputs: dev embeddings, devtest embeddings, bin_edges array, device string, and block_rows integer.
    Outputs: counts array of length len(bin_edges)-1.
    """
    if emb_dev.size == 0 or emb_devtest.size == 0:
        return np.zeros(len(bin_edges) - 1, dtype=np.int64)

    dev_t = torch.from_numpy(emb_dev.astype(np.float32))
    test_t = torch.from_numpy(emb_devtest.astype(np.float32))

    dev_t = _row_normalize(dev_t).to(device)
    test_t = _row_normalize(test_t).to(device)

    counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)

    with torch.no_grad():
        test_T = test_t.transpose(0, 1).contiguous()
        n_dev = dev_t.shape[0]
        for start in range(0, n_dev, block_rows):
            dev_block = dev_t[start : start + block_rows]
            sims = (dev_block @ test_T).float()
            sims_np = sims.detach().cpu().numpy().ravel()
            block_counts, _ = np.histogram(sims_np, bins=bin_edges)
            counts += block_counts.astype(np.int64)

    return counts


def embed_labse(sentences: List[str]) -> np.ndarray:
    """
    Purpose: Compute LaBSE embeddings for input sentences.
    Inputs: sentences list of strings.
    Outputs: float32 numpy array of embeddings.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/LaBSE", device=DEVICE)
    emb = model.encode(
        sentences,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return emb.astype(np.float32)


def embed_cohere_multilingual_v3(sentences: List[str]) -> np.ndarray:
    """
    Purpose: Compute Cohere multilingual v3 embeddings for input sentences.
    Inputs: sentences list of strings.
    Outputs: float32 numpy array of embeddings.
    """
    import time
    import cohere

    api_key = os.environ.get(COHERE_API_KEY_ENV, "")
    if not api_key:
        raise RuntimeError(f"{COHERE_API_KEY_ENV} not set in environment.")

    co = cohere.Client(api_key)

    chunks = []
    for batch in iter_batches(sentences, COHERE_BATCH_SIZE):
        resp = co.embed(
            texts=batch,
            model="embed-multilingual-v3.0",
            input_type="search_document",
        )
        time.sleep(5)
        chunks.append(np.asarray(resp.embeddings, dtype=np.float32))

    return np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)


def embed_multilingual_e5(sentences: List[str]) -> np.ndarray:
    """
    Purpose: Compute multilingual-e5 embeddings for input sentences.
    Inputs: sentences list of strings.
    Outputs: float32 numpy array of embeddings.
    """
    from transformers import AutoModel, AutoTokenizer

    model_name = "intfloat/multilingual-e5-large"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    sentences = [f"passage: {s}" for s in sentences]

    all_vecs = []
    with torch.no_grad():
        for batch in iter_batches(sentences, BATCH_SIZE):
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(DEVICE)
            out = model(**enc)
            vec = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            all_vecs.append(vec.detach().cpu())

    return (
        torch.cat(all_vecs, dim=0).numpy().astype(np.float32)
        if all_vecs
        else np.zeros((0, 1024), dtype=np.float32)
    )


def embed_sonar(sentences: List[str], lang: str) -> np.ndarray:
    """
    Purpose: Compute SONAR text embeddings for input sentences.
    Inputs: sentences list of strings and lang code string.
    Outputs: float32 numpy array of embeddings.
    """
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    sonar_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    pipeline = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=sonar_device,
    )

    chunks = []
    with torch.no_grad():
        for batch in iter_batches(sentences, BATCH_SIZE):
            emb = pipeline.predict(batch, source_lang=lang)
            chunks.append(emb.detach().cpu().numpy().astype(np.float32))

    return np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)


def embed_minilm(sentences: List[str]) -> np.ndarray:
    """
    Purpose: Compute multilingual MiniLM embeddings for input sentences.
    Inputs: sentences list of strings.
    Outputs: float32 numpy array of embeddings.
    """
    from sentence_transformers import SentenceTransformer

    model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_id, device=DEVICE)
    emb = model.encode(
        sentences,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return emb.astype(np.float32)


def bm25_score_matrix(
    dev_sentences: List[str],
    query_sentences: List[str],
    num_workers: int,
) -> np.ndarray:
    """
    Purpose: Compute BM25Okapi score vectors for each query sentence against the dev corpus.
    Inputs: dev_sentences list, query_sentences list, and num_workers integer.
    Outputs: float32 numpy array of shape (n_queries, n_dev).
    """
    from rank_bm25 import BM25Okapi

    tokenized_dev = [doc.split(" ") for doc in dev_sentences]
    bm25 = BM25Okapi(tokenized_dev)

    def f(example: str):
        return bm25.get_scores(example.split(" "))

    if num_workers is None or num_workers <= 1:
        scores = [f(q) for q in query_sentences]
        return np.asarray(scores, dtype=np.float32)

    import multiprocess as mp

    p = mp.Pool(num_workers)
    scores = p.map(f, query_sentences)
    p.close()
    p.join()
    return np.asarray([s for s in scores], dtype=np.float32)


def load_sentinel_src_model(model_name: str = SENTINEL_MODEL_NAME):
    """
    Purpose: Load the official SENTINEL-SRC model using the supported guardians-mt-eval API.
    Inputs: model_name Hugging Face model id.
    Outputs: loaded SENTINEL model object.
    """
    from sentinel_metric import download_model, load_from_checkpoint

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    return model


def compute_sentinel_src_scores(
    sentences: List[str],
    model,
    batch_size: int = SENTINEL_BATCH_SIZE,
    gpus: int = SENTINEL_GPUS,
) -> np.ndarray:
    """
    Purpose: Compute SENTINEL-SRC source-difficulty scores using the official model.predict API.
    Inputs: sentences list, loaded SENTINEL model, batch_size integer, and gpus integer.
    Outputs: float32 numpy array of shape (n_sentences, 1), where higher means easier to translate.
    """
    if not sentences:
        return np.zeros((0, 1), dtype=np.float32)

    data = [{"src": s} for s in sentences]
    gpus_to_use = int(gpus) if torch.cuda.is_available() else 0

    output = model.predict(
        data,
        batch_size=int(batch_size),
        gpus=gpus_to_use,
    )

    scores = np.asarray(output.scores, dtype=np.float32).reshape(-1, 1)

    if scores.shape[0] != len(sentences):
        raise ValueError(
            f"SENTINEL returned {scores.shape[0]} scores for {len(sentences)} sentences."
        )

    return scores


def plot_histograms_per_model(
    model_order: List[str],
    bin_edges: np.ndarray,
    per_model_counts: Dict[str, np.ndarray],
    out_path: str,
) -> None:
    """
    Purpose: Plot one histogram subplot per model with a shared y-axis and save the image.
    Inputs: model_order list, bin_edges array, counts dict, and out_path string.
    Outputs: None.
    """
    import matplotlib.pyplot as plt

    n = len(model_order)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    global_max = 0
    for m in model_order:
        if m in per_model_counts:
            global_max = max(global_max, int(per_model_counts[m].max()))
    if global_max == 0:
        global_max = 1

    bin_left = bin_edges[:-1]
    widths = np.diff(bin_edges)

    for j, m in enumerate(model_order):
        ax = axes[j]
        counts = per_model_counts.get(m, np.zeros(len(bin_left), dtype=np.int64))
        ax.bar(bin_left, counts, width=widths, align="edge", alpha=0.75, color=colors[j % len(colors)])
        ax.set_title(m)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(0, global_max * 1.05)

        if j == 0:
            ax.set_ylabel("Count")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        ax.set_xlabel("Cosine similarity")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Cosine Similarity Distributions (dev × devtest)", y=1.02)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()


def main() -> None:
    """
    Purpose: Compute embeddings (including BM25Okapi), save them, and plot cosine similarity distributions per language.
    Inputs: global configuration variables and FLORES dataset splits.
    Outputs: .bin embedding files, SENTINEL score files, and per-language histogram images.
    """
    ensure_dir(OUT_DIR)
    ensure_dir(os.path.join(OUT_DIR, "_plots"))

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_fns = {
        "labse": lambda sents, lang=None: embed_labse(sents),
        "cohere": lambda sents, lang=None: embed_cohere_multilingual_v3(sents),
        "e5": lambda sents, lang=None: embed_multilingual_e5(sents),
        "sonar": lambda sents, lang: embed_sonar(sents, lang),
        "MiniLM": lambda sents, lang=None: embed_minilm(sents),
    }

    bin_edges = np.arange(-1.0, 1.0 + BIN_SIZE, BIN_SIZE, dtype=np.float32)

    sentinel_model = load_sentinel_src_model(SENTINEL_MODEL_NAME)

    for lang_code in LANG_LIST:
        flores = load_flores(lang_code)

        per_model_counts: Dict[str, np.ndarray] = {}

        for model_key in MODELS_TO_RUN:
            emb_dev = None
            emb_devtest = None

            if model_key == "bm25":
                dev_sents = flores["dev"]
                devtest_sents = flores["devtest"]

                emb_dev = np.eye(len(dev_sents), dtype=np.float32)
                emb_devtest = bm25_score_matrix(
                    dev_sentences=dev_sents,
                    query_sentences=devtest_sents,
                    num_workers=BM25_NUM_WORKERS,
                ).astype(np.float32)

                save_bin(output_path(model_key, lang_code, "dev"), emb_dev)
                save_bin(output_path(model_key, lang_code, "devtest"), emb_devtest)
                print(f"[ok] bm25 saved for lang={lang_code} | dev={emb_dev.shape} devtest={emb_devtest.shape}")

            else:
                if model_key not in model_fns:
                    raise ValueError(f"Unknown model in MODELS_TO_RUN: {model_key}")
                fn = model_fns[model_key]

                for split_out in SPLITS:
                    sents = flores[split_out]
                    emb = fn(sents, lang=lang_code)

                    save_bin(output_path(model_key, lang_code, split_out), emb)

                    if split_out == "dev":
                        emb_dev = emb
                    else:
                        emb_devtest = emb

                    print(f"[ok] {model_key} saved for lang={lang_code} split={split_out} shape={emb.shape}")

            assert emb_dev is not None and emb_devtest is not None
            counts = cosine_histogram_dev_vs_devtest(
                emb_dev=emb_dev,
                emb_devtest=emb_devtest,
                bin_edges=bin_edges,
                device=DEVICE,
                block_rows=SIM_BLOCK_ROWS,
            )
            per_model_counts[model_key] = counts

            del emb_dev, emb_devtest
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        sentinel_dev = compute_sentinel_src_scores(
            flores["dev"],
            model=sentinel_model,
            batch_size=SENTINEL_BATCH_SIZE,
            gpus=SENTINEL_GPUS,
        )
        sentinel_devtest = compute_sentinel_src_scores(
            flores["devtest"],
            model=sentinel_model,
            batch_size=SENTINEL_BATCH_SIZE,
            gpus=SENTINEL_GPUS,
        )

        save_bin(output_path(SENTINEL_METHOD_NAME, lang_code, "dev"), sentinel_dev)
        save_bin(output_path(SENTINEL_METHOD_NAME, lang_code, "devtest"), sentinel_devtest)
        print(
            f"[ok] {SENTINEL_METHOD_NAME} saved for lang={lang_code} "
            f"| dev={sentinel_dev.shape} devtest={sentinel_devtest.shape}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        plot_path = os.path.join(OUT_DIR, "_plots", lang_code, HIST_OUT_NAME)
        plot_histograms_per_model(
            model_order=MODELS_TO_RUN,
            bin_edges=bin_edges,
            per_model_counts=per_model_counts,
            out_path=plot_path,
        )
        print(f"[ok] saved histogram plot for lang={lang_code} -> {plot_path}")

    del sentinel_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Guarded so this module is importable (generate_wmt24pp_embeddings.py reuses
# the embed_* / bm25 / sentinel functions); behavior when run as a script is
# unchanged.
if __name__ == "__main__":
    main()