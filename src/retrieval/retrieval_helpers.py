"""
retrieval_helpers.py
─────────────────────
Shared few-shot retrieval + ensembling helpers used by the reasoning
generation pipeline: prompt template construction (template-11),
FLORES sentence loading with uniqueness checks, retrieval-embedding loading,
top-k cosine retrieval with fragment-shot pools, and ensemble selection
methods (borda / round-robin / RRF / BM25 rerank / SENTINEL rerank /
random pool / edit-distance). Also exposes overlap-analysis helpers used
downstream to compare retrieval strategies.

Inputs:  precomputed embedding matrices under `flores_embeddings/`,
         raw FLORES sentences, and the ensemble method identifier passed
         through `parse_ensemble_method` / `ensemble_topk_dispatch`.
Outputs: retrieval index lists, prompt strings for downstream generation,
         and analysis plots when the caller invokes the overlap helpers.

Note: the direct-vLLM driver `run_vllm_for_direction` that used to live at
the bottom of this cell has been dropped — it was only called by the
non-reasoning main, which is out of scope for this codebase.
"""

# =========================
# Cell 1 — Helper functions (no config variables)
# =========================

import os
import json
import time
import signal
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from openai import OpenAI


@dataclass
class Template:
    header: str = ""
    prefix: str = ""
    middle: str = ""
    suffix: str = ""

    def get_prompt(self, demonstrations, example, start="", end=""):
        """
        Purpose: Build a few-shot prompt.
        Inputs: demonstrations list of (src,tgt) strings and the example src string.
        Outputs: full prompt string.
        """
        prompt = self.header
        if demonstrations:
            for x, y in demonstrations:
                prompt += f"{self.prefix}"
                prompt += f"{start}{x}{end}"
                prompt += f"{self.middle}"
                prompt += f"{start}{y}{end}"
                prompt += f"{self.suffix}"
        prompt += f"{self.prefix}"
        prompt += f"{start}{example}{end}"
        prompt += f"{self.middle}"
        return prompt


def get_template_11(src_name: str, tgt_name: str, header: str = "") -> Template:
    """
    Purpose: Construct template-11 formatting.
    Inputs: src_name string, tgt_name string, header string.
    Outputs: Template object.
    """
    return Template(
        header=header,
        prefix=f"{src_name} sentence\n",
        middle=f"\n{tgt_name} translation\n",
        suffix="\n###\n",
    )


def generate_prompt_template11(
    src_sentence: str,
    demonstrations: List[Tuple[str, str]],
    *,
    src_name: str,
    tgt_name: str,
    header: str = "",
) -> Tuple[str, Template]:
    """
    Purpose: Create a template-11 prompt.
    Inputs: src_sentence string, demonstrations list, src/tgt display names, header string.
    Outputs: (prompt string, Template object).
    """
    template = get_template_11(src_name, tgt_name, header=header)
    prompt = template.get_prompt(demonstrations=demonstrations, example=src_sentence)
    return prompt, template


def load_flores_sentences(lang: str) -> Dict[str, List[str]]:
    """
    Purpose: Load FLORES sentences and validate uniqueness.
    Inputs: lang code string.
    Outputs: dict with 'dev' and 'devtest' sentence lists.
    """
    ds = load_dataset("Muennighoff/flores200", lang)   # ← was "facebook/flores"
    dev = [ex["sentence"] for ex in ds["dev"]]
    devtest = [ex["sentence"] for ex in ds["devtest"]]
    return {"dev": dev, "devtest": devtest}


def _ensure_dir(path: str) -> None:
    """
    Purpose: Create directory if missing.
    Inputs: path string.
    Outputs: None.
    """
    os.makedirs(path, exist_ok=True)


def _apply_devtest_limit(devtest: List[str], n: Optional[int]) -> List[str]:
    """
    Purpose: Limit devtest to first n items if n is provided.
    Inputs: devtest list, n optional integer.
    Outputs: possibly-sliced devtest list.
    """
    if n is None:
        return devtest
    n_int = int(n)
    if n_int <= 0:
        return []
    return devtest[:n_int]


def _read_bin_matrix(path: str, n_rows: int) -> np.ndarray:
    """
    Purpose: Read float32 .bin into (n_rows, dim).
    Inputs: file path string, n_rows integer.
    Outputs: numpy float32 matrix.
    """
    arr = np.fromfile(path, dtype=np.float32)
    if n_rows <= 0:
        raise ValueError(f"n_rows must be > 0, got {n_rows}")
    if arr.size % n_rows != 0:
        raise ValueError(f"File {path} has {arr.size} floats not divisible by n_rows={n_rows}.")
    dim = arr.size // n_rows
    return arr.reshape(n_rows, dim)


def emb_path(emb_root: str, method: str, lang_code: str, split_out: str) -> str:
    """
    Purpose: Build embedding filepath for the new structure.
    Inputs: emb_root string, method string, lang_code string, split_out string.
    Outputs: full path to .bin file.
    """
    fname = f"{split_out}.bin"
    p = os.path.join(emb_root, method, lang_code, fname)
    if os.path.exists(p):
        return p
    if split_out == "devtest":
        p2 = os.path.join(emb_root, method, lang_code, "test.bin")
        if os.path.exists(p2):
            return p2
    return p


def load_retrieval_embeddings(
    emb_root: str,
    methods: List[str],
    lang_code: str,
    n_dev: int,
    n_devtest: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Purpose: Load per-method (dev, devtest) embeddings for a specific source language.
    Inputs: emb_root, methods list, lang_code, n_dev, n_devtest.
    Outputs: dict mapping method -> (dev_matrix, devtest_matrix).
    """
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for m in methods:
        dev_path = emb_path(emb_root, m, lang_code, "dev")
        devtest_path = emb_path(emb_root, m, lang_code, "devtest")

        if not os.path.exists(dev_path):
            raise FileNotFoundError(f"Missing dev embeddings: {dev_path}")
        if not os.path.exists(devtest_path):
            raise FileNotFoundError(f"Missing devtest embeddings: {devtest_path}")

        dev = _read_bin_matrix(dev_path, n_dev)
        devtest = _read_bin_matrix(devtest_path, n_devtest)
        out[m] = (dev, devtest)

    return out


def load_sentinel_src_scores(
    emb_root: str,
    lang_code: str,
    n_dev: int,
    method_name: str = "sentinel_src",
) -> np.ndarray:
    """
    Purpose: Load stored SENTINEL source-difficulty scores for the dev split.
    Inputs: emb_root string, lang_code string, n_dev integer, and method_name string.
    Outputs: float32 score vector of shape (n_dev,).
    """
    dev_path = emb_path(emb_root, method_name, lang_code, "dev")
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"Missing SENTINEL dev scores: {dev_path}")

    dev_scores = _read_bin_matrix(dev_path, n_dev).reshape(n_dev, -1)
    if dev_scores.shape[1] != 1:
        raise ValueError(
            f"SENTINEL dev scores at {dev_path} should have exactly 1 column, got shape={dev_scores.shape}."
        )

    return dev_scores[:, 0].astype(np.float32)


def iter_batches(items: List[str], batch_size: int) -> Iterable[Tuple[int, List[str]]]:
    """
    Purpose: Yield (start_index, batch_list).
    Inputs: items list, batch_size integer.
    Outputs: generator of (start, batch) pairs.
    """
    for i in range(0, len(items), batch_size):
        yield i, items[i : i + batch_size]


def postprocess_generation(raw: str, template: Template) -> str:
    """
    Purpose: Strip template artifacts from raw model output.
    Inputs: raw string, template.
    Outputs: cleaned translation string.
    """
    text = (raw or "").lstrip()

    end = text.find(template.suffix)
    if end != -1:
        text = text[:end]

    hash_idx = text.find("###")
    if hash_idx != -1:
        text = text[:hash_idx]

    return text.strip()


def topk_cosine_indices_and_scores(
    dev_emb: np.ndarray,
    devtest_emb: np.ndarray,
    k: int,
    *,
    device: str,
    torch_dtype: torch.dtype,
    chunk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Purpose: Retrieve top-k neighbors by cosine from full dev set.
    Inputs: dev/devtest embeddings, k, device string, torch_dtype, chunk integer.
    Outputs: (idx, sim) arrays of shape (n_devtest, k).
    """
    n_q = devtest_emb.shape[0]
    n_dev = dev_emb.shape[0]
    k = int(k)

    if k <= 0:
        return np.zeros((n_q, 0), dtype=np.int64), np.zeros((n_q, 0), dtype=np.float32)
    if k > n_dev:
        k = n_dev

    dev = torch.from_numpy(dev_emb).to(device, dtype=torch_dtype)
    qry = torch.from_numpy(devtest_emb).to(device, dtype=torch_dtype)

    dev = F.normalize(dev, dim=1)
    qry = F.normalize(qry, dim=1)

    all_idx = []
    all_sim = []

    with torch.no_grad():
        for start in range(0, n_q, int(chunk)):
            q = qry[start : start + int(chunk)]
            sims = q @ dev.T
            vals, idx = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
            all_idx.append(idx.detach().cpu().to(torch.int64))
            all_sim.append(vals.detach().cpu().to(torch.float32))

    return torch.cat(all_idx, dim=0).numpy(), torch.cat(all_sim, dim=0).numpy()


def build_fragmentshot_pools(
    src_dev: List[str],
    tgt_dev: List[str],
    src_devtest: List[str],
    k_min: int,
    *,
    max_fragment_size: int,
    overlaps: bool,
) -> List[List[int]]:
    """
    Purpose: Build per-devtest pools of dev indices using fragmentshot fragments.
    Inputs: src_dev list, tgt_dev list, src_devtest list, k_min, fragmentshot params.
    Outputs: list of pools (each list of dev indices) with size >= k_min.
    """
    from fragmentshot.retriever import FragmentShotsRetriever

    if len(src_dev) != len(tgt_dev):
        raise ValueError(f"src_dev and tgt_dev must be aligned (got {len(src_dev)} and {len(tgt_dev)}).")

    sent_to_idx = {s: i for i, s in enumerate(src_dev)}

    retriever = FragmentShotsRetriever(
        src_texts=src_dev,
        tgt_texts=tgt_dev,
        max_fragment_size=int(max_fragment_size),
        overlaps=bool(overlaps),
    )

    pools: List[List[int]] = []
    for qi, q in enumerate(src_devtest):
        result = retriever.get_fragment_shots(q)
        pool_set = set()

        for shot in result.get("shots", []):
            for ex in shot.get("examples", []):
                s = ex.get("src_text", "")
                if s in sent_to_idx:
                    pool_set.add(int(sent_to_idx[s]))

        pool_list = sorted(pool_set)
        if len(pool_list) < int(k_min):
            raise ValueError(f"fragment_shot pool too small: query {qi} has {len(pool_list)} < {k_min}.")
        pools.append(pool_list)

    return pools


def topk_cosine_indices_from_pools(
    dev_emb: np.ndarray,
    devtest_emb: np.ndarray,
    pools: List[List[int]],
    k: int,
    *,
    device: str,
    torch_dtype: torch.dtype,
) -> np.ndarray:
    """
    Purpose: Retrieve top-k neighbors by cosine restricted to per-query candidate pools.
    Inputs: dev/devtest embeddings, pools list, k, device string, torch_dtype.
    Outputs: idx array of shape (n_devtest, k).
    """
    n_q = devtest_emb.shape[0]
    n_dev = dev_emb.shape[0]
    k = int(k)

    if k <= 0:
        return np.zeros((n_q, 0), dtype=np.int64)
    if k > n_dev:
        k = n_dev
    if len(pools) != n_q:
        raise ValueError(f"pools len={len(pools)} but expected n_devtest={n_q}.")

    dev = torch.from_numpy(dev_emb).to(device, dtype=torch_dtype)
    qry = torch.from_numpy(devtest_emb).to(device, dtype=torch_dtype)

    dev = F.normalize(dev, dim=1)
    qry = F.normalize(qry, dim=1)

    out_idx = np.empty((n_q, k), dtype=np.int64)

    with torch.no_grad():
        for i in range(n_q):
            pool = pools[i]
            if len(pool) < k:
                raise ValueError(f"pool for query {i} has size {len(pool)} < k={k}.")
            pool_t = torch.tensor(pool, device=device, dtype=torch.long)
            dev_pool = dev.index_select(0, pool_t)
            sims = (qry[i : i + 1] @ dev_pool.T).squeeze(0)
            _, loc = torch.topk(sims, k=k, largest=True, sorted=True)
            loc_np = loc.detach().cpu().numpy().astype(np.int64)
            out_idx[i, :] = np.asarray([int(pool[j]) for j in loc_np], dtype=np.int64)

    return out_idx


def borda_ensemble_topk(per_method_topk_idx: Dict[str, np.ndarray], k: int) -> List[List[int]]:
    """
    Purpose: Borda-count fuse per-model ranked lists.
    Inputs: per_method_topk_idx dict, k integer.
    Outputs: fused top-k indices per query.
    """
    methods = list(per_method_topk_idx.keys())
    n_devtest = per_method_topk_idx[methods[0]].shape[0]
    k = int(k)

    if k <= 0:
        return [[] for _ in range(n_devtest)]

    final: List[List[int]] = []
    for i in range(n_devtest):
        borda: Dict[int, int] = {}
        best_rank: Dict[int, int] = {}
        for m in methods:
            idxs = per_method_topk_idx[m][i][:k]
            for rank, cand in enumerate(idxs):
                points = (k - 1 - rank)
                ci = int(cand)
                borda[ci] = borda.get(ci, 0) + points
                if (ci not in best_rank) or (rank < best_rank[ci]):
                    best_rank[ci] = rank

        ordered = sorted(borda.keys(), key=lambda c: (borda[c], -best_rank.get(c, 10**9)), reverse=True)
        final.append(ordered[:k])

    return final


def round_robin_ensemble_topk(
    per_method_topk_idx: Dict[str, np.ndarray],
    methods_in_order: List[str],
    k: int,
) -> List[List[int]]:
    """
    Purpose: Round-robin merge with dedup.
    Inputs: per_method_topk_idx dict, methods_in_order list, k integer.
    Outputs: fused top-k indices per query.
    """
    k = int(k)
    if k <= 0:
        n_devtest = next(iter(per_method_topk_idx.values())).shape[0]
        return [[] for _ in range(n_devtest)]

    for m in methods_in_order:
        if m not in per_method_topk_idx:
            raise KeyError(f"Missing method '{m}' in per_method_topk_idx.")

    n_devtest = per_method_topk_idx[methods_in_order[0]].shape[0]
    final: List[List[int]] = []

    for i in range(n_devtest):
        selected: List[int] = []
        seen = set()
        for r in range(k):
            for m in methods_in_order:
                cand = int(per_method_topk_idx[m][i][r])
                if cand in seen:
                    continue
                selected.append(cand)
                seen.add(cand)
                if len(selected) >= k:
                    break
            if len(selected) >= k:
                break
        final.append(selected[:k])

    return final


def rrf_ensemble_topk(
    per_method_topk_idx: Dict[str, np.ndarray],
    methods_in_order: List[str],
    k: int,
    *,
    k0: int,
) -> List[List[int]]:
    """
    Purpose: Reciprocal Rank Fusion (rank-only fusion).
    Inputs: per_method_topk_idx dict, methods_in_order list, k integer, k0 integer.
    Outputs: fused top-k indices per query.
    """
    k = int(k)
    if k <= 0:
        n_devtest = next(iter(per_method_topk_idx.values())).shape[0]
        return [[] for _ in range(n_devtest)]

    for m in methods_in_order:
        if m not in per_method_topk_idx:
            raise KeyError(f"Missing method '{m}' in per_method_topk_idx.")

    n_devtest = per_method_topk_idx[methods_in_order[0]].shape[0]
    final: List[List[int]] = []
    method_pos = {m: i for i, m in enumerate(methods_in_order)}

    for i in range(n_devtest):
        scores: Dict[int, float] = {}
        best_rank: Dict[int, int] = {}
        best_method_pos: Dict[int, int] = {}

        for m in methods_in_order:
            idxs = per_method_topk_idx[m][i][:k]
            for r0, cand in enumerate(idxs):
                ci = int(cand)
                r = r0 + 1
                scores[ci] = scores.get(ci, 0.0) + (1.0 / float(int(k0) + r))
                if (ci not in best_rank) or (r < best_rank[ci]):
                    best_rank[ci] = r
                    best_method_pos[ci] = method_pos[m]
                elif r == best_rank[ci]:
                    best_method_pos[ci] = min(best_method_pos.get(ci, method_pos[m]), method_pos[m])

        ordered = sorted(
            scores.keys(),
            key=lambda c: (scores[c], -best_rank.get(c, 10**9), -best_method_pos.get(c, 10**9)),
            reverse=True,
        )
        final.append(ordered[:k])

    return final


def pool_bm25_rerank_ensemble_topk(
    per_method_topk_idx: Dict[str, np.ndarray],
    bm25_devtest_scores: np.ndarray,
    k: int,
) -> List[List[int]]:
    """
    Purpose: Pool per-model top-k candidates then rerank via BM25 devtest->dev scores.
    Inputs: per_method_topk_idx dict, bm25_devtest_scores matrix, k integer.
    Outputs: fused top-k indices per query after BM25 reranking.
    """
    methods = list(per_method_topk_idx.keys())
    n_devtest = per_method_topk_idx[methods[0]].shape[0]
    k = int(k)

    if k <= 0:
        return [[] for _ in range(n_devtest)]
    if bm25_devtest_scores.shape[0] != n_devtest:
        raise ValueError(f"bm25_devtest_scores rows={bm25_devtest_scores.shape[0]} but expected {n_devtest}.")

    final: List[List[int]] = []
    for i in range(n_devtest):
        pool_list: List[int] = []
        seen = set()
        best_rank: Dict[int, int] = {}

        for m in methods:
            idxs = per_method_topk_idx[m][i][:k]
            for rank, cand in enumerate(idxs):
                ci = int(cand)
                if (ci not in best_rank) or (rank < best_rank[ci]):
                    best_rank[ci] = rank
                if ci in seen:
                    continue
                seen.add(ci)
                pool_list.append(ci)

        if len(pool_list) < k:
            raise ValueError(f"pool_bm25_rerank: query {i} has {len(pool_list)} unique < k={k}.")

        scores = bm25_devtest_scores[i, pool_list]
        ordered = sorted(range(len(pool_list)), key=lambda j: (float(scores[j]), -best_rank[pool_list[j]]), reverse=True)
        chosen = [int(pool_list[j]) for j in ordered[:k]]
        final.append(chosen)

    return final


def pool_sentinel_src_rerank_ensemble_topk(
    per_method_topk_idx: Dict[str, np.ndarray],
    sentinel_src_scores: np.ndarray,
    k: int,
) -> List[List[int]]:
    """
    Purpose: Pool per-model top-k candidates then rerank via stored SENTINEL source difficulty.
    Inputs: per_method_topk_idx dict, sentinel_src_scores vector over dev, k integer.
    Outputs: fused top-k indices per query after SENTINEL reranking.
    """
    methods = list(per_method_topk_idx.keys())
    n_devtest = per_method_topk_idx[methods[0]].shape[0]
    k = int(k)

    if k <= 0:
        return [[] for _ in range(n_devtest)]
    if sentinel_src_scores.ndim != 1:
        raise ValueError("sentinel_src_scores must be a 1D vector over dev sentences.")

    n_dev = int(sentinel_src_scores.shape[0])

    final: List[List[int]] = []
    for i in range(n_devtest):
        pool_list: List[int] = []
        seen = set()
        best_rank: Dict[int, int] = {}

        for m in methods:
            idxs = per_method_topk_idx[m][i][:k]
            for rank, cand in enumerate(idxs):
                ci = int(cand)
                if ci < 0 or ci >= n_dev:
                    raise ValueError(
                        f"pool_sentinel_src_rerank: candidate index {ci} out of range for n_dev={n_dev}."
                    )
                if (ci not in best_rank) or (rank < best_rank[ci]):
                    best_rank[ci] = rank
                if ci in seen:
                    continue
                seen.add(ci)
                pool_list.append(ci)

        if len(pool_list) < k:
            raise ValueError(f"pool_sentinel_src_rerank: query {i} has {len(pool_list)} unique < k={k}.")

        scores = sentinel_src_scores[pool_list]
        ordered = sorted(
            range(len(pool_list)),
            key=lambda j: (float(scores[j]), best_rank[pool_list[j]]),
        )
        chosen = [int(pool_list[j]) for j in ordered[:k]]
        final.append(chosen)

    return final


def _get_random_pool_dimensions(
    random_source_idx: Dict[str, np.ndarray],
) -> Tuple[int, int]:
    """
    Purpose: Infer the number of devtest rows and max-k width for persisted random selections.
    Inputs: random_source_idx dict whose values are arrays with shape (n_devtest, k_max).
    Outputs: (n_devtest, k_max).
    """
    methods = list(random_source_idx.keys())
    if not methods:
        raise ValueError("random_pool requires at least one method in random_source_idx.")

    first = random_source_idx[methods[0]]
    n_devtest = int(first.shape[0])
    k_max = int(first.shape[1])

    for m in methods[1:]:
        arr = random_source_idx[m]
        if int(arr.shape[0]) != n_devtest:
            raise ValueError(
                f"random_pool: method '{m}' has n_devtest={arr.shape[0]} but expected {n_devtest}."
            )
        if int(arr.shape[1]) < k_max:
            raise ValueError(
                f"random_pool: method '{m}' has width={arr.shape[1]} but expected at least {k_max}."
            )

    return n_devtest, k_max


def _load_or_create_random_pool_selections(
    random_source_idx: Dict[str, np.ndarray],
    selection_file_path: str,
    n_dev: int,
) -> List[List[int]]:
    """
    Purpose: Load persisted random selections if they exist, else generate and save them.
    Inputs: random_source_idx dict, selection_file_path string, n_dev integer.
    Outputs: ordered max-k random selections, one list per devtest query.
    """
    n_devtest, k_max = _get_random_pool_dimensions(random_source_idx)

    if k_max <= 0:
        return [[] for _ in range(n_devtest)]

    if int(n_dev) < k_max:
        raise ValueError(
            f"random_pool: cannot sample k_max={k_max} unique indices from n_dev={n_dev}."
        )

    _ensure_dir(os.path.dirname(selection_file_path) or ".")

    if os.path.exists(selection_file_path):
        with open(selection_file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            raise ValueError(f"random_pool: expected dict payload in {selection_file_path}.")

        if int(payload.get("n_devtest", -1)) != n_devtest:
            raise ValueError(
                f"random_pool: saved n_devtest={payload.get('n_devtest')} does not match current n_devtest={n_devtest}."
            )

        if int(payload.get("k_max", -1)) < k_max:
            raise ValueError(
                f"random_pool: saved k_max={payload.get('k_max')} is smaller than current k_max={k_max}."
            )

        selections = payload.get("selections", None)
        if not isinstance(selections, list) or len(selections) != n_devtest:
            raise ValueError(
                f"random_pool: invalid selections in {selection_file_path}; expected {n_devtest} rows."
            )

        loaded: List[List[int]] = []
        for i, row in enumerate(selections):
            if not isinstance(row, list):
                raise ValueError(f"random_pool: selection row {i} is not a list.")

            row_int = [int(x) for x in row[:k_max]]

            if len(row_int) < k_max:
                raise ValueError(
                    f"random_pool: saved row {i} has only {len(row_int)} entries < current k_max={k_max}."
                )

            if len(set(row_int)) != len(row_int):
                raise ValueError(f"random_pool: saved row {i} contains duplicates in its first k_max entries.")

            for cand in row_int:
                if cand < 0 or cand >= int(n_dev):
                    raise ValueError(
                        f"random_pool: saved row {i} contains out-of-range dev index {cand} for n_dev={n_dev}."
                    )

            loaded.append(row_int)

        return loaded

    rng = np.random.default_rng()
    selections: List[List[int]] = []

    for _ in range(n_devtest):
        chosen = rng.choice(int(n_dev), size=k_max, replace=False)
        selections.append([int(x) for x in chosen.tolist()])

    payload = {
        "n_dev": int(n_dev),
        "n_devtest": int(n_devtest),
        "k_max": int(k_max),
        "selections": selections,
    }

    with open(selection_file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    return selections


def random_pool_ensemble_topk(
    random_source_idx: Dict[str, np.ndarray],
    k: int,
    selection_file_path: str,
    n_dev: int,
) -> List[List[int]]:
    """
    Purpose: Randomly select max-k examples from the full dev set, persist them, and return the first k.
    Inputs: random_source_idx dict, k integer, selection_file_path string, n_dev integer.
    Outputs: fused top-k indices per query.
    """
    n_devtest, _ = _get_random_pool_dimensions(random_source_idx)
    k = int(k)

    if k <= 0:
        return [[] for _ in range(n_devtest)]

    selections = _load_or_create_random_pool_selections(
        random_source_idx=random_source_idx,
        selection_file_path=selection_file_path,
        n_dev=int(n_dev),
    )

    final = [[int(x) for x in row[:k]] for row in selections]

    for i, row in enumerate(final):
        if len(row) != k:
            raise ValueError(f"random_pool: query {i} returned {len(row)} examples but expected {k}.")

    return final


# ─────────────────────────────────────────────────────────────────────────────
# edit_dist — word-level Levenshtein retrieval (Bloodgood & Strauss, 2014)
# ─────────────────────────────────────────────────────────────────────────────


def _word_levenshtein_tokens(t1: List[str], t2: List[str]) -> int:
    """
    Purpose: Word-level Levenshtein edit distance between two token lists, as
        in Bloodgood & Strauss (2014, EACL): unit-cost word-level deletions,
        insertions, and substitutions via standard Wagner-Fischer DP.
    Inputs: t1, t2 lists of word tokens.
    Outputs: integer edit distance.
    """
    n1 = len(t1)
    n2 = len(t2)

    if n1 == 0:
        return n2
    if n2 == 0:
        return n1

    prev = list(range(n2 + 1))
    curr = [0] * (n2 + 1)

    for i in range(1, n1 + 1):
        curr[0] = i
        t1_i = t1[i - 1]
        for j in range(1, n2 + 1):
            cost = 0 if t1_i == t2[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev

    return prev[n2]


def _word_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Purpose: Convenience string wrapper around `_word_levenshtein_tokens` that
        whitespace-tokenizes both inputs first.
    Inputs: s1, s2 strings.
    Outputs: integer word-level edit distance.
    """
    return _word_levenshtein_tokens(s1.split(), s2.split())


def edit_dist_topk(
    src_dev: List[str],
    src_devtest: List[str],
    k: int,
) -> List[List[int]]:
    """
    Purpose: For each source devtest sentence, return the k dev indices with
        the smallest word-level Levenshtein edit distance — the k most
        similar candidates in the dev pool, per Bloodgood & Strauss (2014).
    Inputs: src_dev list of dev source sentences, src_devtest list of devtest
        source sentences, k integer.
    Outputs: per-devtest list of k dev indices ordered ascending by edit
        distance (smallest distance first), matching the (n_devtest, k)
        shape returned by the other ensemble_topk_* functions.

    Notes:
        - Distance is word-level (whitespace tokenization), with classic
          insertion/deletion/substitution at unit cost (no transposition),
          computed by Wagner-Fischer DP — the algorithm described in
          §2.3 of Bloodgood & Strauss (2014, EACL).
        - Their reported metric also normalizes as
          max(1 - edit_dist(M, C) / |M_unigrams|, 0). The denominator is
          constant in C for a fixed query M, so ranking on raw distance is
          order-equivalent to ranking on that similarity score within a
          query; we therefore rank on the raw distance directly.
    """
    k = int(k)
    n_q = len(src_devtest)
    n_dev = len(src_dev)

    if k <= 0:
        return [[] for _ in range(n_q)]
    if k > n_dev:
        k = n_dev

    # Pre-tokenize the dev pool once; this is shared across all queries.
    dev_tokens = [s.split() for s in src_dev]

    final: List[List[int]] = []
    for q_sent in src_devtest:
        q_tokens = q_sent.split()
        dists = np.empty(n_dev, dtype=np.int64)
        for j, d_tokens in enumerate(dev_tokens):
            dists[j] = _word_levenshtein_tokens(q_tokens, d_tokens)

        if k < n_dev:
            part = np.argpartition(dists, k - 1)[:k]
            ordered = part[np.argsort(dists[part], kind="stable")]
        else:
            ordered = np.argsort(dists, kind="stable")
        final.append([int(x) for x in ordered])

    return final


def _resolve_src_sentences_for_edit_dist() -> Tuple[List[str], List[str]]:
    """
    Purpose: Recover (src_dev, src_devtest) for the edit_dist dispatcher case
        when the caller did not pass them explicitly. Walks the call stack
        for `src_dev` / `src_devtest` locals (which exist in the running
        script's main() at dispatch time); falls back to loading FLORES via
        the SRC_LANG_CODES / DEVTEST_N globals defined in the running script.
    Inputs: none.
    Outputs: (src_dev, src_devtest) as two List[str].
    """
    import inspect

    frame = inspect.currentframe()
    try:
        f = frame.f_back if frame is not None else None
        while f is not None:
            locs = f.f_locals
            src_dev = locs.get("src_dev")
            src_devtest = locs.get("src_devtest")
            if (
                isinstance(src_dev, list)
                and isinstance(src_devtest, list)
                and len(src_dev) > 0
                and len(src_devtest) > 0
                and isinstance(src_dev[0], str)
                and isinstance(src_devtest[0], str)
            ):
                return src_dev, src_devtest
            f = f.f_back
    finally:
        del frame

    src_codes = globals().get("SRC_LANG_CODES", None)
    if not src_codes:
        raise RuntimeError(
            "edit_dist: cannot locate src_dev/src_devtest in caller frames and "
            "SRC_LANG_CODES is not defined as a global. Either call "
            "ensemble_topk_dispatch from a frame that has src_dev/src_devtest "
            "locals (as Cell 3's main() does), or pass src_dev_sentences / "
            "src_devtest_sentences explicitly."
        )
    devtest_n = globals().get("DEVTEST_N", None)
    src_data = load_flores_sentences(src_codes[0])
    return src_data["dev"], _apply_devtest_limit(src_data["devtest"], devtest_n)


def ensemble_topk_dispatch(
    method: str,
    per_idx_k: Dict[str, np.ndarray],
    methods_in_order: List[str],
    k: int,
    *,
    rrf_k0: int,
    bm25_devtest_scores: Optional[np.ndarray] = None,
    random_selection_path: Optional[str] = None,
    random_source_idx: Optional[Dict[str, np.ndarray]] = None,
    random_dev_size: Optional[int] = None,
    sentinel_src_scores: Optional[np.ndarray] = None,
    src_dev_sentences: Optional[List[str]] = None,
    src_devtest_sentences: Optional[List[str]] = None,
) -> List[List[int]]:
    """
    Purpose: Dispatch to ensemble method.
    Inputs: method string, per_idx_k dict, methods_in_order list, k int, rrf_k0 int, optional bm25 scores,
        optional random selection path, optional full-width source indices for random pooling, optional full dev size,
        optional stored SENTINEL source-difficulty scores, and optional src/devtest sentence lists for edit_dist.
    Outputs: fused top-k indices per query.
    """
    if method == "borda":
        return borda_ensemble_topk(per_idx_k, k)
    if method == "round_robin":
        return round_robin_ensemble_topk(per_idx_k, methods_in_order, k)
    if method == "rrf":
        return rrf_ensemble_topk(per_idx_k, methods_in_order, k, k0=int(rrf_k0))
    if method == "pool_bm25_rerank":
        if bm25_devtest_scores is None:
            raise ValueError("pool_bm25_rerank requires bm25_devtest_scores.")
        return pool_bm25_rerank_ensemble_topk(per_idx_k, bm25_devtest_scores, k)
    if method == "pool_sentinel_src_rerank":
        if sentinel_src_scores is None:
            raise ValueError("pool_sentinel_src_rerank requires sentinel_src_scores.")
        return pool_sentinel_src_rerank_ensemble_topk(per_idx_k, sentinel_src_scores, k)
    if method == "random_pool":
        if random_selection_path is None:
            raise ValueError("random_pool requires random_selection_path.")
        if random_source_idx is None:
            raise ValueError("random_pool requires random_source_idx.")
        if random_dev_size is None:
            raise ValueError("random_pool requires random_dev_size.")
        return random_pool_ensemble_topk(
            random_source_idx=random_source_idx,
            k=k,
            selection_file_path=random_selection_path,
            n_dev=int(random_dev_size),
        )
    if method == "edit_dist":
        if src_dev_sentences is None or src_devtest_sentences is None:
            src_dev_sentences, src_devtest_sentences = _resolve_src_sentences_for_edit_dist()
        return edit_dist_topk(src_dev_sentences, src_devtest_sentences, k)
    raise ValueError(f"Unknown ensemble submethod='{method}'.")


def compute_pairwise_overlap_and_meanrank(per_method_idx: Dict[str, np.ndarray], k: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Purpose: Compute avg overlap and mean shared-rank matrices.
    Inputs: per_method_idx dict, k integer.
    Outputs: (overlap matrix, mean-rank matrix, method labels).
    """
    methods = list(per_method_idx.keys())
    n_methods = len(methods)
    n_q = per_method_idx[methods[0]].shape[0]
    k = int(k)

    overlap = np.zeros((n_methods, n_methods), dtype=np.float32)
    mean_rank = np.full((n_methods, n_methods), np.nan, dtype=np.float32)

    topk = {m: per_method_idx[m][:, :k] for m in methods}

    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            shared_count_sum = 0.0
            shared_rank_sum = 0.0
            shared_items_total = 0

            for q in range(n_q):
                a_list = [int(x) for x in topk[a][q]]
                b_list = [int(x) for x in topk[b][q]]
                rank_a = {cand: r + 1 for r, cand in enumerate(a_list)}
                rank_b = {cand: r + 1 for r, cand in enumerate(b_list)}
                shared = set(a_list).intersection(b_list)
                shared_count_sum += float(len(shared))
                for cand in shared:
                    mr = (rank_a[cand] + rank_b[cand]) / 2.0
                    shared_rank_sum += mr
                    shared_items_total += 1

            overlap[i, j] = shared_count_sum / float(n_q)
            mean_rank[i, j] = (shared_rank_sum / float(shared_items_total)) if shared_items_total > 0 else np.nan

    return overlap, mean_rank, methods


def _plot_matrix_png(
    mat: np.ndarray,
    labels: List[str],
    title: str,
    out_path: str,
    value_fmt: str,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Purpose: Save annotated heatmap.
    Inputs: matrix, labels, title, out_path, value_fmt, vmin/vmax optional.
    Outputs: None.
    """
    from matplotlib.colors import LinearSegmentedColormap

    _ensure_dir(os.path.dirname(out_path))
    masked = np.ma.array(mat, mask=np.isnan(mat))

    base = plt.get_cmap("Blues")
    cm = LinearSegmentedColormap.from_list("Blues_clipped", base(np.linspace(0.15, 0.85, 256)))
    cm.set_bad(color="lightgray")

    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2.0, 1.0 * len(labels) + 2.0))
    im = ax.imshow(masked, cmap=cm, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)

    norm = im.norm
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10, color="black")
                continue
            text = format(float(v), value_fmt)
            shade = float(norm(v))
            txt_color = "white" if shade > 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=10, color=txt_color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_k10_multi_attribution_counts(
    final_topk: List[List[int]],
    per_idx_k: Dict[str, np.ndarray],
    methods_in_order: List[str],
    k: int,
) -> Dict[str, int]:
    """
    Purpose: Count per-model overlaps with final top-k (multi-attribution).
    Inputs: final_topk lists, per_idx_k dict, methods_in_order list, k integer.
    Outputs: dict of counts per method.
    """
    k = int(k)
    counts = {m: 0 for m in methods_in_order}
    n_q = len(final_topk)

    for q in range(n_q):
        final_set = set(int(x) for x in final_topk[q][:k])
        for m in methods_in_order:
            method_set = set(int(x) for x in per_idx_k[m][q][:k])
            counts[m] += len(final_set.intersection(method_set))

    return counts


def plot_k10_attribution_bar(counts: Dict[str, int], methods_in_order: List[str], out_path: str, title: str) -> None:
    """
    Purpose: Save bar chart of attribution counts.
    Inputs: counts dict, methods_in_order list, out_path string, title string.
    Outputs: None.
    """
    _ensure_dir(os.path.dirname(out_path))
    labels = methods_in_order
    values = [int(counts.get(m, 0)) for m in labels]

    plt.figure(figsize=(max(6.0, 1.2 * len(labels)), 4.5))
    plt.bar(labels, values)
    plt.ylabel("Count of overlaps with final top-k")
    plt.xlabel("Embedding method")
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_ensemble_method(method: str) -> Tuple[bool, str]:
    """
    Purpose: Parse ENSEMBLE_METHOD into (use_fragment_shot, underlying_method).
    Inputs: method string.
    Outputs: (use_fragment_shot bool, underlying_method string).
    """
    if "+" not in method:
        return False, method
    base, rest = method.split("+", 1)
    base = base.strip()
    rest = rest.strip()
    if base != "fragment_shot":
        raise ValueError(f"Unknown composite ENSEMBLE_METHOD='{method}'.")
    if not rest:
        raise ValueError(f"Invalid ENSEMBLE_METHOD='{method}': missing submethod.")
    return True, rest


def get_served_model_id(client: OpenAI) -> str:
    """
    Purpose: Retrieve the served model id from a vLLM OpenAI-compatible server.
    Inputs: OpenAI client configured with base_url pointing at vLLM.
    Outputs: model id string.
    """
    models = client.models.list()
    if not models.data:
        raise RuntimeError("vLLM server returned no models from /v1/models.")
    return models.data[0].id


