# rtrace_reasoning_mt

Reasoning-based machine-translation experiments — a package refactor of the
original notebook `EmbeddingEnsemblingReb5.ipynb`.

The paper studies whether reasoning models (Qwen3 8B/14B/32B, Ministral
8B/14B, Magistral Small 24B) benefit from reasoning traces when translating
from English into low-resource targets (Wolof, Swahili, Mizo, Manipuri,
Telugu, Tamil, Northern Uzbek) under a matched few-shot prompt built by an
ensemble of retrieval methods (RRF, Sentinel, Random, Edit-Distance).

## Layout

```
src/
├── common/                  # shared utilities extracted from the analysis scripts
│   ├── fasttext_compat.py       # NumPy 2.x fasttext.predict patch
│   └── plots.py                 # _collect_legend (used by eval_pipeline + chrfpp_per_sentence)
├── embeddings/
│   └── generate_flores_embeddings.py   # SONAR/LaBSE/mE5/Cohere/MiniLM/BM25/SENTINEL over FLORES-200
├── retrieval/
│   └── retrieval_helpers.py            # prompt templates + top-k + ensemble selection methods
├── generation/
│   ├── vllm_server.py                  # VLLMServerSpec / VLLMServerProcess + OpenAI-client driver
│   └── reasoning_main.py               # entry point: boot vLLM per model, translate every direction
├── evaluation/
│   └── eval_pipeline.py                # corpus BLEU/chrF++/COMET/LID + plots + tables
├── significance/
│   ├── paired_bootstrap_chrfpp.py      # paired bootstrap on chrF++ ON vs OFF
│   └── paired_bootstrap_lid.py         # paired bootstrap on LID accuracy ON vs OFF
└── analysis/
    ├── token_count_inference_budget.py # per-sentence tokenizer counts + Spearman vs chrF++
    └── chrfpp_per_sentence_analysis.py # per-sentence chrF++, correlations, ANOVA, LMM, plots
```

Each script's original module docstring lives at the top of its file.

## Pipeline order

1. `setup.sh` — bootstrap the environment (pip installs, unzip embeddings, guardians-mt-eval, vLLM, etc.).
2. `python -m src.embeddings.generate_flores_embeddings` — produces `flores_embeddings/`.
3. `python -m src.generation.reasoning_main` — generates translation JSONLs under `GEN_ROOT`.
4. `python -m src.evaluation.eval_pipeline` — scores every run, produces plots + tables.
5. Significance:
   - `python -m src.significance.paired_bootstrap_chrfpp`
   - `python -m src.significance.paired_bootstrap_lid`
6. Analysis:
   - `python -m src.analysis.token_count_inference_budget`
   - `python -m src.analysis.chrfpp_per_sentence_analysis`

Every script reads its configuration from top-of-file constants (paths under
`drive/MyDrive/...`, model keys, k-values, ensemble method identifier). Edit
those constants before running in a non-Colab environment. The generation
entrypoint (`reasoning_main`) additionally accepts `RTRACE_*` environment
variables (dataset, model, method, reasoning state, roots, sampling) so SLURM
jobs can parameterize runs without touching code — defaults preserve the
original notebook values.

## WMT24++ evaluation setup

In addition to FLORES-200, the pipeline supports the WMT24++ test set
(`google/wmt24pp`, 998 rows per en→xx language pair — verified via the HF
datasets API). Coverage of our targets: Swahili (`en-sw_TZ`, override with
`RTRACE_WMT_SWH_CONFIG=en-sw_KE`), Tamil (`en-ta_IN`), Telugu (`en-te_IN`);
Wolof / Mizo / Manipuri / Uzbek are not in WMT24++.

- `data/wmt24pp_split.json` — the committed FIXED split (seed 12345): 38
  `is_bad_source` rows (incl. the canary GUID) excluded, then **100 test /
  860 selection-pool** sentences drawn from the remaining 960. Identical
  across every model, language, k, and reasoning run; regenerate only from
  scratch via `scripts/make_wmt24pp_split.py` (never after runs exist).
- `src/data/wmt24pp.py` — loader with the same `{"dev": pool, "devtest":
  test}` contract as the FLORES loader. English sources are identical across
  pairs (verified), so one shared English embedding set serves all targets.
- `src/embeddings/generate_wmt24pp_embeddings.py` — embeds the English pool +
  test with every retrieval method into `wmt24pp_embeddings/` (same layout as
  `flores_embeddings/`); skips files already on disk.
- Run generation with `RTRACE_DATASET=wmt24pp`.

## SLURM (UMIACS Nexus)

`slurm/` mirrors the conventions of the embedding-research repo (requeue +
wall-clock USR1 trap + per-size GPU routing):

- `slurm/generate.sbatch` — one (model × method × reasoning × dataset) cell
  per job; builds the eval-compatible `GEN_ROOT` name, exports `RTRACE_*`,
  requeues itself 5 min before the wall and on scavenger preemption.
- `slurm/wmt24pp_embeddings.sbatch` — one-off embedding job (clip, 1×A6000).
- `slurm/submit_all_generation.sh` — submits the whole grid (48 jobs/dataset)
  with routing: 8B/14B → 1× rtxa6000 (clip, huge-long); 24B/32B → 2× rtxa6000
  TP=2 (clip) or 1× Hopper H100/H200 (`SCAVENGER=1`). WMT24++ generation jobs
  are dependency-chained (`afterok`) behind the embeddings job.

Progress is never lost on preemption: every finished sentence is flushed to
its JSONL immediately, and restarted attempts skip all non-empty stored
translations (`SKIP_EXISTING_TRANSLATIONS=True`), re-doing at most the single
in-flight sentence.

### Quickstart on Nexus (repo in clip-scratch)

The repo MUST live under `/fs/clip-scratch` (home quotas cannot hold the venvs
+ model caches), and needs TWO virtualenvs: `.venv` for generation (vLLM picks
its own torch) and `.venv-emb` for embeddings (sonar-space→fairseq2 caps torch
and cannot coexist with vLLM). Job scripts load `Python3/3.11.11` and activate
the right venv themselves.

```bash
cd /fs/clip-scratch/$USER
git clone https://github.com/Aarushvinod/rtrace_reasoning_mt.git
cd rtrace_reasoning_mt
# secrets (gitignored): create .env with COHERE_API_KEY + HF_TOKEN, chmod 600

module load Python3/3.11.11

# env 1 — generation (vLLM first, alone, wheels only):
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install --only-binary=:all: vllm
python -m pip install -r requirements-gen.txt
deactivate

# env 2 — embeddings (SONAR + Sentinel stack):
python3 -m venv .venv-emb && source .venv-emb/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-emb.txt
git clone https://github.com/SapienzaNLP/guardians-mt-eval.git
python -m pip install -e guardians-mt-eval
deactivate

# smoke test (embeddings + one cheap OFF job):
MODELS="qwen3_8b" STATES="off" METHODS="rrf" bash slurm/submit_all_generation.sh wmt24pp
# full grid (embeddings already on disk):
WMT_EMB_READY=1 bash slurm/submit_all_generation.sh wmt24pp
```

With the repo under `/fs/clip-scratch`, the default `RTRACE_OUT_BASE=runs`
already resolves to scratch storage (`<repo>/runs/`, gitignored) — no env
vars needed at submit time; jobs are self-contained.

## Notes on the refactor

- Non-reasoning-model inference (BLOOM, Mistral-7B, Llama-2-7B, Gemma-7B) was
  in scope for an earlier revision of the notebook and has been dropped
  entirely — that includes both its main cell and the `run_vllm_for_direction`
  driver in the retrieval helpers.
- Shared utilities across the analysis scripts (`ensure_dir`, `slugify`,
  `read_jsonl_translations`, `resolve_model_dirname`, plotting helpers, etc.)
  are not yet centralised. Each script has its own copy verbatim from the
  notebook because the copies diverged in whitespace, docstring style, and
  in a couple of places in dependencies (`direction_display_from_folder`).
  Only `_collect_legend` and `_patch_fasttext_for_numpy2` — the two functions
  that were byte-identical across cells — have moved to `src/common/`.
- Function bodies were not modified. Only cell 8 (non-reasoning main),
  `run_vllm_for_direction` (in the retrieval helpers), and duplicated copies
  of the two centralised functions were removed.

## Requirements

`pip install -r requirements.txt` (or run `bash setup.sh`).
