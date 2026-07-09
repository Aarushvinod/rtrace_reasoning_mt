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
those constants before running in a non-Colab environment.

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
