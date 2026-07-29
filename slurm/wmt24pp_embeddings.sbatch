#!/bin/bash
# Compute the WMT24++ English-side retrieval embeddings for the fixed
# 860-pool / 100-test split, in TWO PASSES with different virtualenvs:
#
#   pass 1 (.venv-emb):  sonar, labse, e5, cohere, MiniLM, bm25
#   pass 2 (.venv-sent): sentinel_src difficulty scores
#
# Two envs because the stacks are irreconcilable: SONAR's fairseq2n requires
# numpy>=2.2 while Sentinel's unbabel-comet requires numpy<2. Every .bin is
# skipped if already on disk, so requeues and reruns are free. Must finish
# before any wmt24pp generation job starts — submit_all_generation.sh chains
# that dependency automatically.
#
#   sbatch slurm/wmt24pp_embeddings.sbatch
#
# Requires COHERE_API_KEY in .env (pass 1 fails fast without it).
#
#SBATCH --job-name=rtrace-wmt-emb
# clip's default QoS caps jobs at 4 CPUs / 32GB / 1 GPU — stay inside it
# (embedding is GPU/API-bound; 4 CPUs is plenty).
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa6000:1

set -euo pipefail

module load "${RTRACE_PYTHON_MODULE:-Python3/3.11.11}" 2>/dev/null || true

# Secrets: the cohere embedding method REQUIRES COHERE_API_KEY; sourced from
# the gitignored env file (.env at repo root, or RTRACE_ENV_FILE).
ENV_FILE="${RTRACE_ENV_FILE:-.env}"
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
fi
if [ -z "${COHERE_API_KEY:-}" ]; then
  echo "FATAL: COHERE_API_KEY not set — the cohere method cannot run. Place your env file at .env or set RTRACE_ENV_FILE."; exit 3
fi

OUT_BASE="${RTRACE_OUT_BASE:-runs}"
export RTRACE_WMT_EMB_ROOT="${RTRACE_WMT_EMB_ROOT:-$OUT_BASE/wmt24pp_embeddings}"
export HF_HOME="${RTRACE_HF_HOME:-$OUT_BASE/hf_cache}"
export XDG_CACHE_HOME="${RTRACE_XDG_CACHE:-$OUT_BASE/xdg_cache}"

echo "=== attempt $(date -Is) job=$SLURM_JOB_ID node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "emb_root=$RTRACE_WMT_EMB_ROOT"

# Pass 1 — dense retrieval methods + bm25 (SONAR stack env). Subshell keeps
# each venv activation isolated.
(
  . "${RTRACE_VENV_EMB:-.venv-emb}/bin/activate"
  export RTRACE_EMB_METHODS="sonar,labse,e5,cohere,MiniLM,bm25"
  srun python -u -m src.embeddings.generate_wmt24pp_embeddings
)

# Pass 2 — SENTINEL source-difficulty scores (comet stack env).
(
  . "${RTRACE_VENV_SENT:-.venv-sent}/bin/activate"
  export RTRACE_EMB_METHODS="sentinel_src"
  srun python -u -m src.embeddings.generate_wmt24pp_embeddings
)

echo "both embedding passes complete."
