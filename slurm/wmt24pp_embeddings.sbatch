#!/bin/bash
# Compute the WMT24++ English-side retrieval embeddings (sonar, labse, e5,
# cohere, MiniLM, bm25 + sentinel_src) for the fixed 860-pool / 100-test split.
# One short GPU job; every .bin is skipped if already on disk, so requeues and
# reruns are free. Must finish before any wmt24pp generation job starts —
# submit_all_generation.sh chains that dependency automatically.
#
#   sbatch slurm/wmt24pp_embeddings.sbatch
#
# Requires COHERE_API_KEY in the submission environment (sbatch exports it).
#
#SBATCH --job-name=rtrace-wmt-emb
#SBATCH --cpus-per-task=8
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
export HF_HOME="${HF_HOME:-$OUT_BASE/hf_cache}"

echo "=== attempt $(date -Is) job=$SLURM_JOB_ID node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "emb_root=$RTRACE_WMT_EMB_ROOT"

srun python -u -m src.embeddings.generate_wmt24pp_embeddings
