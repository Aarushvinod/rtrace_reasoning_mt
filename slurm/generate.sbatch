#!/bin/bash
# One (model x method x reasoning-state x dataset) cell of the generation grid
# per job. Output roots follow the eval pipeline's naming convention, so the
# downstream eval/significance/analysis scripts read them without config edits.
#
#   sbatch [routing flags] slurm/generate.sbatch <model_key> <method> <on|off> <flores|wmt24pp>
#   e.g. sbatch --partition=clip --account=clip --qos=huge-long --gres=gpu:rtxa6000:1 \
#          slurm/generate.sbatch qwen3_8b rrf on flores
#
# Use slurm/submit_all_generation.sh to submit the whole grid with per-model
# GPU routing — do not hand-pick flags per job unless debugging.
#
#SBATCH --job-name=rtrace-gen
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
# Default routing = preemptible scavenger on a 48GB Ampere card; submit_all
# overrides per model size (24B/32B go to 2x A6000 on clip or 1x Hopper).
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:1

set -euo pipefail

MODEL_KEY="$1"          # qwen3_8b|qwen3_14b|qwen3_32b|ministral_8b|ministral_14b|magistral_small
METHOD="$2"             # rrf|random|sentinel|edit_dist
STATE="$3"              # on|off
DATASET="${4:-flores}"  # flores|wmt24pp

# OUT_BASE: shared-filesystem root for all run artifacts (JSONLs, traces).
# Point this at lab storage (e.g. /fs/clip-scratch/$USER/rtrace) — NOT /tmp.
OUT_BASE="${RTRACE_OUT_BASE:-runs}"

# Record allocated hardware at the top of the (appended) log: with requeue a
# single log file accumulates every attempt, and the GPU can differ per attempt.
echo "=== attempt $(date -Is) job=$SLURM_JOB_ID node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "model=$MODEL_KEY method=$METHOD state=$STATE dataset=$DATASET out_base=$OUT_BASE"

# Family prefix + method suffix reproduce the folder convention the eval
# pipeline already expects (rrf lives in the base root with NO suffix).
case "$MODEL_KEY" in
  qwen3_*)  FAMILY="Qwen" ;;
  *)        FAMILY="Mistral" ;;
esac
case "$METHOD" in
  rrf)       SUFFIX="" ;;
  random)    SUFFIX="_random" ;;
  sentinel)  SUFFIX="_sentinel" ;;
  edit_dist) SUFFIX="_edit_dist" ;;
  *) echo "unknown method: $METHOD"; exit 2 ;;
esac
case "$STATE" in
  on)  STATE_TITLE="On" ;;
  off) STATE_TITLE="Off" ;;
  *) echo "state must be on|off, got: $STATE"; exit 2 ;;
esac

DS_PREFIX=""
if [ "$DATASET" = "wmt24pp" ]; then
  DS_PREFIX="WMT24PP_"
fi

export RTRACE_DATASET="$DATASET"
export RTRACE_MODELS="$MODEL_KEY"
export RTRACE_METHOD="$METHOD"
export RTRACE_REASONING="$STATE"
export RTRACE_GEN_ROOT="$OUT_BASE/${DS_PREFIX}${FAMILY}_All_Reasoning_${STATE_TITLE}${SUFFIX}"
export RTRACE_REASONING_ROOT="$OUT_BASE/${DS_PREFIX}reasoning_traces_${FAMILY}_${METHOD}_${STATE}"
export RTRACE_ANALYSIS_DIR="$OUT_BASE/retrieval_overlap_analysis"
# Embedding roots: flores runs read the FLORES embeddings; wmt24pp runs read
# the shared-English WMT24++ embeddings produced by wmt24pp_embeddings.sbatch.
if [ "$DATASET" = "wmt24pp" ]; then
  export RTRACE_EMB_ROOT="${RTRACE_WMT_EMB_ROOT:-$OUT_BASE/wmt24pp_embeddings}"
else
  export RTRACE_EMB_ROOT="${RTRACE_FLORES_EMB_ROOT:-$OUT_BASE/flores_embeddings}"
fi
# Mistral reasoning-on runs follow the Magistral card (temp 0.7 / top_p 0.95);
# Qwen3 keeps the state-dependent defaults baked into reasoning_main.
if [ "$FAMILY" = "Mistral" ] && [ "$STATE" = "on" ]; then
  export RTRACE_TEMPERATURE="${RTRACE_TEMPERATURE:-0.7}"
  export RTRACE_TOP_P="${RTRACE_TOP_P:-0.95}"
fi

# Keep the HF model/dataset cache off the node-local disk so requeued attempts
# (possibly on a different node) reuse the same downloads.
export HF_HOME="${HF_HOME:-$OUT_BASE/hf_cache}"

# Node-unique vLLM port: two jobs sharing a node must not both bind 8000.
export RTRACE_PORT_BASE=$(( 10000 + (SLURM_JOB_ID % 500) * 100 ))

# AUTO-RESUME across the wall limit: SIGUSR1 arrives 5 min before timeout, the
# trap requeues THIS job id, and the restarted attempt skips every sentence
# already flushed to the output JSONLs (SKIP_EXISTING_TRANSLATIONS=True +
# per-sentence flush), so nothing is recomputed. Scavenger preemption is
# handled the same way — SLURM requeues preempted jobs with identical args.
# (srun is backgrounded so `wait` is interruptible by the trap.)
trap 'echo "[wall] SIGUSR1 -> requeue $SLURM_JOB_ID (resume skips finished sentences)"; scontrol requeue "$SLURM_JOB_ID" || true' USR1

srun python -u -m src.generation.reasoning_main &
wait
