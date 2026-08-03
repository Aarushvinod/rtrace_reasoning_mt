#!/bin/bash
# One (model x reasoning-state x dataset) cell per job; ALL example-selection
# methods run sequentially inside the job (method is not fanned out across
# jobs — GPU supply is the constraint, wall time is cheap and resumable).
# Output roots follow the eval pipeline's naming convention per method.
#
#   sbatch [routing flags] slurm/generate.sbatch <model_key> <methods-csv|all> <on|off> <flores|wmt24pp>
#   e.g. sbatch --partition=scavenger --account=scavenger --qos=scavenger \
#          --constraint=Hopper --gres=gpu:1 \
#          slurm/generate.sbatch qwen3_32b all on wmt24pp
#
# Use slurm/submit_all_generation.sh to submit the whole grid.
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
# overrides per model size (24B/32B go to 1x Hopper).
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa6000:1

set -euo pipefail

MODEL_KEY="$1"          # qwen3_8b|qwen3_14b|qwen3_32b|ministral_8b|ministral_14b|magistral_small
METHODS_ARG="$2"        # "all" or csv subset of rrf,random,sentinel,edit_dist
STATE="$3"              # on|off
DATASET="${4:-flores}"  # flores|wmt24pp

if [ "$METHODS_ARG" = "all" ]; then
  METHODS_LIST="rrf random sentinel edit_dist"
else
  METHODS_LIST="${METHODS_ARG//,/ }"
fi

# Python env: self-contained so requeued attempts (possibly submitted from a
# dead shell) always resolve the same interpreter. Generation uses .venv
# (vLLM stack); override with RTRACE_VENV.
module load "${RTRACE_PYTHON_MODULE:-Python3/3.11.11}" 2>/dev/null || true
VENV="${RTRACE_VENV:-.venv}"
if [ -f "$VENV/bin/activate" ]; then
  . "$VENV/bin/activate"
fi

# Secrets (COHERE_API_KEY, HF_TOKEN) come from an env file that lives OUTSIDE
# git (.env is gitignored; chmod 600 it on the cluster). Never inline keys here.
ENV_FILE="${RTRACE_ENV_FILE:-.env}"
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
fi

# OUT_BASE: shared-filesystem root for all run artifacts (JSONLs, traces).
# Point this at lab storage (e.g. /fs/clip-scratch/$USER/rtrace) — NOT /tmp.
OUT_BASE="${RTRACE_OUT_BASE:-runs}"

echo "=== attempt $(date -Is) job=$SLURM_JOB_ID node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "model=$MODEL_KEY methods=[$METHODS_LIST] state=$STATE dataset=$DATASET out_base=$OUT_BASE"

case "$MODEL_KEY" in
  qwen3_*)  FAMILY="Qwen" ;;
  *)        FAMILY="Mistral" ;;
esac
case "$STATE" in
  on)  STATE_TITLE="On" ;;
  off) STATE_TITLE="Off" ;;
  *) echo "state must be on|off, got: $STATE"; exit 2 ;;
esac
DS_PREFIX=""
[ "$DATASET" = "wmt24pp" ] && DS_PREFIX="WMT24PP_"

suffix_for() {  # eval-compatible root suffix per method (rrf = bare root)
  case "$1" in
    rrf)       echo "" ;;
    random)    echo "_random" ;;
    sentinel)  echo "_sentinel" ;;
    edit_dist) echo "_edit_dist" ;;
    *) echo "unknown method: $1" >&2; exit 2 ;;
  esac
}

export RTRACE_DATASET="$DATASET"
export RTRACE_MODELS="$MODEL_KEY"
export RTRACE_REASONING="$STATE"
export RTRACE_ANALYSIS_DIR="$OUT_BASE/retrieval_overlap_analysis"
if [ "$DATASET" = "wmt24pp" ]; then
  export RTRACE_EMB_ROOT="${RTRACE_WMT_EMB_ROOT:-$OUT_BASE/wmt24pp_embeddings}"
else
  export RTRACE_EMB_ROOT="${RTRACE_FLORES_EMB_ROOT:-$OUT_BASE/flores_embeddings}"
fi
# Mistral cards recommend one sampling config per model with no thinking/
# non-thinking split — apply in BOTH states. ministral_14b's card says
# temperature=1.0; the rest of the family uses 0.7. top_p is 0.95 across
# the family. Qwen keeps its state-conditional defaults from reasoning_main.
if [ "$FAMILY" = "Mistral" ]; then
  if [ "$MODEL_KEY" = "ministral_14b" ]; then
    export RTRACE_TEMPERATURE="${RTRACE_TEMPERATURE:-1.0}"
  else
    export RTRACE_TEMPERATURE="${RTRACE_TEMPERATURE:-0.7}"
  fi
  export RTRACE_TOP_P="${RTRACE_TOP_P:-0.95}"
fi
# Mistral REASONING gets a larger window (28672 + 12288 headroom = 40960)
# so marginal traces finish instead of dying at the ceiling, and a 5-retry
# cap since each dud costs a full-budget generation. At this window only
# ministral_8b fits a 48GB card; 14b/magistral need Hopper-class GPUs.
if [ "$FAMILY" = "Mistral" ] && [ "$STATE" = "on" ]; then
  export RTRACE_MAX_NEW_TOKENS="${RTRACE_MAX_NEW_TOKENS:-28672}"
  export RTRACE_MAX_MODEL_LEN="${RTRACE_MAX_MODEL_LEN:-40960}"
  export RTRACE_MAX_RETRIES="${RTRACE_MAX_RETRIES:-5}"
fi
# Caches PINNED to OUT_BASE (clip-scratch), deliberately ignoring any
# login-shell HF_HOME — ambient values often point at the small nexus-scratch
# quota, which cannot hold ~200GB of model weights (this exact failure mode
# killed the first full submission). RTRACE_HF_HOME overrides intentionally.
export HF_HOME="${RTRACE_HF_HOME:-$OUT_BASE/hf_cache}"
# Compile caches (triton / vLLM torch.compile) default under ~/.cache — also
# quota-constrained; route them to scratch too.
export XDG_CACHE_HOME="${RTRACE_XDG_CACHE:-$OUT_BASE/xdg_cache}"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"

# Multi-GPU (tensor-parallel) jobs on PCIe A6000/A5000 nodes can deadlock in
# NCCL peer-to-peer setup during engine init — stage TP traffic through host
# memory (no-op for single-GPU jobs; override with NCCL_P2P_DISABLE=0).
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

# AUTO-RESUME: SIGUSR1 arrives 5 min before the wall; the trap requeues this
# job. The restarted attempt re-runs the method loop from the top — completed
# methods fast-forward via skip-existing in seconds, and the interrupted
# method resumes at its first unfinished sentence. Scavenger preemption is
# handled identically by SLURM's automatic requeue.
trap 'echo "[wall] SIGUSR1 -> requeue $SLURM_JOB_ID (resume skips finished work)"; scontrol requeue "$SLURM_JOB_ID" || true' USR1

MI=0
for METHOD in $METHODS_LIST; do
  SUFFIX="$(suffix_for "$METHOD")"
  export RTRACE_METHOD="$METHOD"
  export RTRACE_GEN_ROOT="$OUT_BASE/${DS_PREFIX}${FAMILY}_All_Reasoning_${STATE_TITLE}${SUFFIX}"
  export RTRACE_REASONING_ROOT="$OUT_BASE/${DS_PREFIX}reasoning_traces_${FAMILY}_${METHOD}_${STATE}"
  # Node-unique port, bumped per method iteration to dodge TIME_WAIT rebinds.
  export RTRACE_PORT_BASE=$(( 10000 + (SLURM_JOB_ID % 500) * 100 + MI ))
  echo "--- method=$METHOD gen_root=$RTRACE_GEN_ROOT ---"
  srun python -u -m src.generation.reasoning_main &
  wait $!
  MI=$((MI + 1))
done
echo "all methods complete."
