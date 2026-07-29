#!/bin/bash
# LANGUAGE-PARALLEL reasoning-ON generation for models that fit ONE GPU
# (8B/14B): a single job holds N GPUs on one node, boots one single-GPU vLLM
# server per GPU, and drives each target language on its own GPU stream.
# Within a stream, ALL selection methods run sequentially (method is not
# fanned out across jobs). 5 languages / 5 GPUs -> one language per GPU;
# with N < #langs the extras round-robin onto earlier GPUs.
#
#   sbatch [routing flags] [--gres=gpu:rtxa6000:N] \
#     slurm/generate_langpar.sbatch <model_key> <methods-csv|all> <flores|wmt24pp>
#
# NOT for magistral_small / qwen3_32b — their weights exceed one 48GB card;
# those run as single-Hopper generate.sbatch jobs (all methods+langs inside).
#
#SBATCH --job-name=rtrace-langpar
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
# Default routing = clip huge-long with 4x A6000 (safe if clip nodes are
# 4-GPU; bump to :5 at submit time if show_nodes confirms >=5 GPUs/node).
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=huge-long
#SBATCH --gres=gpu:rtxa6000:4

set -euo pipefail

MODEL_KEY="$1"          # qwen3_8b|qwen3_14b|ministral_8b|ministral_14b
METHODS_ARG="$2"        # "all" or csv subset of rrf,random,sentinel,edit_dist
DATASET="${3:-flores}"  # flores|wmt24pp

case "$MODEL_KEY" in
  magistral_small|qwen3_32b)
    echo "refusing: $MODEL_KEY does not fit one 48GB GPU; use generate.sbatch on a Hopper"; exit 2 ;;
esac

if [ "$METHODS_ARG" = "all" ]; then
  METHODS_LIST="rrf random sentinel edit_dist"
else
  METHODS_LIST="${METHODS_ARG//,/ }"
fi

# Secrets (HF_TOKEN for gated Mistral repos) from the gitignored env file.
ENV_FILE="${RTRACE_ENV_FILE:-.env}"
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
fi

OUT_BASE="${RTRACE_OUT_BASE:-runs}"

echo "=== attempt $(date -Is) job=$SLURM_JOB_ID node=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "model=$MODEL_KEY methods=[$METHODS_LIST] dataset=$DATASET out_base=$OUT_BASE"

case "$MODEL_KEY" in
  qwen3_*)  FAMILY="Qwen" ;;
  *)        FAMILY="Mistral" ;;
esac
DS_PREFIX=""
[ "$DATASET" = "wmt24pp" ] && DS_PREFIX="WMT24PP_"

suffix_for() {
  case "$1" in
    rrf)       echo "" ;;
    random)    echo "_random" ;;
    sentinel)  echo "_sentinel" ;;
    edit_dist) echo "_edit_dist" ;;
    *) echo "unknown method: $1" >&2; exit 2 ;;
  esac
}

# Shared env for every per-GPU worker (method-specific vars set per iteration).
export RTRACE_DATASET="$DATASET"
export RTRACE_MODELS="$MODEL_KEY"
export RTRACE_REASONING="on"
export RTRACE_ANALYSIS_DIR="$OUT_BASE/retrieval_overlap_analysis"
if [ "$DATASET" = "wmt24pp" ]; then
  export RTRACE_EMB_ROOT="${RTRACE_WMT_EMB_ROOT:-$OUT_BASE/wmt24pp_embeddings}"
else
  export RTRACE_EMB_ROOT="${RTRACE_FLORES_EMB_ROOT:-$OUT_BASE/flores_embeddings}"
fi
if [ "$FAMILY" = "Mistral" ]; then
  export RTRACE_TEMPERATURE="${RTRACE_TEMPERATURE:-0.7}"
  export RTRACE_TOP_P="${RTRACE_TOP_P:-0.95}"
fi
export HF_HOME="${HF_HOME:-$OUT_BASE/hf_cache}"

# Language list: env override, else the dataset's default set.
if [ -n "${RTRACE_TGT_LANGS:-}" ]; then
  LANGS_CSV="$RTRACE_TGT_LANGS"
elif [ "$DATASET" = "wmt24pp" ]; then
  LANGS_CSV="cat_Latn,zul_Latn,mal_Mlym,slk_Latn,isl_Latn"
else
  LANGS_CSV="wol_Latn,swh_Latn,lus_Latn,mni_Beng,tel_Telu,tam_Taml,uzn_Latn"
fi
IFS=',' read -r -a LANGS <<< "$LANGS_CSV"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NGPU="${#GPUS[@]}"
echo "gpus=[${CUDA_VISIBLE_DEVICES:-0}] ngpu=$NGPU langs=${LANGS_CSV}"

# Round-robin languages onto GPUs: GPU i gets langs i, i+N, i+2N, ...
declare -a SHARD
for gi in $(seq 0 $((NGPU - 1))); do SHARD[$gi]=""; done
for li in "${!LANGS[@]}"; do
  gi=$((li % NGPU))
  SHARD[$gi]="${SHARD[$gi]:+${SHARD[$gi]},}${LANGS[$li]}"
done

# Pre-download the weights ONCE so N workers don't race the HF download.
python -u -c "
from huggingface_hub import snapshot_download
from src.generation.reasoning_main import MODEL_REGISTRY
snapshot_download(MODEL_REGISTRY['$MODEL_KEY'][0])
"

trap 'echo "[wall] SIGUSR1 -> requeue $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID" || true' USR1

# Each worker: its own GPU + language shard, looping every method
# sequentially. Ports are unique per (job, gpu, method-iteration).
worker() {
  local gi="$1" gpu="$2" shard="$3"
  local mi=0
  for METHOD in $METHODS_LIST; do
    local SUFFIX
    SUFFIX="$(suffix_for "$METHOD")"
    CUDA_VISIBLE_DEVICES="$gpu" \
    RTRACE_TGT_LANGS="$shard" \
    RTRACE_METHOD="$METHOD" \
    RTRACE_GEN_ROOT="$OUT_BASE/${DS_PREFIX}${FAMILY}_All_Reasoning_On${SUFFIX}" \
    RTRACE_REASONING_ROOT="$OUT_BASE/${DS_PREFIX}reasoning_traces_${FAMILY}_${METHOD}_on" \
    RTRACE_PORT_BASE=$(( 10000 + (SLURM_JOB_ID % 500) * 100 + gi * 10 + mi )) \
      python -u -m src.generation.reasoning_main
    mi=$((mi + 1))
  done
}

PIDS=()
for gi in $(seq 0 $((NGPU - 1))); do
  [ -z "${SHARD[$gi]}" ] && continue
  WLOG="slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}-gpu${gi}.out"
  echo "gpu ${GPUS[$gi]} -> langs=${SHARD[$gi]} methods=[$METHODS_LIST]  (log: $WLOG)"
  worker "$gi" "${GPUS[$gi]}" "${SHARD[$gi]}" >> "$WLOG" 2>&1 &
  PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=1
done
exit $FAIL
