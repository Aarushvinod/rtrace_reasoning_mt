#!/bin/bash
# Dispatch the FULL generation grid as SLURM jobs with per-model, per-state GPU
# routing: 6 models x 4 methods x 2 reasoning states = 48 jobs per dataset,
# every job independently resumable (requeue + per-sentence flush + skip-existing).
#
#   bash slurm/submit_all_generation.sh                # FLORES grid
#   bash slurm/submit_all_generation.sh wmt24pp        # WMT24++ grid (chains embeddings job)
#   SCAVENGER=1 bash slurm/submit_all_generation.sh    # force everything to scavenger
#   CLIP_ONLY=1 bash slurm/submit_all_generation.sh    # keep big models on clip 2xA6000
#   MODELS="qwen3_32b" METHODS="rrf" STATES="on" bash slurm/submit_all_generation.sh
#
# DEFAULT ROUTING (hybrid — clip for what fits, scavenger Hopper for what doesn't):
#   8B/14B  x ON   -> clip 1x rtxa6000, qos huge-long   (long jobs, non-preemptible)
#   8B/14B  x OFF  -> clip 1x rtxa6000, qos default     (~1-2h; keeps huge-long slots free)
#   24B/32B x any  -> scavenger 1x Hopper (H100/H200)   (66GB weights need >48GB; H100
#                     halves wall time vs 2xA6000 TP=2; preemption costs <= 1 sentence)
# qwen3_32b on a 1x H100 (80GB) additionally caps context at 32768 (30000 new
# tokens) — 66GB of weights leave too little KV budget for 40960. Roomier on
# H200; harmless there. The clip fallback (2x rtxa6000, TP=2) keeps 40960.
# reasoning_main derives tensor-parallel size from torch.cuda.device_count(),
# so the gres line alone controls TP.
set -euo pipefail

DATASET="${1:-flores}"
MODELS="${MODELS:-qwen3_8b ministral_8b qwen3_14b ministral_14b magistral_small qwen3_32b}"
METHODS="${METHODS:-rrf random sentinel edit_dist}"
STATES="${STATES:-on off}"
SCAVENGER="${SCAVENGER:-0}"
CLIP_ONLY="${CLIP_ONLY:-0}"

CLIP_1GPU_LONG=(--partition=clip --account=clip --qos=huge-long --gres=gpu:rtxa6000:1)
CLIP_1GPU_SHORT=(--partition=clip --account=clip --qos=default --gres=gpu:rtxa6000:1)
CLIP_2GPU=(--partition=clip --account=clip --qos=huge-long --gres=gpu:rtxa6000:2)
SCAV_1GPU=(--partition=scavenger --account=scavenger --qos=scavenger --gres=gpu:rtxa6000:1)
SCAV_HOPPER=(--partition=scavenger --account=scavenger --qos=scavenger --constraint=Hopper --gres=gpu:1)

is_big() {  # models whose bf16 weights exceed one 48GB A6000
  case "$1" in magistral_small|qwen3_32b) return 0 ;; *) return 1 ;; esac
}

flags_for() {  # set FL[] (sbatch routing) + EXPORTS (extra --export k=v list) for (model, state)
  local m="$1" st="$2"
  EXPORTS=""
  if [ "$SCAVENGER" = "1" ]; then
    if is_big "$m"; then FL=("${SCAV_HOPPER[@]}"); else FL=("${SCAV_1GPU[@]}"); fi
  elif [ "$CLIP_ONLY" = "1" ]; then
    if is_big "$m"; then FL=("${CLIP_2GPU[@]}"); else
      if [ "$st" = "on" ]; then FL=("${CLIP_1GPU_LONG[@]}"); else FL=("${CLIP_1GPU_SHORT[@]}"); fi
    fi
  else
    # hybrid default
    if is_big "$m"; then
      FL=("${SCAV_HOPPER[@]}")
    elif [ "$st" = "on" ]; then
      FL=("${CLIP_1GPU_LONG[@]}")
    else
      FL=("${CLIP_1GPU_SHORT[@]}")
    fi
  fi
  # 32B on a single Hopper: cap context so the KV allocation fits an 80GB H100.
  if [ "$m" = "qwen3_32b" ] && [[ " ${FL[*]} " == *" --constraint=Hopper "* ]]; then
    EXPORTS=",RTRACE_MAX_MODEL_LEN=32768,RTRACE_MAX_NEW_TOKENS=30000"
  fi
}

# WMT24++ runs need the shared English embeddings first: submit the embedding
# job once and make every generation job depend on it (afterok). Skip with
# WMT_EMB_READY=1 once the embeddings exist on disk.
DEP_FLAG=()
if [ "$DATASET" = "wmt24pp" ] && [ "${WMT_EMB_READY:-0}" != "1" ]; then
  EMB_JOB=$(sbatch --parsable slurm/wmt24pp_embeddings.sbatch)
  echo "wmt24pp embeddings: $EMB_JOB"
  DEP_FLAG=(--dependency=afterok:"$EMB_JOB")
fi

N=0
for m in $MODELS; do
  for meth in $METHODS; do
    for st in $STATES; do
      flags_for "$m" "$st"
      jid=$(sbatch --parsable "${FL[@]}" "${DEP_FLAG[@]}" \
            --export=ALL"$EXPORTS" \
            --job-name="rt-${m}-${meth}-${st}-${DATASET}" \
            slurm/generate.sbatch "$m" "$meth" "$st" "$DATASET")
      echo "$jid  $m $meth $st $DATASET  -> ${FL[*]}${EXPORTS:+  [$EXPORTS]}"
      N=$((N + 1))
    done
  done
done
echo "submitted $N generation jobs for dataset=$DATASET (monitor: squeue -u \$USER)"
