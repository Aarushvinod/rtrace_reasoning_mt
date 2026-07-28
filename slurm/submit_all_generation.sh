#!/bin/bash
# Dispatch the FULL generation grid as SLURM jobs with per-model GPU routing:
# 6 models x 4 methods x 2 reasoning states = 48 jobs per dataset, every job
# independently resumable (requeue + per-sentence flush + skip-existing).
#
#   bash slurm/submit_all_generation.sh                # FLORES grid, clip-first routing
#   bash slurm/submit_all_generation.sh wmt24pp        # WMT24++ grid (chains embeddings job)
#   SCAVENGER=1 bash slurm/submit_all_generation.sh    # route everything to scavenger
#   MODELS="qwen3_32b" METHODS="rrf" STATES="on" bash slurm/submit_all_generation.sh
#
# PER-MODEL GPU ROUTING (bf16 weights + vLLM gpu_memory_utilization=0.90):
#   8B  (~17GB)  -> 1x rtxa6000 (48GB, clip)
#   14B (~29GB)  -> 1x rtxa6000 (48GB, clip)
#   24B (~47GB)  -> 2x rtxa6000 TP=2 (clip)   | SCAVENGER=1: 1x Hopper (80GB+)
#   32B (~66GB)  -> 2x rtxa6000 TP=2 (clip)   | SCAVENGER=1: 1x Hopper (80GB+)
# reasoning_main derives tensor-parallel size from torch.cuda.device_count(),
# so the gres line alone controls TP.
set -euo pipefail

DATASET="${1:-flores}"
MODELS="${MODELS:-qwen3_8b ministral_8b qwen3_14b ministral_14b magistral_small qwen3_32b}"
METHODS="${METHODS:-rrf random sentinel edit_dist}"
STATES="${STATES:-on off}"
SCAVENGER="${SCAVENGER:-0}"

# Routing flag sets. clip = non-preemptible lab pool (A6000s); scavenger =
# preemptible but bigger/faster pool (typed A6000 for small models so the
# scheduler never hands us an 11GB 2080Ti; Hopper H100/H200 for 24B/32B).
CLIP_1GPU=(--partition=clip --account=clip --qos=huge-long --gres=gpu:rtxa6000:1)
CLIP_2GPU=(--partition=clip --account=clip --qos=huge-long --gres=gpu:rtxa6000:2)
SCAV_1GPU=(--partition=scavenger --account=scavenger --qos=scavenger --gres=gpu:rtxa6000:1)
SCAV_HOPPER=(--partition=scavenger --account=scavenger --qos=scavenger --constraint=Hopper --gres=gpu:1)

flags_for() {  # set FL[] to the sbatch routing flags for one model key
  local m="$1"
  if [ "$SCAVENGER" = "1" ]; then
    case "$m" in
      magistral_small|qwen3_32b) FL=("${SCAV_HOPPER[@]}") ;;
      *)                         FL=("${SCAV_1GPU[@]}") ;;
    esac
  else
    case "$m" in
      magistral_small|qwen3_32b) FL=("${CLIP_2GPU[@]}") ;;
      *)                         FL=("${CLIP_1GPU[@]}") ;;
    esac
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
  flags_for "$m"
  for meth in $METHODS; do
    for st in $STATES; do
      jid=$(sbatch --parsable "${FL[@]}" "${DEP_FLAG[@]}" \
            --job-name="rt-${m}-${meth}-${st}-${DATASET}" \
            slurm/generate.sbatch "$m" "$meth" "$st" "$DATASET")
      echo "$jid  $m $meth $st $DATASET  -> ${FL[*]}"
      N=$((N + 1))
    done
  done
done
echo "submitted $N generation jobs for dataset=$DATASET (monitor: squeue -u \$USER)"
