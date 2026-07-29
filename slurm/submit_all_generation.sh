#!/bin/bash
# Dispatch the generation grid as ONE JOB PER (MODEL x REASONING STATE) —
# 12 jobs total. Methods are NOT fanned out: every job loops all four
# selection methods internally. Languages parallelize across GPUs inside the
# small-model reasoning-on jobs.
#
#   ON, 8B/14B   -> language-parallel job: N GPUs on one node, one single-GPU
#                   vLLM server per GPU, one language per GPU stream, methods
#                   sequential within each stream (generate_langpar.sbatch).
#   ON, 24B/32B  -> one single-Hopper job per model: all methods + languages
#                   sequential; rolls across ~2 walls via auto-requeue+resume.
#   OFF, all     -> one single-GPU job per model (short; methods sequential).
#
#   bash slurm/submit_all_generation.sh                 # FLORES grid
#   bash slurm/submit_all_generation.sh wmt24pp         # WMT24++ (chains embeddings)
#   NGPU=5 bash slurm/submit_all_generation.sh wmt24pp  # 5 GPUs per langpar job
#   SCAVENGER=1 ...                                     # small models to scavenger too
#   CLIP_ONLY=1 ...                                     # big models to clip 2xA6000 TP=2
#   MODELS="qwen3_32b" METHODS="rrf,random" STATES="on" ...   # grid subsets
set -euo pipefail

DATASET="${1:-flores}"
MODELS="${MODELS:-qwen3_8b ministral_8b qwen3_14b ministral_14b magistral_small qwen3_32b}"
METHODS="${METHODS:-all}"        # passed INTO each job (csv or "all"), not fanned out
STATES="${STATES:-on off}"
SCAVENGER="${SCAVENGER:-0}"
CLIP_ONLY="${CLIP_ONLY:-0}"
# NGPU: GPUs per language-parallel job. 5 = one GPU per WMT24++ language.
# Verified against show_nodes: scavenger has many nodes with >=5 free A6000s
# (tron00-05 6-8/node, cml30 8, vulcan29-32/45 8, cbcb27 7); on clip ONLY
# clip13 (8x A6000) can host a 5-GPU job (clip12 maxes at 4, clip05/06 at 2).
NGPU="${NGPU:-5}"

# LANGPAR default = scavenger typed-A6000: clip owns just 16 A6000s TOTAL
# (2+2+4+8 across clip05/06/12/13), so four multi-GPU langpar jobs there would
# monopolize the lab pool; scavenger's A6000 fleet is ~6x larger and our
# preemption cost is <= 1 sentence per GPU stream.
SCAV_LANGPAR=(--partition=scavenger --account=scavenger --qos=scavenger --gres=gpu:rtxa6000:"$NGPU")
CLIP_LANGPAR=(--partition=clip --account=clip --qos=huge-long --gres=gpu:rtxa6000:"$NGPU")
CLIP_1GPU_SHORT=(--partition=clip --account=clip --qos=default --gres=gpu:rtxa6000:1)
CLIP_2GPU=(--partition=clip --account=clip --qos=huge-long --gres=gpu:rtxa6000:2)
SCAV_1GPU=(--partition=scavenger --account=scavenger --qos=scavenger --gres=gpu:rtxa6000:1)
# Hopper GRES names on Nexus are h100-nvl / h100-sxm / h200-sxm — never plain
# "h100" — so we select by node FEATURE (Hopper) with an untyped gpu count.
# Candidates: cml31 (h100-nvl 94GB), cml33 (4x h100-sxm 80GB), cml35/36 +
# vulcan46 (8x h200-sxm 141GB each).
SCAV_HOPPER=(--partition=scavenger --account=scavenger --qos=scavenger --constraint=Hopper --gres=gpu:1)

is_big() { case "$1" in magistral_small|qwen3_32b) return 0 ;; *) return 1 ;; esac; }

# WMT24++ needs the shared English embeddings first.
DEP_FLAG=()
if [ "$DATASET" = "wmt24pp" ] && [ "${WMT_EMB_READY:-0}" != "1" ]; then
  EMB_JOB=$(sbatch --parsable slurm/wmt24pp_embeddings.sbatch)
  echo "wmt24pp embeddings: $EMB_JOB"
  DEP_FLAG=(--dependency=afterok:"$EMB_JOB")
fi

N=0
for m in $MODELS; do
  for st in $STATES; do
    EXPORTS=""
    if [ "$st" = "on" ] && ! is_big "$m"; then
      # small/medium ON: one language-parallel multi-GPU job per model.
      # Default scavenger (see routing note above); CLIP_ONLY forces clip,
      # where only clip13 can host NGPU=5 — expect queueing.
      if [ "$CLIP_ONLY" = "1" ]; then FL=("${CLIP_LANGPAR[@]}"); else FL=("${SCAV_LANGPAR[@]}"); fi
      jid=$(sbatch --parsable "${FL[@]}" "${DEP_FLAG[@]}" \
            --job-name="rt-${m}-on-${DATASET}" \
            slurm/generate_langpar.sbatch "$m" "$METHODS" "$DATASET")
      echo "$jid  $m on $DATASET  [langpar x$NGPU, methods=$METHODS]  -> ${FL[*]}"
    else
      # big ON + every OFF: one single-allocation job per model
      if [ "$st" = "on" ]; then
        if [ "$CLIP_ONLY" = "1" ]; then FL=("${CLIP_2GPU[@]}"); else FL=("${SCAV_HOPPER[@]}"); fi
      else
        if is_big "$m"; then
          if [ "$CLIP_ONLY" = "1" ]; then FL=("${CLIP_2GPU[@]}"); else FL=("${SCAV_HOPPER[@]}"); fi
        else
          if [ "$SCAVENGER" = "1" ]; then FL=("${SCAV_1GPU[@]}"); else FL=("${CLIP_1GPU_SHORT[@]}"); fi
        fi
      fi
      # 32B on a single 80GB Hopper: cap context so the KV allocation fits.
      if [ "$m" = "qwen3_32b" ] && [[ " ${FL[*]} " == *" --constraint=Hopper "* ]]; then
        if [ "$st" = "on" ]; then
          EXPORTS=",RTRACE_MAX_MODEL_LEN=32768,RTRACE_MAX_NEW_TOKENS=30000"
        else
          EXPORTS=",RTRACE_MAX_MODEL_LEN=8192"
        fi
      fi
      jid=$(sbatch --parsable "${FL[@]}" "${DEP_FLAG[@]}" \
            --export=ALL"$EXPORTS" \
            --job-name="rt-${m}-${st}-${DATASET}" \
            slurm/generate.sbatch "$m" "$METHODS" "$st" "$DATASET")
      echo "$jid  $m $st $DATASET  [methods=$METHODS]  -> ${FL[*]}"
    fi
    N=$((N + 1))
  done
done
echo "submitted $N jobs for dataset=$DATASET (monitor: squeue -u \$USER)"
