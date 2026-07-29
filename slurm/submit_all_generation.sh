#!/bin/bash
# Dispatch the FULL generation grid with per-language parallelism on the
# reasoning-ON side:
#
#   ON, 8B/14B   -> one LANGUAGE-PARALLEL job per (model x method): N GPUs on
#                   one node, one single-GPU vLLM server per GPU, languages
#                   round-robined across servers (generate_langpar.sbatch).
#   ON, 24B/32B  -> one job per (model x method x LANGUAGE), 1x Hopper each
#                   (weights >48GB; in-job language-parallel would need 2 GPUs
#                   per language on one node — unschedulable).
#   OFF, all     -> unchanged: one job per (model x method), short.
#
#   bash slurm/submit_all_generation.sh                 # FLORES grid
#   bash slurm/submit_all_generation.sh wmt24pp         # WMT24++ (chains embeddings)
#   NGPU=5 bash slurm/submit_all_generation.sh wmt24pp  # 5 GPUs per langpar job
#   SCAVENGER=1 ...                                     # small models to scavenger too
#   CLIP_ONLY=1 ...                                     # big models to clip 2xA6000 TP=2
#   MODELS="qwen3_32b" METHODS="rrf" STATES="on" ...    # subset of the grid
set -euo pipefail

DATASET="${1:-flores}"
MODELS="${MODELS:-qwen3_8b ministral_8b qwen3_14b ministral_14b magistral_small qwen3_32b}"
METHODS="${METHODS:-rrf random sentinel edit_dist}"
STATES="${STATES:-on off}"
SCAVENGER="${SCAVENGER:-0}"
CLIP_ONLY="${CLIP_ONLY:-0}"
# NGPU: GPUs per language-parallel job. 4 is safe for 4-GPU nodes; set 5 once
# show_nodes confirms clip has >=5 A6000s on a single node.
NGPU="${NGPU:-4}"

if [ "$DATASET" = "wmt24pp" ]; then
  GRID_LANGS="cat_Latn zul_Latn mal_Mlym slk_Latn isl_Latn"
else
  GRID_LANGS="wol_Latn swh_Latn lus_Latn mni_Beng tel_Telu tam_Taml uzn_Latn"
fi

CLIP_LANGPAR=(--partition=clip --account=clip --qos=huge-long --gres=gpu:rtxa6000:"$NGPU")
SCAV_LANGPAR=(--partition=scavenger --account=scavenger --qos=scavenger --gres=gpu:rtxa6000:"$NGPU")
CLIP_1GPU_SHORT=(--partition=clip --account=clip --qos=default --gres=gpu:rtxa6000:1)
CLIP_2GPU=(--partition=clip --account=clip --qos=huge-long --gres=gpu:rtxa6000:2)
SCAV_1GPU=(--partition=scavenger --account=scavenger --qos=scavenger --gres=gpu:rtxa6000:1)
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
  for meth in $METHODS; do
    for st in $STATES; do
      case " $STATES " in *" $st "*) ;; esac
      if [ "$st" = "on" ] && ! is_big "$m"; then
        # small/medium ON: one language-parallel multi-GPU job
        if [ "$SCAVENGER" = "1" ]; then FL=("${SCAV_LANGPAR[@]}"); else FL=("${CLIP_LANGPAR[@]}"); fi
        jid=$(sbatch --parsable "${FL[@]}" "${DEP_FLAG[@]}" \
              --job-name="rt-${m}-${meth}-on-${DATASET}" \
              slurm/generate_langpar.sbatch "$m" "$meth" "$DATASET")
        echo "$jid  $m $meth on $DATASET  [langpar x$NGPU]  -> ${FL[*]}"
        N=$((N + 1))
      elif [ "$st" = "on" ]; then
        # big ON: one single-GPU job per language
        for lang in $GRID_LANGS; do
          if [ "$CLIP_ONLY" = "1" ]; then FL=("${CLIP_2GPU[@]}"); else FL=("${SCAV_HOPPER[@]}"); fi
          EXPORTS=",RTRACE_TGT_LANGS=$lang"
          if [ "$m" = "qwen3_32b" ] && [ "$CLIP_ONLY" != "1" ]; then
            EXPORTS="$EXPORTS,RTRACE_MAX_MODEL_LEN=32768,RTRACE_MAX_NEW_TOKENS=30000"
          fi
          jid=$(sbatch --parsable "${FL[@]}" "${DEP_FLAG[@]}" \
                --export=ALL"$EXPORTS" \
                --job-name="rt-${m}-${meth}-on-${lang%%_*}-${DATASET}" \
                slurm/generate.sbatch "$m" "$meth" "on" "$DATASET")
          echo "$jid  $m $meth on $lang $DATASET  -> ${FL[*]}"
          N=$((N + 1))
        done
      else
        # OFF: unchanged — one short job per (model, method)
        if [ "$SCAVENGER" = "1" ]; then
          if is_big "$m"; then FL=("${SCAV_HOPPER[@]}"); else FL=("${SCAV_1GPU[@]}"); fi
        else
          if is_big "$m"; then FL=("${SCAV_HOPPER[@]}"); else FL=("${CLIP_1GPU_SHORT[@]}"); fi
          [ "$CLIP_ONLY" = "1" ] && is_big "$m" && FL=("${CLIP_2GPU[@]}")
        fi
        EXPORTS=""
        if [ "$m" = "qwen3_32b" ] && [[ " ${FL[*]} " == *" --constraint=Hopper "* ]]; then
          EXPORTS=",RTRACE_MAX_MODEL_LEN=8192"
        fi
        jid=$(sbatch --parsable "${FL[@]}" "${DEP_FLAG[@]}" \
              --export=ALL"$EXPORTS" \
              --job-name="rt-${m}-${meth}-off-${DATASET}" \
              slurm/generate.sbatch "$m" "$meth" "off" "$DATASET")
        echo "$jid  $m $meth off $DATASET  -> ${FL[*]}"
        N=$((N + 1))
      fi
    done
  done
done
echo "submitted $N jobs for dataset=$DATASET (monitor: squeue -u \$USER)"
