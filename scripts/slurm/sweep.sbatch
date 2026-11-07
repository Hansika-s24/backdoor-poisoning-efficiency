#!/bin/bash
#SBATCH --job-name=bd-sweep
#SBATCH --partition=ClsGenAI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-2
# Poison-count sweep. One array task = one training seed; it runs every poison count in
# sequence. --skip-existing makes a requeued/timed-out task resume where it stopped.
#
# Examples (DO NOT submit before reviewing the cost estimate in README.md):
#   sbatch scripts/slurm/sweep.sbatch                                   # LoRA r=8, seeds 0-2
#   EXTRA="arm.rank=16" sbatch scripts/slurm/sweep.sbatch               # rank ablation
#   EXTRA="arm.type=full train.lr=2e-5 save_adapter=false" sbatch scripts/slurm/sweep.sbatch
#   NS="0 1" sbatch --array=0 scripts/slurm/sweep.sbatch                # quick timing probe
source scripts/slurm/common.sh

CONFIG=${CONFIG:-configs/base.yaml}
NS=${NS:-"0 1 2 5 10 20 50 100 200"}
SEED=${SLURM_ARRAY_TASK_ID:-0}
SETS=()
for kv in ${EXTRA:-}; do SETS+=(--set "$kv"); done

for N in $NS; do
  echo "=== n_poison=$N seed=$SEED extra='${EXTRA:-}' ==="
  python scripts/run_experiment.py --config "$CONFIG" --skip-existing \
      --set seed="$SEED" --set attack.n_poison="$N" ${SETS[@]+"${SETS[@]}"}
done
