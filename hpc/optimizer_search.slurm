#!/bin/bash

#SBATCH --job-name=optimizer
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=out/%x_%A.out
#SBATCH --error=err/%x_%A.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 <mlp|cnn>"
    exit 1
fi

MODEL="$1"

if [[ "$MODEL" != "mlp" && "$MODEL" != "cnn" ]]; then
    echo "Model must be mlp or cnn."
    exit 1
fi

cd "$SLURM_SUBMIT_DIR"

module purge
module load /software/modules/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mlp-conv

LEARNING_RATES=(0.0003 0.001)
WEIGHT_DECAYS=(0.0001 0.0005)
SEEDS=(11 22)

for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
    for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            python -m scripts.train \
                --model "$MODEL" \
                --condition baseline \
                --seed "$SEED" \
                --run-name "${MODEL}_optimizer_lr${LEARNING_RATE//./p}_wd${WEIGHT_DECAY//./p}_seed${SEED}" \
                --epochs 100 \
                --batch-size 256 \
                --learning-rate "$LEARNING_RATE" \
                --weight-decay "$WEIGHT_DECAY" \
                --validation-interval 5 \
                --num-workers 4 \
                --output-directory results/development/optimizer
        done
    done
done
