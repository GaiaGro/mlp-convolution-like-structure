#!/bin/bash
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch hpc/epoch_ablation.slurm <11|22>"
    exit 1
fi

SEED="$1"
case "$SEED" in
    11|22) ;;
    *)
        echo "Development seed must be 11 or 22."
        exit 1
        ;;
esac

cd "$SLURM_SUBMIT_DIR"

module purge
module load /software/modules/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mlp-conv

SELECTED="results/development/optimizer/selected_optimizers.yaml"

if [[ ! -f "$SELECTED" ]]; then
    echo "Missing $SELECTED"
    exit 1
fi

VALUES="$(
    python -c '
import sys
from pathlib import Path
import yaml
data = yaml.safe_load(Path(sys.argv[1]).read_text())
print(data["mlp"]["learning_rate"], data["mlp"]["weight_decay"])
' "$SELECTED"
)"

read -r LEARNING_RATE WEIGHT_DECAY <<< "$VALUES"

for EPOCHS in 100 200 300; do
    python -m scripts.train \
        --model mlp \
        --condition baseline \
        --seed "$SEED" \
        --run-name "mlp_epoch${EPOCHS}_seed${SEED}" \
        --epochs "$EPOCHS" \
        --batch-size 256 \
        --learning-rate "$LEARNING_RATE" \
        --weight-decay "$WEIGHT_DECAY" \
        --validation-interval 5 \
        --num-workers 4 \
        --output-directory results/development/epoch_ablation
done
