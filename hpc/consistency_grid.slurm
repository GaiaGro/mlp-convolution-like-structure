#!/bin/bash
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch hpc/consistency_grid.slurm <11|22>"
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

OPTIMIZER_FILE="results/development/optimizer/selected_optimizers.yaml"
EPOCH_FILE="results/development/epoch_ablation/selected_epoch_budget.yaml"
CONFIG_FILE="configs/experiment.yaml"
OUTPUT_ROOT="results/development/consistency_grid"

run_complete() {
    local RUN_NAME="$1"
    [[ -s "$OUTPUT_ROOT/checkpoints/${RUN_NAME}_best.pt" \
        && -s "$OUTPUT_ROOT/histories/${RUN_NAME}.csv" \
        && -s "$OUTPUT_ROOT/configs/${RUN_NAME}.yaml" ]]
}

for REQUIRED in \
    "$OPTIMIZER_FILE" \
    "$EPOCH_FILE" \
    "$CONFIG_FILE"
do
    if [[ ! -f "$REQUIRED" ]]; then
        echo "Missing required file: $REQUIRED"
        exit 1
    fi
done

VALUES="$(
    python -c '
import sys
from pathlib import Path
import yaml

optimizer = yaml.safe_load(
    Path(sys.argv[1]).read_text()
)["mlp"]
epoch = yaml.safe_load(
    Path(sys.argv[2]).read_text()
)

print(
    optimizer["learning_rate"],
    optimizer["weight_decay"],
    epoch["epochs"],
)
' "$OPTIMIZER_FILE" "$EPOCH_FILE"
)"

read -r LEARNING_RATE WEIGHT_DECAY EPOCHS <<< "$VALUES"

mapfile -t LAMBDAS < <(
    python -c '
from pathlib import Path
import yaml
config = yaml.safe_load(
    Path("configs/experiment.yaml").read_text()
)
for value in config["consistency"]["lambda_candidates"]:
    print(value)
'
)

mapfile -t RHOS < <(
    python -c '
from pathlib import Path
import yaml
config = yaml.safe_load(
    Path("configs/experiment.yaml").read_text()
)
for value in config["consistency"]["rho_candidates"]:
    print(value)
'
)

EXPOSURE_RUN="mlp_exposure_seed${SEED}"
if run_complete "$EXPOSURE_RUN"; then
    echo "Skipping completed run: $EXPOSURE_RUN"
else
    python -m scripts.train \
        --model mlp \
        --condition exposure \
        --seed "$SEED" \
        --run-name "$EXPOSURE_RUN" \
        --epochs "$EPOCHS" \
        --batch-size 256 \
        --learning-rate "$LEARNING_RATE" \
        --weight-decay "$WEIGHT_DECAY" \
        --validation-interval 5 \
        --num-workers 4 \
        --output-directory "$OUTPUT_ROOT"
fi

for LAMBDA in "${LAMBDAS[@]}"; do
    for RHO in "${RHOS[@]}"; do
        LAMBDA_LABEL="${LAMBDA//./p}"
        RHO_LABEL="${RHO//./p}"
        RUN_NAME="mlp_consistency_lambda${LAMBDA_LABEL}_rho${RHO_LABEL}_seed${SEED}"
        if run_complete "$RUN_NAME"; then
            echo "Skipping completed run: $RUN_NAME"
            continue
        fi

        python -m scripts.train \
            --model mlp \
            --condition consistency \
            --seed "$SEED" \
            --run-name "$RUN_NAME" \
            --consistency-weight "$LAMBDA" \
            --variance-covariance-weight "$RHO" \
            --epochs "$EPOCHS" \
            --batch-size 256 \
            --learning-rate "$LEARNING_RATE" \
            --weight-decay "$WEIGHT_DECAY" \
            --validation-interval 5 \
            --num-workers 4 \
            --output-directory "$OUTPUT_ROOT"
    done
done
