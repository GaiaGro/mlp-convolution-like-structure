#!/bin/bash
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch hpc/pixel_permutation.slurm <43|44|45|46|47>"
    exit 1
fi

SEED="$1"
case "$SEED" in
    43|44|45|46|47) ;;
    *)
        echo "Final seed must be 43, 44, 45, 46 or 47."
        exit 1
        ;;
esac

cd "$SLURM_SUBMIT_DIR"

module purge
module load /software/modules/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mlp-conv

sha256sum -c docs/structural_controls.sha256

CONTROL_CONFIG="configs/structural_controls.yaml"
EXPERIMENT_CONFIG="configs/experiment.yaml"
OPTIMIZER_FILE="results/final/selection/selected_optimizers.yaml"
EPOCH_FILE="results/final/selection/selected_epoch_budget.yaml"
CONSISTENCY_FILE="results/final/selection/selected_consistency.yaml"
OUTPUT_ROOT="results/structural/pixel_permutation"

for REQUIRED in \
    "$CONTROL_CONFIG" \
    "$EXPERIMENT_CONFIG" \
    "$OPTIMIZER_FILE" \
    "$EPOCH_FILE" \
    "$CONSISTENCY_FILE"
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

control = yaml.safe_load(Path(sys.argv[1]).read_text())
experiment = yaml.safe_load(Path(sys.argv[2]).read_text())
optimizer = yaml.safe_load(
    Path(sys.argv[3]).read_text()
)["mlp"]
epoch = yaml.safe_load(Path(sys.argv[4]).read_text())
consistency = yaml.safe_load(Path(sys.argv[5]).read_text())

pixel = control["pixel_permutation"]
runtime = control["runtime"]

print(
    pixel["spatial_seed"],
    epoch["epochs"],
    experiment["training"]["batch_size"],
    optimizer["learning_rate"],
    optimizer["weight_decay"],
    runtime["validation_interval"],
    runtime["num_workers"],
    consistency["lambda"],
    consistency["rho"],
)
' "$CONTROL_CONFIG" "$EXPERIMENT_CONFIG" \
    "$OPTIMIZER_FILE" "$EPOCH_FILE" \
    "$CONSISTENCY_FILE"
)"

read -r PERMUTATION_SEED EPOCHS BATCH_SIZE \
    LEARNING_RATE WEIGHT_DECAY VALIDATION_INTERVAL \
    NUM_WORKERS LAMBDA RHO <<< "$VALUES"

mapfile -t CONDITIONS < <(
    python -c '
from pathlib import Path
import yaml

config = yaml.safe_load(
    Path("configs/structural_controls.yaml").read_text()
)

for condition in config["pixel_permutation"]["conditions"]:
    print(condition)
'
)

run_complete() {
    local RUN_NAME="$1"

    [[ -s "$OUTPUT_ROOT/checkpoints/${RUN_NAME}_best.pt" \
        && -s "$OUTPUT_ROOT/histories/${RUN_NAME}.csv" \
        && -s "$OUTPUT_ROOT/configs/${RUN_NAME}.yaml" ]]
}

for CONDITION in "${CONDITIONS[@]}"; do
    RUN_NAME="mlp_pixelperm_${CONDITION}_seed${SEED}"

    if run_complete "$RUN_NAME"; then
        echo "Skipping completed run: $RUN_NAME"
        continue
    fi

    COMMAND=(
        python -m scripts.train
        --model mlp
        --condition "$CONDITION"
        --seed "$SEED"
        --pixel-permutation-seed "$PERMUTATION_SEED"
        --run-name "$RUN_NAME"
        --epochs "$EPOCHS"
        --batch-size "$BATCH_SIZE"
        --learning-rate "$LEARNING_RATE"
        --weight-decay "$WEIGHT_DECAY"
        --validation-interval "$VALIDATION_INTERVAL"
        --num-workers "$NUM_WORKERS"
        --output-directory "$OUTPUT_ROOT"
    )

    if [[ "$CONDITION" == "consistency" \
        || "$CONDITION" == "augmentation_consistency" ]]
    then
        COMMAND+=(
            --consistency-weight "$LAMBDA"
            --variance-covariance-weight "$RHO"
        )
    fi

    "${COMMAND[@]}"
done
