#!/bin/bash
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch hpc/final_training.slurm <43|44|45|46|47>"
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

sha256sum -c docs/protocol.sha256

OUTPUT_ROOT="results/final/training"
EPOCHS=100
MLP_LR=0.0003
MLP_WD=0.0001
CNN_LR=0.001
CNN_WD=0.0001
LAMBDA=7.5
RHO=0.2

run_complete() {
    local RUN_NAME="$1"
    [[ -s "$OUTPUT_ROOT/checkpoints/${RUN_NAME}_best.pt"         && -s "$OUTPUT_ROOT/histories/${RUN_NAME}.csv"         && -s "$OUTPUT_ROOT/configs/${RUN_NAME}.yaml" ]]
}

MLP_CONDITIONS=(
    baseline
    exposure
    consistency
    augmentation
    augmentation_consistency
)

for CONDITION in "${MLP_CONDITIONS[@]}"; do
    RUN_NAME="mlp_${CONDITION}_seed${SEED}"

    if run_complete "$RUN_NAME"; then
        echo "Skipping completed run: $RUN_NAME"
        continue
    fi

    COMMAND=(
        python -m scripts.train
        --model mlp
        --condition "$CONDITION"
        --seed "$SEED"
        --run-name "$RUN_NAME"
        --epochs "$EPOCHS"
        --batch-size 256
        --learning-rate "$MLP_LR"
        --weight-decay "$MLP_WD"
        --validation-interval 5
        --num-workers 4
        --output-directory "$OUTPUT_ROOT"
    )

    if [[ "$CONDITION" == "consistency"         || "$CONDITION" == "augmentation_consistency" ]]; then
        COMMAND+=(
            --consistency-weight "$LAMBDA"
            --variance-covariance-weight "$RHO"
        )
    fi

    "${COMMAND[@]}"
done

for CONDITION in baseline augmentation; do
    RUN_NAME="cnn_${CONDITION}_seed${SEED}"

    if run_complete "$RUN_NAME"; then
        echo "Skipping completed run: $RUN_NAME"
        continue
    fi

    python -m scripts.train         --model cnn         --condition "$CONDITION"         --seed "$SEED"         --run-name "$RUN_NAME"         --epochs "$EPOCHS"         --batch-size 256         --learning-rate "$CNN_LR"         --weight-decay "$CNN_WD"         --validation-interval 5         --num-workers 4         --output-directory "$OUTPUT_ROOT"
done
