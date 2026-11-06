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
    echo "Usage: sbatch hpc/structural_analysis.slurm <43|44|45|46|47>"
    exit 1
fi

SEED="$1"

case "$SEED" in
    43|44|45|46|47) ;;
    *)
        echo "Seed must be one of 43, 44, 45, 46, 47."
        exit 1
        ;;
esac

cd "$SLURM_SUBMIT_DIR"

module purge
module load /software/modules/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mlp-conv

sha256sum -c docs/structural_metrics.sha256
sha256sum -c docs/structural_analysis_runtime.sha256

python -m unittest tests.test_structural_metrics -v

python -m scripts.structural_analysis \
    --seed "$SEED"
