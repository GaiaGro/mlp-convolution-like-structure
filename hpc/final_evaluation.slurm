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

cd "$SLURM_SUBMIT_DIR"

module purge
module load /software/modules/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mlp-conv

python -m scripts.final_evaluation     --training-directory results/final/training     --output-directory results/final/evaluation     --batch-size 256     --num-workers 4
