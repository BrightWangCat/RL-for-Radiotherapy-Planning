#!/bin/bash
#SBATCH --account=PCON0023
#SBATCH --partition=nextgen
#SBATCH --job-name=okbp_step1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/slurm_step1_%j.out
#SBATCH --error=logs/slurm_step1_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310

# 你如果已有环境，把这行改成你自己的环境名
source activate rlfplan



DATA_ROOT="/fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans"

python -m preprocess.state_frames_openkbp.step1_inspect_case \
  --root "${DATA_ROOT}" \
  --outdir "preprocess_outputs/step1_inspect" \
  --load-dij
