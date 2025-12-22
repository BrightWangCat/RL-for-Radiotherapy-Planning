#!/bin/bash
#SBATCH --account=PCON0023
#SBATCH --partition=nextgen
#SBATCH --job-name=kbp_case_sanity_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /fs/scratch/PCON0023/mingshiw/RLfPlan5
mkdir -p logs

module load miniconda3/24.1.2-py310
source activate rlfplan
export PYTHONNOUSERSITE=True

python scripts/b_case_sanity_v2.py /fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans pt_241
