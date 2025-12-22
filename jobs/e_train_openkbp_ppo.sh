#!/bin/bash
#SBATCH --account=PCON0023
#SBATCH --partition=nextgen
#SBATCH --job-name=kbp_ppo_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /fs/scratch/PCON0023/mingshiw/RLfPlan5
mkdir -p logs

module load miniconda3/24.1.2-py310
source activate rlfplan
export PYTHONNOUSERSITE=True

# Unbuffer Python stdout/stderr so logs update promptly under Slurm
export PYTHONUNBUFFERED=1

# Ensure imports from project root
export PYTHONPATH="/fs/scratch/PCON0023/mingshiw/RLfPlan5:${PYTHONPATH:-}"

# Avoid CPU oversubscription surprises
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# OpenKBP env config (Run A: PTV-only)
export OPENKBP_ROOT="/fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans"
export OPENKBP_CASE="pt_241"
export OPENKBP_K="64"
export OPENKBP_MAX_STEPS="50"
export OPENKBP_STEP_SCALE="0.05"
export OPENKBP_OAR_LAMBDA="0.02"
export OPENKBP_SEED="0"

python scripts/e_train_openkbp_ppo.py \
  --env-id OpenKBPGrouped-v0 \
  --total-timesteps 500000 \
  --learning-rate 3e-4 \
  --num-envs 4 \
  --cuda
