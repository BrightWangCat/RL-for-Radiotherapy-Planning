#!/bin/bash
#SBATCH --account=PCON0023
#SBATCH --partition=nextgen
#SBATCH --job-name=vmat2d_3A
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
export PYTHONUNBUFFERED=1
export PYTHONPATH="/fs/scratch/PCON0023/mingshiw/RLfPlan5:${PYTHONPATH:-}"

# ===== OpenKBP / VMAT2D env config =====
export OPENKBP_ROOT="/fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans"

# IMPORTANT: some envs require OPENKBP_CASE at init-time.
# Set it to the first line of your train_cases.txt
export OPENKBP_CASE="$(head -n 1 splits/train_cases.txt)"

# Paper max CP upper bound is 192 (2 arcs)【Hrinivich&Lee 2020】
export OPENKBP_MAX_STEPS="192"

export OPENKBP_OAR_LAMBDA="0.02"
export OPENKBP_SEED="0"

# Optional init knobs (if supported by your env_openkbp_vmat2d.py)
export OPENKBP_INIT_D0="100"
export OPENKBP_INIT_X1_MM="120"
export OPENKBP_INIT_X2_MM="200"
export OPENKBP_CALIBRATE_INIT="1"

# ===== Train =====
python scripts/e_train_openkbp_ppo_vmat2d.py \
  --env-id OpenKBPVMAT2D-v0 \
  --total-timesteps 600000 \
  --learning-rate 3e-4 \
  --num-envs 8 \
  --num-steps 128 \
  --update-epochs 4 \
  --num-minibatches 4 \
  --ent-coef 0.01 \
  --vf-coef 0.5 \
  --clip-coef 0.2 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --cuda \
  --save-model \
  --train-cases-file splits/train_cases.txt \
  --val-cases-file splits/val_cases.txt \
  --case-sample-mode random \
  --eval-every-updates 1 \
  --eval-episodes-per-case 1
