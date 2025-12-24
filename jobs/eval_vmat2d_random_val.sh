#!/bin/bash
#SBATCH --account=PCON0023
#SBATCH --partition=nextgen
#SBATCH --job-name=vmat2d_rand
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /fs/scratch/PCON0023/mingshiw/RLfPlan5
mkdir -p logs splits

module load miniconda3/24.1.2-py310
source activate rlfplan

export PYTHONNOUSERSITE=True
export PYTHONUNBUFFERED=1
export PYTHONPATH="/fs/scratch/PCON0023/mingshiw/RLfPlan5:${PYTHONPATH:-}"

export OPENKBP_ROOT="/fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans"
export OPENKBP_MAX_STEPS="192"
export OPENKBP_OAR_LAMBDA="0.02"
export OPENKBP_ACTION_LAMBDA="0.02"
export OPENKBP_SEED="0"

# env construction needs a case id
export OPENKBP_CASE="$(head -n 1 splits/val_cases.txt)"

RUN_DIR="runs/OpenKBPVMAT2D-v0__ppo_discrete_cnn__0__1766557937"

echo "=== random (val set) with action penalty ==="
python scripts/f_eval_saved_policy_vmat2d.py \
  --run-dir "${RUN_DIR}" \
  --env-id OpenKBPVMAT2D-v0 \
  --cases-file splits/val_cases.txt \
  --episodes-per-case 1 \
  --max-steps 192 \
  --random
