#!/bin/bash
#SBATCH --account=PCON0023
#SBATCH --partition=nextgen
#SBATCH --job-name=kbp_vmat2d_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
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

# --- Must match training env vars ---
export OPENKBP_ROOT="/fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans"
export OPENKBP_CASE="pt_241"

export OPENKBP_K="64"
export OPENKBP_N_CPS="96"
export OPENKBP_MAX_STEPS="192"
export OPENKBP_OAR_LAMBDA="0.02"
export OPENKBP_SEED="0"

export OPENKBP_D0_MIN="20"
export OPENKBP_D0_MAX="600"
export OPENKBP_INIT_D0="100"
export OPENKBP_INIT_LEAF_HALF_WIDTH="8"

# 改成你实际训练产生的 runs 目录
RUN_DIR="runs/OpenKBPVMAT2D-v0__ppo_discrete_cnn__0__1766551890"

echo "=== deterministic ==="
python scripts/f_eval_saved_policy_vmat2d.py --run-dir "${RUN_DIR}" --env-id OpenKBPVMAT2D-v0 --episodes 20

echo "=== stochastic ==="
python scripts/f_eval_saved_policy_vmat2d.py --run-dir "${RUN_DIR}" --env-id OpenKBPVMAT2D-v0 --episodes 20 --stochastic

echo "=== random ==="
python scripts/f_eval_saved_policy_vmat2d.py --run-dir "${RUN_DIR}" --env-id OpenKBPVMAT2D-v0 --episodes 20 --random
