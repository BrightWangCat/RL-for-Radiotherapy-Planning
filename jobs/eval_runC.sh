#!/bin/bash
#SBATCH --account=PCON0023
#SBATCH --partition=nextgen
#SBATCH --job-name=kbp_eval_runC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
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

# Match Run C config
export OPENKBP_ROOT="/fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans"
export OPENKBP_CASE="pt_241"
export OPENKBP_K="64"
export OPENKBP_MAX_STEPS="50"
export OPENKBP_STEP_SCALE="0.05"
export OPENKBP_OAR_LAMBDA="0.02"
export OPENKBP_SEED="0"

RUN_DIR="runs/OpenKBPGrouped-v0__ppo_continuous_action__1__1766529930"

echo "=== deterministic ==="
python scripts/f_eval_saved_policy.py --run-dir "${RUN_DIR}" --episodes 20

echo "=== stochastic ==="
python scripts/f_eval_saved_policy.py --run-dir "${RUN_DIR}" --episodes 20 --stochastic

echo "=== random ==="
python scripts/f_eval_saved_policy.py --run-dir "${RUN_DIR}" --episodes 20 --random
