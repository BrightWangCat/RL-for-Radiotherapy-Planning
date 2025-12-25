#!/bin/bash
#SBATCH --account=PCON0023
#SBATCH --partition=nextgen
#SBATCH --job-name=dump_frames
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
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

export OPENKBP_ROOT="/fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans"
export OPENKBP_MAX_STEPS="192"
export OPENKBP_OAR_LAMBDA="0.02"
export OPENKBP_ACTION_LAMBDA="0.02"
export OPENKBP_SEED="0"

# required for env construction on your setup
export OPENKBP_CASE="$(head -n 1 splits/val_cases.txt)"

# choose a case for presentation
CASE_ID="pt_241"

# dump s0 and s10 with fixed no-op (shows rotation/sequence behavior in your pipeline)
python scripts/g_dump_state_frames.py \
  --env-id OpenKBPVMAT2D-v0 \
  --case-id "${CASE_ID}" \
  --policy fixed \
  --fixed-action 0 \
  --n-steps 10 \
  --outdir "state_frames_${CASE_ID}" \
  --tag "s"
