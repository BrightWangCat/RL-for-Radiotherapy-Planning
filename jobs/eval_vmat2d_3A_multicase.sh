#!/bin/bash
#SBATCH --account=PCON0023
#SBATCH --partition=nextgen
#SBATCH --job-name=vmat2d_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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
export OPENKBP_SEED="0"

# Ensure splits exist (same generator as train)
if [[ ! -f splits/train_cases.txt || ! -f splits/val_cases.txt ]]; then
  echo "[split] splits/*.txt not found; generating from OPENKBP_ROOT=${OPENKBP_ROOT}"
  python - <<'PY'
import os, random, glob
root = os.environ["OPENKBP_ROOT"]
seed = int(os.environ.get("OPENKBP_SEED", "0"))
random.seed(seed)

dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
cases = [d for d in dirs if d.startswith("pt_") or d.startswith("PT_")]

if not cases:
    files = glob.glob(os.path.join(root, "*.npz")) + glob.glob(os.path.join(root, "*.npy"))
    cases = [os.path.splitext(os.path.basename(fp))[0] for fp in files]

cases = sorted(set(cases))
if len(cases) < 2:
    raise RuntimeError(f"Found too few cases under {root}: {cases[:10]}")
random.shuffle(cases)

n = len(cases)
val_n = max(1, int(round(0.2 * n)))
val_n = min(val_n, n - 1)

val = cases[:val_n]
train = cases[val_n:]

os.makedirs("splits", exist_ok=True)
open("splits/train_cases.txt", "w", encoding="utf-8").write("\n".join(train) + "\n")
open("splits/val_cases.txt", "w", encoding="utf-8").write("\n".join(val) + "\n")
print(f"[split] total={n} train={len(train)} val={len(val)}")
PY
fi

# OPENKBP_CASE needed at env construction
export OPENKBP_CASE="$(head -n 1 splits/val_cases.txt)"

RUN_DIR="${1:?Usage: sbatch eval_vmat2d_3A_multicase.sh <RUN_DIR>}"

echo "=== deterministic (val set) ==="
python scripts/f_eval_saved_policy_vmat2d.py \
  --run-dir "${RUN_DIR}" \
  --env-id OpenKBPVMAT2D-v0 \
  --cases-file splits/val_cases.txt \
  --episodes-per-case 1 \
  --max-steps 192

echo "=== stochastic (val set) ==="
python scripts/f_eval_saved_policy_vmat2d.py \
  --run-dir "${RUN_DIR}" \
  --env-id OpenKBPVMAT2D-v0 \
  --cases-file splits/val_cases.txt \
  --episodes-per-case 1 \
  --max-steps 192 \
  --stochastic

echo "=== random (val set) ==="
python scripts/f_eval_saved_policy_vmat2d.py \
  --run-dir "${RUN_DIR}" \
  --env-id OpenKBPVMAT2D-v0 \
  --cases-file splits/val_cases.txt \
  --episodes-per-case 1 \
  --max-steps 192 \
  --random
