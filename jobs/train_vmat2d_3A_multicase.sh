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
mkdir -p logs splits

module load miniconda3/24.1.2-py310
source activate rlfplan

export PYTHONNOUSERSITE=True
export PYTHONUNBUFFERED=1
export PYTHONPATH="/fs/scratch/PCON0023/mingshiw/RLfPlan5:${PYTHONPATH:-}"

# ===== OpenKBP / VMAT2D env config =====
export OPENKBP_ROOT="/fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans"
export OPENKBP_MAX_STEPS="192"
export OPENKBP_OAR_LAMBDA="0.02"
export OPENKBP_SEED="0"

export OPENKBP_ACTION_LAMBDA="0.02"
# Optional init knobs
export OPENKBP_INIT_D0="100"
export OPENKBP_INIT_LEAF_HALF_WIDTH="8" 
export OPENKBP_CALIBRATE_INIT="1"

# ===== Auto-generate case splits if missing =====
if [[ ! -f splits/train_cases.txt || ! -f splits/val_cases.txt ]]; then
  echo "[split] splits/*.txt not found; generating from OPENKBP_ROOT=${OPENKBP_ROOT}"
  python - <<'PY'
import os, random, glob

root = os.environ["OPENKBP_ROOT"]
seed = int(os.environ.get("OPENKBP_SEED", "0"))
random.seed(seed)

# 1) Prefer directories like pt_241/
dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
cases = [d for d in dirs if d.startswith("pt_") or d.startswith("PT_")]

# 2) Fallback: infer from files if no pt_* dirs
if not cases:
    files = glob.glob(os.path.join(root, "*.npz")) + glob.glob(os.path.join(root, "*.npy"))
    for fp in files:
        base = os.path.basename(fp)
        stem = os.path.splitext(base)[0]
        # keep only reasonable ids
        if stem:
            cases.append(stem)

# de-dup + sort then shuffle
cases = sorted(set(cases))
if len(cases) < 2:
    raise RuntimeError(f"Found too few cases under {root}. dirs={len(dirs)} cases={cases[:10]}")

random.shuffle(cases)

n = len(cases)
# 20% val with bounds
val_n = max(1, int(round(0.2 * n)))
# keep at least 1 train
val_n = min(val_n, n - 1)

val = cases[:val_n]
train = cases[val_n:]

os.makedirs("splits", exist_ok=True)
with open("splits/train_cases.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train) + "\n")
with open("splits/val_cases.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(val) + "\n")

print(f"[split] total={n} train={len(train)} val={len(val)}")
print("[split] first_train:", train[0])
print("[split] first_val:", val[0])
PY
fi

# Set OPENKBP_CASE for env construction (some envs require it at gym.make time)
export OPENKBP_CASE="$(head -n 1 splits/train_cases.txt)"
echo "[split] OPENKBP_CASE=${OPENKBP_CASE}"

# ===== Train (3A) =====
python scripts/e_train_openkbp_ppo_vmat2d.py \
  --env-id OpenKBPVMAT2D-v0 \
  --total-timesteps 20000 \
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
  --eval-every-updates 5 \
  --eval-episodes-per-case 1
