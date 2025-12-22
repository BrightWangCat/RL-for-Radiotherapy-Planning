#!/bin/bash
#SBATCH --account=PCON0023
#SBATCH --partition=nextgen
#SBATCH --job-name=rlf_sanity_ppo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /fs/scratch/PCON0023/mingshiw/RLfPlan5
mkdir -p logs

module load miniconda3/24.1.2-py310
source activate rlfplan
export PYTHONNOUSERSITE=True

echo "===== ENV CHECK ====="
which python
python -V

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "===== RUN CleanRL PPO (short) ====="
cd /fs/scratch/PCON0023/mingshiw/RLfPlan5/cleanrl
python cleanrl/ppo.py --env-id CartPole-v1 --total-timesteps 20000 --cuda
