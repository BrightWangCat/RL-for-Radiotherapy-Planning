#!/bin/bash
#SBATCH --job-name=openkbp_step1
#SBATCH --account=PCON0023
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=step1_log.out

# 加载你的环境，假设你用的是conda
module load miniconda3/24.1.2-py310
conda activate rlfplan

echo "Job started at $(date)"

# 运行 Python 脚本
python step1_check_and_normalize.py

echo "Job finished at $(date)"