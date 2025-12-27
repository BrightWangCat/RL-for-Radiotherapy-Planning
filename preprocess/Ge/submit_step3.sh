#!/bin/bash
#SBATCH --job-name=openkbp_step3
#SBATCH --account=PCON0023
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=step3_log.out

module load miniconda3/24.1.2-py310
conda activate rlfplan
echo "Job started at $(date)"
python step3_state_frame_2.py
echo "Job finished at $(date)"