#!/bin/bash
#SBATCH --job-name=preview_edge
#SBATCH --account=PCON0023
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=step4_preview.out

module load miniconda3/24.1.2-py310
conda activate rlfplan
echo "Job started"
python step4_preview_edge.py
echo "Job finished"