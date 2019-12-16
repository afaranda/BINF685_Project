#!/bin/bash
#SBATCH --job-name=BLASTrun
#SBATCH --ntasks=8
#SBATCH --mem=16000


# Run HW2 Bayes Net as a slurm job
python3 main.py > result_summary.txt

# Plot Training Results
R CMD BATCH Plot_Train.R




