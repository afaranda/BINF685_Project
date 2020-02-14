#!/bin/bash
#SBATCH --job-name=Bayes
#SBATCH --ntasks=8
#SBATCH --mem=64000


# Run HW2 Bayes Net as a slurm job
export fn=$(echo $1 | sed 's/_Analysis\.py/_out.txt')
python3 $1 > ${fn}

# Plot Training Results
#R CMD BATCH Plot_Train.R




