#!/bin/bash
#SBATCH --job-name=Bayes
#SBATCH --ntasks=8
#SBATCH --mem=64000


# Run HW2 Bayes Net as a slurm job
export resdir=results2
export fn=$(echo $1 | sed 's/_Analysis\.py/_out.txt/')
echo "python3 $1 > ${resdir}/${fn}"
python3 $1 > ${resdir}/${fn}

# Plot Training Results
#R CMD BATCH Plot_Train.R




