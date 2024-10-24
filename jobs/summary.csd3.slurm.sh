#!/bin/bash
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p icelake
#SBATCH --job-name=fair-mast-summary
#SBATCH --output=fair-mast-summary-%A.out
#SBATCH --time=1:00:00
#SBATCH --mem=60G
#SBATCH --ntasks=64
#SBATCH -N 1

num_workers=$SLURM_NTASKS

source /rds/project/rds-sPGbyCAPsJI/uda-ssl.sh

mpirun -np $num_workers python -m src.summary