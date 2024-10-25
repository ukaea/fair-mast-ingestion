#!/bin/bash
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p icelake
#SBATCH --job-name=fair-mast-ingest
#SBATCH --output=%A_%a.out
#SBATCH --time=36:00:00
#SBATCH --mem=60G
#SBATCH --ntasks=8
#SBATCH -N 1

num_workers=$SLURM_NTASKS

mpirun -n $num_workers python3 -m src.consolidate_s3