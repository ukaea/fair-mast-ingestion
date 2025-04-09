#!/bin/bash
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p ukaea-icl
#SBATCH --job-name=fair-mast-ingest-level2
#SBATCH --output=fair-mast-ingest_%A.out
#SBATCH --time=36:00:00
#SBATCH --mem=250G
#SBATCH --ntasks=64
#SBATCH -N 8


num_workers=$SLURM_NTASKS
source .venv/bin/activate
source /rds/project/rds-mOlK9qn0PlQ/fairmast/uda-ssl.sh

mpirun -n $num_workers \
    python3 -m src.level2.main mappings/level2/mast.yml --shot-min 11695 --shot-max 30474 -e camera_visible --force

