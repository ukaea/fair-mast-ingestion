#!/bin/bash
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p icelake
#SBATCH --job-name=fair-mast-ingest
#SBATCH --output=%A_%a.out
#SBATCH --time=0:20:00
#SBATCH --mem=60G
#SBATCH --ntasks=128
#SBATCH -N 2

num_workers=$SLURM_NTASKS

uda_path="/rds/project/rds-mOlK9qn0PlQ/fairmast/data/uda"
source /rds/project/rds-mOlK9qn0PlQ/fairmast/uda-ssl.sh

# Parse Signal and Source metadata from UDA
mpirun -n $num_workers python3 -m src.create_uda_metadata $uda_path campaign_shots/M9.csv 
mpirun -n $num_workers python3 -m src.create_uda_metadata $uda_path campaign_shots/M8.csv 
mpirun -n $num_workers python3 -m src.create_uda_metadata $uda_path campaign_shots/M7.csv 
mpirun -n $num_workers python3 -m src.create_uda_metadata $uda_path campaign_shots/M6.csv 
mpirun -n $num_workers python3 -m src.create_uda_metadata $uda_path campaign_shots/M5.csv 

