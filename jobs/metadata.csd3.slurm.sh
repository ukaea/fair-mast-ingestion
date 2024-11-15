#!/bin/bash
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p icelake
#SBATCH --job-name=fair-mast-metadata
#SBATCH --output=fair-mast-metadata-%A.out
#SBATCH --time=1:00:00
#SBATCH --mem=60G
#SBATCH --ntasks=64
#SBATCH -N 1

num_workers=$SLURM_NTASKS

uda_path="/rds/project/rds-mOlK9qn0PlQ/fairmast/data/uda"
s3_path="/rds/project/rds-mOlK9qn0PlQ/fairmast/data/s3"
index_path=./data/index
mkdir -p $index_path
source /rds/project/rds-mOlK9qn0PlQ/fairmast/uda-ssl.sh

# Parse Signal and Source metadata from UDA
mpirun -n $num_workers python3 -m src.create_uda_metadata $uda_path campaign_shots/M9.csv 
mpirun -n $num_workers python3 -m src.create_uda_metadata $uda_path campaign_shots/M8.csv 
mpirun -n $num_workers python3 -m src.create_uda_metadata $uda_path campaign_shots/M7.csv 
mpirun -n $num_workers python3 -m src.create_uda_metadata $uda_path campaign_shots/M6.csv 
mpirun -n $num_workers python3 -m src.create_uda_metadata $uda_path campaign_shots/M5.csv 

mpirun -n $num_workers python3 -m src.read_source_metadata $uda_path/sources s3://mast/level1/shots $s3_path/sources
mpirun -n $num_workers python3 -m src.read_signal_metadata $uda_path/sources s3://mast/level1/shots $s3_path/signals

python3 -m src.group_parquet $s3_path/sources/ $index_path/sources.parquet
python3 -m src.group_parquet $s3_path/signals/ $index_path/signals.parquet
