#!/bin/bash
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p icelake
#SBATCH --job-name=fair-mast-ingest
#SBATCH --output=fair-mast-ingest_%A.out
#SBATCH --time=5:00:00
#SBATCH --mem=256G
#SBATCH --ntasks=128
#SBATCH -N 2


summary_file=$1
bucket_path=$2
num_workers=$SLURM_NTASKS

random_string=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 16)
temp_dir="/rds/project/rds-sPGbyCAPsJI/local_cache/$random_string"
metadata_dir="/rds/project/rds-sPGbyCAPsJI/data/uda"

source /rds/project/rds-sPGbyCAPsJI/uda-ssl.sh

mpirun -np $num_workers \
    python3 -m src.main $temp_dir $summary_file --metadata_dir $metadata_dir --bucket_path $bucket_path --upload --force --source_names ${@:3}
