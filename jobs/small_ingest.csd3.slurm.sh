#!/bin/bash
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p icelake
#SBATCH --job-name=fair-mast-small-ingest
#SBATCH --output=fair-mast-ingest_%A.out
#SBATCH --time=5:00:00
#SBATCH --mem=250G
#SBATCH --ntasks=128
#SBATCH -N 2

bucket_path=$1
num_workers=$SLURM_NTASKS

random_string=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 16)
temp_dir="/rds/project/rds-mOlK9qn0PlQ/fairmast/local_cache/$random_string"
metadata_dir="/rds/project/rds-mOlK9qn0PlQ/fairmast/data/uda"

source /rds/project/rds-mOlK9qn0PlQ/fairmast/uda-ssl.sh

summary_file=$2
mpirun -np $num_workers \
    python3 -m src.main $temp_dir $summary_file --metadata_dir $metadata_dir --bucket_path $bucket_path --upload --force --source_names ${@:3}