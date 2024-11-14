#!/bin/bash
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p icelake
#SBATCH --job-name=fair-mast-ingest
#SBATCH --output=fair-mast-ingest_%A.out
#SBATCH --time=5:00:00
#SBATCH --mem=250G
#SBATCH --ntasks=128
#SBATCH -N 2


bucket_path="s3://mast/test/"
num_workers=$SLURM_NTASKS

random_string=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 16)
temp_dir="/rds/project/rds-sPGbyCAPsJI/local_cache/$random_string"
metadata_dir="/rds/project/rds-sPGbyCAPsJI/data/uda"
endpoint_url="https://object.arcus.openstack.hpc.cam.ac.uk"
credentials_file=".s5cfg.csd3"

source ./fair-mast-ingestion/bin/activate
source /rds/project/rds-sPGbyCAPsJI/uda-ssl.sh

summary_file="./campaign_shots/M9.csv"
time mpirun -np $num_workers \
    python3 -m src.main $temp_dir $summary_file --metadata_dir $metadata_dir --bucket_path $bucket_path --endpoint_url $endpoint_url --credentials_file $credentials_file --upload --force --source_names ${@:1}
