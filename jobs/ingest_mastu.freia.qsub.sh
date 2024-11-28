#!/bin/bash

# Choose parallel environment
#$ -pe mpi 16

# Specify the job name in the queue system
#$ -N fairmast-dataset-writer

# Start the script in the current working directory
#$ -cwd

# Time requirements
#$ -l h_rt=48:00:00
#$ -l s_rt=48:00:00

# Activate your environment here!
module load python/3.9
module load uda/2.6.1
module load uda-mast/1.3.9
source /home/rt2549/envs/fmast/bin/activate

# Get command line arguments
bucket_path="s3://fairmast/mastu/level1/shots"
num_workers=8

export PATH="/home/rt2549/dev/:$PATH"

random_string=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 16)
temp_dir="/common/tmp/sjackson/local_cache/$random_string"
metadata_dir="/common/tmp/sjackson/data/uda/"
credentials_file=".s5cfg.ukaea"
endpoint_url="http://mon3.cepheus.hpc.l:8000"

# Run script
summary_file="./campaign_shots/mastu.csv"
mpirun -np $num_workers \
    python3 -m src.main $temp_dir $summary_file --credentials_file $credentials_file --endpoint_url $endpoint_url --metadata_dir $metadata_dir --bucket_path $bucket_path --upload --force --facility "MASTU" --source_names ${@:1}
