#!/bin/bash

# Choose parallel environment
#$ -pe mpi 8

# Specify the job name in the queue system
#$ -N fairmast-consolidate

# Start the script in the current working directory
#$ -cwd

# Time requirements
#$ -l h_rt=48:00:00
#$ -l s_rt=48:00:00

source .venv/bin/activate

num_workers=8
bucket_path="s3://mast/level1/shots/"
local_path="/common/tmp/sjackson/fair-mast/consolidate"

python3 -m scripts.consolidate_s3 $bucket_path $local_path -n $num_workers --start-shot 11695 --end-shot 14830