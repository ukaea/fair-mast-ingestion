#!/bin/bash

# Choose parallel environment
#$ -pe mpi 1

# Specify the job name in the queue system
#$ -N fair-mast-writer

# Start the script in the current working directory
#$ -cwd

# Time requirements
#$ -l h_rt=48:00:00
#$ -l s_rt=48:00:00

num_workers=16
source .venv/bin/activate
# python3 -m src.level2.main mappings/level2/mastu.yml -n $num_workers --shot-min 41139 --shot-max 51056 -e camera_visible
python3 -m src.level2.main mappings/level2/mast.yml  -n $num_workers --shot-min 11695 --shot-max 30474 -e camera_visible