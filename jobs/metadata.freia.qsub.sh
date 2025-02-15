#!/bin/bash

# Choose parallel environment
#$ -pe mpi 16

# Specify the job name in the queue system
#$ -N fairmast-read-metadata

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

uda_path=/common/tmp/sjackson/data/uda
s3_path=/common/tmp/sjackson/data/s3
index_path=./data/index

# Parse Signal and Source metadata from UDA
mpirun -n 16 python3 -m src.create_uda_metadata $uda_path campaign_shots/mastu.csv 
mpirun -n 16 python3 -m src.create_uda_metadata $uda_path campaign_shots/M9.csv 
mpirun -n 16 python3 -m src.create_uda_metadata $uda_path campaign_shots/M8.csv 
mpirun -n 16 python3 -m src.create_uda_metadata $uda_path campaign_shots/M7.csv 
mpirun -n 16 python3 -m src.create_uda_metadata $uda_path campaign_shots/M6.csv 
mpirun -n 16 python3 -m src.create_uda_metadata $uda_path campaign_shots/M5.csv 

# Parse the metadata from S3 items
mpirun -n 16 python3 -m src.read_source_metadata $uda_path/sources s3://mast/level1/shots $s3_path/sources
mpirun -n 16 python3 -m src.read_signal_metadata $uda_path/sources s3://mast/level1/shots $s3_path/signals

# Group shot files
python3 -m src.group_parquet $s3_path/sources/ $index_path/sources.parquet
python3 -m src.group_parquet $s3_path/signals/ $index_path/signals.parquet
