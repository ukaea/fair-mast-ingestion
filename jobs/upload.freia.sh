#!/bin/bash

# Choose parallel environment
#$ -pe mpi 1

# Specify the job name in the queue system
#$ -N fair-mast-upload

# Start the script in the current working directory
#$ -cwd

# Time requirements
#$ -l h_rt=48:00:00
#$ -l s_rt=48:00:00

source .venv/bin/activate

credential=.s5cfg.ukaea
endpoint=http://mon3.cepheus.hpc.l:8000
local_path=/common/tmp/sjackson/upload-tmp/zarr/level2/
remote_path=s3://fairmast/mastu/level2/shots/

s5cmd --credentials-file $credential --endpoint-url $endpoint cp --acl public-read $local_path $remote_path