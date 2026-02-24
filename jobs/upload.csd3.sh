#!/bin/bash
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p ukaea-icl
#SBATCH --job-name=fair-mast-ingest-level2-upload
#SBATCH --output=fair-mast-ingest-upload_%A.out
#SBATCH --time=36:00:00
#SBATCH --mem=250G
#SBATCH --ntasks=1
#SBATCH -N 1


num_workers=$SLURM_NTASKS
source .venv/bin/activate
source ~/.uda-ssl.sh

credential=.s5cfg.stfc
endpoint=https://s3.echo.stfc.ac.uk
local_path=/rds/project/rds-mOlK9qn0PlQ/fairmast/level2/tmp/
remote_path=s3://mast/level2/shots/

s5cmd --credentials-file $credential --endpoint-url $endpoint cp --acl public-read $local_path $remote_path
s5cmd --credentials-file $credential --endpoint-url $endpoint sync --delete --acl public-read $local_path $remote_path
