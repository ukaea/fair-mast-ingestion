#!/bin/bash
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p ukaea-spr
#SBATCH --job-name=fair-mast-metadata-level2
#SBATCH --output=fair-mast-metadata_%A.out
#SBATCH --time=36:00:00
#SBATCH --ntasks=64
#SBATCH -N 1

endpoint_url="https://s3.echo.stfc.ac.uk"
bucket_path="s3://mast/level2/shots"
output_path="/rds/project/rds-mOlK9qn0PlQ/fairmast/metadata/level2"

source .venv/bin/activate
mpirun -n $SLURM_NTASKS python3 -m scripts.read_metadata --shot-min 11695 --shot-max 30475 --output-path $output_path --bucket-path $bucket_path --endpoint-url $endpoint_url
python3 -m scripts.group_parquet $output_path/sources ./mast-level2-sources.parquet
python3 -m scripts.group_parquet $output_path/signals ./mast-level2-signals.parquet
