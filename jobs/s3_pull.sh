#!/bin/bash
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p ukaea-spr
#SBATCH --job-name=fair-mast-pull
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -N 1

endpoint_url="https://echo.stfc.ac.uk/"
output_dir="/rds/project/rds-mOlK9qn0PlQ/ibm/fair-mast/2025-05-13"

s5cmd --no-sign-request --endpoint-url $endpoint_url cp "s3://mast/level2/shots/*.zarr/*" $output_dir
