#!/bin/bash
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p ukaea-icl
#SBATCH --job-name=fair-mast-ingest-level2
#SBATCH --output=fair-mast-ingest_%A.out
#SBATCH --time=36:00:00
#SBATCH --mem=250G
#SBATCH --ntasks=64
#SBATCH -N 8


num_workers=$SLURM_NTASKS
source .venv/bin/activate
source ~/.uda-ssl.sh
export AWS_SHARED_CREDENTIALS_FILE="$PWD/.s5cfg.stfc"

# Output locations (local NetCDF on RDS + zarr on S3) come from the writers list in
# the config file. Do NOT pass -o here: it overrides the output path of every writer,
# including redirecting the S3 zarr writer to local disk.
mpirun -n $num_workers \
    python3 -m src.level2.main mappings/level2/mast.yml -c ./configs/level2.csd3.yml \
    --shot-min 11695 --shot-max 30474 -e camera_visible
