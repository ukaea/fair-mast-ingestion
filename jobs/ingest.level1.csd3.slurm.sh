#!/bin/bash
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p ukaea-icl
#SBATCH --job-name=fair-mast-ingest-level1
#SBATCH --output=fair-mast-ingest_%A.out
#SBATCH --time=36:00:00
#SBATCH --mem=250G
#SBATCH --ntasks=64
#SBATCH -N 2


num_workers=$SLURM_NTASKS
source .venv/bin/activate
source ~/.uda-ssl.sh
export AWS_SHARED_CREDENTIALS_FILE="$PWD/.s5cfg.stfc"

# Output locations (local NetCDF on RDS + zarr on S3) come from the writers list in
# the config file; no separate upload job is needed for the zarr.
mpirun -n $num_workers \
    python3 -m src.level1.main -c ./configs/level1.csd3.yml --facility MAST \
    --uda-group-names \
    --shot-min 11695 --shot-max 30474 \
    -i abm act aga ahx ait alp amb amc ams anb ane anu arp asm atm ayc aye efm esm xbt xdc xim xmo xsx
