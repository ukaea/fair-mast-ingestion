#!/bin/bash
# Level 2 ingestion of the MAST PF test shots on CSD3.
#
# This job is a SLURM array job. Each array task ingests one batch of
# BATCH_SIZE shots from SHOT_FILE on one node. The output is one NetCDF file
# for each shot.
#
# The array range must cover all the batches:
#   n_batches = ceil(n_shots / BATCH_SIZE)
#   --array=0-(n_batches - 1)
# The shot file has 762 shots and BATCH_SIZE is 8, thus 96 batches (0-95).
# If you change the shot file or BATCH_SIZE, change --array to agree.
# The "%3" limits the number of array tasks that run at the same time.
#
# The plasma current (Ip) check is disabled with --force-ip-check, because
# PF test shots have no plasma.
#
# Submit from the root of the repository:
#   sbatch jobs/ingest.level2.pf_test_shots.csd3.slurm.sh

#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p ukaea-icl
#SBATCH --job-name=fair-mast-ingest-pf-test
#SBATCH --output=fair-mast-ingest-pf-test_%A_%a.out
#SBATCH --time=12:00:00
# One process that starts a local dask cluster with one worker for each shot
# in the batch. Do not use mpirun: the venv does not have mpi4py and dask-mpi,
# thus each MPI rank starts its own cluster and processes all the shots.
# ukaea-icl has a limit of 3370 MB for each CPU. Do not set --mem: memory is
# set by the CPU count (16 CPUs = approximately 53 GB).
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -N 1
#SBATCH --array=0-95%3

set -euo pipefail

SHOT_FILE="campaign_shots/mast_pf_test_shots.csv"
BATCH_SIZE=8

source .venv/bin/activate
# Load the UDA host and SSL settings. Without them, all UDA reads fail and
# the job writes no files.
uda_env=""
for f in ~/uda-ssl.sh ~/.uda-ssl.sh; do
    if [ -f "$f" ]; then
        uda_env="$f"
        break
    fi
done
if [ -z "$uda_env" ]; then
    echo "ERROR: no UDA settings file (~/uda-ssl.sh or ~/.uda-ssl.sh). Stop." >&2
    exit 1
fi
source "$uda_env"
if [ -z "${UDA_HOST:-}" ]; then
    echo "ERROR: UDA_HOST is not set after source of ${uda_env}. Stop." >&2
    exit 1
fi

output_dir="/rds/project/rds-mOlK9qn0PlQ/fairmast/level2/pf_test_shots"
mkdir -p "$output_dir"

# Get the shots for this array task. Skip the header row of the shot file.
start=$((SLURM_ARRAY_TASK_ID * BATCH_SIZE + 1))
end=$((start + BATCH_SIZE - 1))
shots=$(tail -n +2 "$SHOT_FILE" | sed -n "${start},${end}p" | tr '\n' ' ')

if [ -z "${shots// /}" ]; then
    echo "No shots for array task ${SLURM_ARRAY_TASK_ID}. Stop."
    exit 0
fi

echo "Array task ${SLURM_ARRAY_TASK_ID}: shots ${shots}"

python3 -m src.level2.main mappings/level2/mast.yml \
    -c ./configs/level2.csd3.yml \
    --shots $shots \
    --force-ip-check \
    -e camera_visible \
    -f netcdf \
    -n "$BATCH_SIZE" \
    -o "$output_dir"
