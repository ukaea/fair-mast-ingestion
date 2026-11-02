#!/bin/bash
# Level 2 ingestion of the MAST "Standard PF test shot." shots on CSD3.
#
# This job is a SLURM array job. Each array task ingests one batch of
# BATCH_SIZE shots from SHOT_FILE on one node.
#
# The array range must cover all the batches:
#   n_batches = ceil(n_shots / BATCH_SIZE)
#   --array=0-(n_batches - 1)
# The shot file has 47 shots and BATCH_SIZE is 8, thus 6 batches (0-5).
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
#SBATCH --mem=250G
#SBATCH --ntasks=10
#SBATCH -N 1
#SBATCH --array=0-5%3

set -euo pipefail

SHOT_FILE="campaign_shots/pf_test_shots.csv"
BATCH_SIZE=8

num_workers=$SLURM_NTASKS
source .venv/bin/activate
source ~/.uda-ssl.sh

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

# dask-mpi uses rank 0 for the scheduler and rank 1 for the client.
# The other ranks are workers.
mpirun -n $num_workers \
    python3 -m src.level2.main mappings/level2/mast.yml \
    -c ./configs/level2.csd3.yml \
    --shots $shots \
    --force-ip-check \
    -e camera_visible \
    -o "$output_dir"
