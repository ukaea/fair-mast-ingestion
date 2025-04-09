#!/bin/bash
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p icelake
#SBATCH --job-name=fair-mast-ingest-level2
#SBATCH --output=fair-mast-ingest_%A.out
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH -N 1

source .venv/bin/activate
# python3 -m scripts.create_uda_metadata --shot-min 41139 --shot-max 51056 --output-dir /common/tmp/sjackson/data/uda/signals/
python3 -m scripts.read_metadata --shot-min 11695 --shot-max 30474 -n 16 --output-file ./level2.mast.stfc.db
