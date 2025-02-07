#!/bin/bash
#SBATCH -A UKAEA-AP002-CPU
#SBATCH -p icelake
#SBATCH --job-name=fair-mast-ingest
#SBATCH --output=fair-mast-ingest_%A.out
#SBATCH --time=5:00:00
#SBATCH --mem=250G
#SBATCH --ntasks=128
#SBATCH -N 2


num_workers=$SLURM_NTASKS
source /rds/project/rds-mOlK9qn0PlQ/fairmast/uda-ssl.sh

mpirun -n $num_workers \
    python3 -m src.level1.main --facility MAST --shot-min 11695 --shot-max 30474 -i abm act aga ahx ait aiv alp amb amc ams ane anu asm asx ayc ayd epm epq esm rba rbb rbc rgb rgc xbt xdc xim xma xmb xmc xsx 


