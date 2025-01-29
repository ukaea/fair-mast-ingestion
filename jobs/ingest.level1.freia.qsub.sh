#!/bin/bash

# Choose parallel environment
#$ -pe mpi 16

# Specify the job name in the queue system
#$ -N fair-mast-writer

# Start the script in the current working directory
#$ -cwd

# Time requirements
#$ -l h_rt=48:00:00
#$ -l s_rt=48:00:00

num_workers=16
source .venv/bin/activate
python3 -m src.level1.main -n $num_workers --facility MASTU --shot-min 41139 --shot-max 51056 -i  abm act aga ahx ait aiv alp amb amc ams anb ane anu asx ayc ayd epm epq esm rba rbb rbc rgb rgc xbt xdc xim xma xmb xmc xsx 