#!/bin/sh
### General options
###specify queue
#BSUB -q "hpc"
###set the job name
#BSUB -J test
###ask for number of cores (default: 1)
#BSUB -n 24
###Select the resources
#BSUB -R "rusage[mem=8GB]"
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
###set walltime limit: hh:mm
#BSUB -W 70:00
### -- set the email address --
#BSUB -u s146996@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
###specify output and error file. %J is the job-id
###-o and -e mean append, -oo and -eo mean overwrite
#BSUB -oo Desktop/models2/model3/test_%J.out
#BSUB -eo Desktop/models2/model3/test_%J.err
# Load modules
module load python3/3.6.2

python3 Desktop/BombermanRL/evolutionarystrategies/main.py
