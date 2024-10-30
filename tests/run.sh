#!/bin/bash

#SBATCH -G a30:1
#SBATCH -c 16
#SBATCH --mem 20G
#SBATCH -p general
#SBATCH -t 2-00:00:00   # time in d-hh:mm:ss



module purge

module load mamba/latest

source activate DLA

python3 Stability.py