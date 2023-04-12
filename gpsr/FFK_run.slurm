#!/bin/bash
#SBATCH --time=120:00:00 # walltime, abbreviated by -t
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u1079968@umail.utah.edu
#SBATCH --nodes=1
#SBATCH --ntasks=40
# additional information for allocated clusters
#SBATCH --account=hochhalter-np     # account - abbreviated by -A
#SBATCH --partition=hochhalter-shared-np # partition, abbreviated by -p

##SBATCH --pty 
##SBATCH --qos=hochhalter-high-np

#pwd

python main_script.py

