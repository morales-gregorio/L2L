#!/bin/bash -x
#SBATCH --account=hhd34
#SBATCH --nodes=2
#SBATCH --time=0:30:00
#SBATCH --mail-user=s.diaz@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --partition=devel

python l2l-comet-ga.py
