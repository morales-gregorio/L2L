#!/bin/bash -x
#SBATCH --account=hhd34
#SBATCH --nodes=1
#SBATCH --time=00:00:20
#SBATCH --mail-user=a.morales-gregorio@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --partition=devel
#SBATCH --job-name=comet

source /p/project/chhd34/comet/activate
python l2l-comet-ga.py
