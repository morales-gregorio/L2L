#!/usr/bin/env bash
#SBATCH --account=hhd34
#SBATCH --nodes=2
#SBATCH --time=00:10:00
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.er
#SBATCH --mail-user=a.yegenoglu@fz-juelich.de
#SBATCH --mail-type=END
#SBATCH --partition=devel
#SBATCH --job-name=comet2ltl

source /p/project/chhd34/comet/activate
python l2l-comet-ga.py

