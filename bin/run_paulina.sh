#!/bin/env bash
#SBATCH --account=hhd34
#SBATCH --nodes=8
#SBATCH --time=00:15:00
#SBATCH --output=/p/scratch/chhd34/comet_p/output_%j.out
#SBATCH	--error=/p/scratch/chhd34/comet_p/error_%j.out
#SBATCH --mail-user=p.dabrowska@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --partition=devel
#SBATCH --job-name=comet

source /p/project/chhd34/comet/activate
python l2l-comet-ga.py
