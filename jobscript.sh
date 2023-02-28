#!/bin/bash
#SBATCH --job-name="Enc"
#SBATCH -p hgx2q # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH -w g002
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -t 5-12:00 # time (D-HH:MM)
#SBATCH -o output/slurm.%N.%j.out
#SBATCH -e output/slurm.%N.%j.err
##SBATCH --exclusive


RUNPATH="/home/rabink1/EEG_Encoder"
cd $RUNPATH

mkdir -p ~/output/

srun python main.py 

