#!/bin/bash
#SBATCH -p cpu_medium,cpu_long
#SBATCH --mem=2G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=10
#SBATCH -t 5-00:00:00

module load julia
echo $HOSTNAME
echo $SLURM_ARRAY_TASK_ID
julia ~/Documents/Julia/SimulatedAnnealing/SimulatedAnnealsing_Poisson_Pairs.jl

wait

