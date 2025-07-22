#!/bin/bash
#SBATCH --job-name=CPUVecAdd
#SBATCH --output=%x.out
#SBATCH --account=ece_gy_9143-2025sp
#SBATCH --partition=n1s8-v100-1  # Using a V100 GPU partition as an example
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1

cd /scratch/stp8232/PartB


module load singularity
module load cuda

#Singularity container
CONTAINER=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif

# Run all commands in a single Singularity session
singularity exec --nv ${CONTAINER} bash -c "
cd /scratch/stp8232/PartB
make clean
make 2>/dev/null
make
./CPUVecAdd 1
./CPUVecAdd 5
./CPUVecAdd 10
./CPUVecAdd 50
./CPUVecAdd 100
"