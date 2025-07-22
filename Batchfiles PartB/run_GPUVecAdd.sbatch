#!/bin/bash
#SBATCH --job-name=GPUVecAdd
#SBATCH --output=%x.out
#SBATCH --account=ece_gy_9143-2025sp
#SBATCH --partition=n1s8-v100-1
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
make GPUVecAdd
nvprof ./GPUVecAdd 1 1 1
nvprof ./GPUVecAdd 1 1 5
nvprof ./GPUVecAdd 1 1 10
nvprof ./GPUVecAdd 1 1 50
nvprof ./GPUVecAdd 1 1 100
nvprof ./GPUVecAdd 1 256 1
nvprof ./GPUVecAdd 1 256 5
nvprof ./GPUVecAdd 1 256 10
nvprof ./GPUVecAdd 1 256 50
nvprof ./GPUVecAdd 1 256 100
nvprof ./GPUVecAdd 16 256 1
nvprof ./GPUVecAdd 16 256 5
nvprof ./GPUVecAdd 16 256 10
nvprof ./GPUVecAdd 16 256 50
nvprof ./GPUVecAdd 16 256 100
"