#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[4,7-8,16-17,25,27-31] 
module load  miniforge3/24.1   compilers/cuda/12.4   cudnn/8.9.5.29_cuda12.x   compilers/gcc/11.3.0  
source activate py312
export PYTHONUNBUFFERED=1

python main.py
