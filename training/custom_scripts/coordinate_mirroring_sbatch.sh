#!/bin/bash
#SBATCH --account=parisahlab                    #REQUIRED: which account to use
#SBATCH --partition=gpulong               #REQUIRED: which partition to use
#SBATCH --job-name=coordinate_mirroring_sbatch                    #Job Name
#SBATCH --output=coordinate_mirroring_output_%j.out                    #File to store output
#SBATCH --error=coordinate_mirroring_output_%j.err                    #File to store error messages
#SBATCH --time=3-00:00:00                            #Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1                        #optional: number of nodes
#SBATCH --ntasks-per-node=1              #number of tasks to be launched per Node
#SBATCH --cpus-per-task=4                #optional: number of cpus, default is 1
#SBATCH --mem=64GB                        #optional: amount of memory, default is 4GB
#SBATCH --gpus=1                         #Necessary to fix CUDA error 
#SBATCH --constraint=gpu-40gb

conda activate mlfold2 

python coordinate_mirroring_script.py
