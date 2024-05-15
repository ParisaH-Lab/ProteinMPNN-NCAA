#!/bin/bash
#SBATCH --account=bgmp                    #REQUIRED: which account to use
#SBATCH --partition=gpu               #REQUIRED: which partition to use
#SBATCH --job-name=vanilla_sample_training_sbatch                    #Job Name
#SBATCH --output=vanilla_sample_training_output_%j.out                    #File to store output
#SBATCH --error=vanilla_sample_training_output_%j.err                    #File to store error messages
#SBATCH --time=1-00:00:00                            #Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1                        #optional: number of nodes
#SBATCH --cpus-per-task=4                #optional: number of cpus, default is 1
#SBATCH --mem=32GB                        #optional: amount of memory, default is 4GB
#SBATCH --gpus=1                         #Necessary to fix CUDA error 
#SBATCH --constraint=gpu-40gb

conda activate mlfold2 

python training.py --path_for_training_data /projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02_sample \
    --path_for_outputs /projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/vanilla_sample_training_output
