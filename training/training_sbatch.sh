#!/bin/bash
#SBATCH --account=bgmp                    #REQUIRED: which account to use
#SBATCH --partition=gpu               #REQUIRED: which partition to use
#SBATCH --job-name=training_sbatch                    #Job Name
#SBATCH --output=training_output                    #File to store output
#SBATCH --error=training_output_error                    #File to store error messages
#SBATCH --time=1-00:00:00                            #Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1                        #optional: number of nodes
#SBATCH --ntasks-per-node=1              #number of tasks to be launched per Node
#SBATCH --cpus-per-task=4                #optional: number of cpus, default is 1
#SBATCH --mem=32GB                        #optional: amount of memory, default is 4GB
#SBATCH --gpus=1                         #Necessary to fix CUDA error 
#SBATCH --constraint=gpu-40gb

python training.py --path_for_training_data /projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02 --path_for_outputs /projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/outputs
