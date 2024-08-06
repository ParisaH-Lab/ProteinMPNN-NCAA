#!/usr/bin/env python

###################
# Script Overview #
###################
# This is a custom script to mirror coordinates from L-chiral amino acids to make complimentary D-chiral amino acids.
# To Dos: Read through directory, load data strucuture, extract xyz coordinates, mirror them, load them into a new data structure.
# Note: Both input and output files should be in a pt format.

###########
# Imports #
###########
import os
import torch
import numpy as np

#############
# Functions #
#############

# Mirror the x-coordinate by changing its sign
def mirror_xyz(xyz):
    # Creates a copy of 'xyz' and assigns it to a new variable.
    mirrored_xyz = xyz.copy()
    # Selects all rows and columns of the array, but only multiplies the x-coordinate by -1.
    mirrored_xyz[:, :, 0] = -mirrored_xyz[:, :, 0]
    return mirrored_xyz

########
# Main #
########

# Define input and output directories
input_directory = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02_sample/pdb'
output_directory = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/mirrored_pdb_2021aug02_sample/pdb'

# Loop over each file in the input directory
for root, _, files in os.walk(input_directory):
    for filename in files:
        if filename.endswith('.pt'):
            file_path = os.path.join(root, filename)
            
            # Load the data
            try:
                data = torch.load(file_path)
            except:
                continue
            
            # Access the 'xyz' key and mirror the coordinates if it exists
            if 'xyz' in data:
                original_xyz = data['xyz']
                mirrored_xyz = mirror_xyz(original_xyz.numpy())
                data['xyz'] = torch.tensor(mirrored_xyz)
                
                # Define the output file path, saving directly in the output directory
                output_filename = os.path.join(output_directory, filename)
                
                # Save the modified data
                try:
                    torch.save(data, output_filename)
                except:
                    pass

print("All files have been processed and mirrored coordinates saved.")