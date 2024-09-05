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
import shutil

# Define input and output directories
input_directory = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02_sample/pdb/l3'
output_directory = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/mirrored_pdb_2021aug02_sample/pdb/l3'

def mirror_xyz(xyz):
    """Mirror the x-coordinates of the xyz tensor."""
    mirrored_xyz = xyz.copy()
    mirrored_xyz[:, :, 0] = -mirrored_xyz[:, :, 0]  # Negate the x-coordinates to mirror them
    return mirrored_xyz

def process_file(input_filepath, output_filepath):
    """Load a .pt file, mirror its xyz coordinates, and save it to a new location."""
    data = torch.load(input_filepath)
    if 'xyz' in data:
        mirrored_xyz = mirror_xyz(data['xyz'].numpy())
        data['xyz'] = torch.tensor(mirrored_xyz)
    torch.save(data, output_filepath)

def process_directory(input_dir, output_dir):
    """Walk through the directory, processing each .pt file and copying others."""
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            input_filepath = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_dir)
            output_root = os.path.join(output_dir, relative_path)
            os.makedirs(output_root, exist_ok=True)

            if file.endswith('.pt'):
                # Append 'mirror' before the file extension to create the mirrored filename
                name, ext = os.path.splitext(file)
                output_filename = f"{name}mirror{ext}"
                output_filepath = os.path.join(output_root, output_filename)
                process_file(input_filepath, output_filepath)
            else:
                # Copy non-.pt files directly to maintain the directory structure
                output_filepath = os.path.join(output_root, file)
                shutil.copyfile(input_filepath, output_filepath)

process_directory(input_directory, output_directory)
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