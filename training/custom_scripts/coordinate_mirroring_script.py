#!/usr/bin/env python

###################
# Script Overview #
###################
# This is a custom script to mirror coordinates from L-chiral amino acids to make complimentary D-chiral amino acids.
# To Dos: Read through directory, load data strucuture, extract xyz coordinates, mirror them, load them into a new data structure.
# Note: Both input and output files should be in a pt format.

###########
# Modules #
###########
import os
import torch
import numpy as np

# Define input and output directories
input_directory = '/projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02_sample/pdb/l3'
output_directory = '/projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/sample_mirrored_dataset'

#############
# Functions #
#############
# # Reverse the order of the coordinates along the specified axis
# def mirror_xyz(xyz):
#     mirrored_xyz = xyz[:, :, ::-1].copy()
#     return mirrored_xyz

# # Mirror the x-coordinate by changing its sign
# def mirror_xyz(xyz):
#     mirrored_xyz = xyz.copy()
#     mirrored_xyz[:, :, 0] *= -1
#     return mirrored_xyz

# # Mirror the x-coordinate by changing its sign
# def mirror_xyz(xyz):
#     # Creates a copy of 'xyz' and assigns it to a new variable.
#     mirrored_xyz = xyz.copy()
#     # Selects all rows and columns of the array, but only multiplies the x-coordinate by -1.
#     mirrored_xyz[:, :, 0] = -mirrored_xyz[:, :, 0]
#     return mirrored_xyz

def mirror_xyz(xyz):
    # Negate all coordinates
    mirrored_xyz = -1 * xyz
    return mirrored_xyz

# Test cases
# xyz1 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# # Original:
# # [[[ 1  2  3]
# #   [ 4  5  6]]
# #
# #  [[ 7  8  9]
# #   [10 11 12]]]
# # Mirrored along the last axis (z-axis):
# # [[[ 3  2  1]
# #   [ 6  5  4]]
# #
# #  [[ 9  8  7]
# #   [12 11 10]]]

# xyz2 = np.array([[[11, 22, 33], [44, 55, 66]], [[77, 88, 99], [100, 111, 122]]])
# # Original:
# # [[[ 11  22  33]
# #   [ 44  55  66]]
# #
# #  [[ 77  88  99]
# #   [100 111 122]]]
# # Mirrored along the last axis (z-axis):
# # [[[ 33  22  11]
# #   [ 66  55  44]]
# #
# #  [[ 99  88  77]
# #   [122 111 100]]]

# # Testing the function
# result1 = mirror_xyz(xyz1)
# result2 = mirror_xyz(xyz2)

# # Checking the results
# print("Original xyz1:")
# print(xyz1)
# print("\nMirrored xyz1:")
# print(result1)

# print("\nOriginal xyz2:")
# print(xyz2)
# print("\nMirrored xyz2:")
# print(result2)

# # Create a sample array
# arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

# # Apply the slicing operation
# sliced_arr = arr[:, :, ::-1]

# # Print the original and sliced arrays for comparison
# print("Original array:")
# print(arr)
# print("\nSliced array:")
# print(sliced_arr)

########
# Main #
########

# Load PDB File
# parser = PDB.PDBParser(QUIET=True)
# structure = parser.get_structure('xyz', '/projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02_sample/pdb/l3')

# # Information Extraction
# for model in structure:
#     for chain in model:
#         for residue in chain:
#             for atom in residue:
#                 print(f"Atom: {atom.get_id()}, Coordinates: {atom.get_coord()}")

# else:
#     print("No structure loaded from PDB file.")

# Initiated a loop over each file in my input directory.
for filename in os.listdir(input_directory):
    # Checks to see if the current file being iterated over ends with .pt.
    if filename.endswith('.pt'):
        # If the file does end in .pt, the script loads the using PyTorch's 'torch.load()' function.
        data = torch.load(os.path.join(input_directory, filename))
        
        # Access the 'xyz' key and mirror the coordinates (also checks to see if the loaded data has an 'xyz' key by default)
        if 'xyz' in data:
            # If the 'xyz' key exists, it retrieves the corresponding value and converts int into a NumPy array. It then applies the 'mirror_xyz' function to mirror the coordinates.
            mirrored_xyz = mirror_xyz(data['xyz'].numpy())
            # After mirroring the coordinates, it replaces the original 'xyz' data in the dict with the mirrored coordinates, and converts it back to a PyTorch tensor using the 'torch.tensor' function.
            data['xyz'] = torch.tensor(mirrored_xyz)
            
            # Output the modified data to a new .pt file in the output directory and prepends "mirrored_" to the og filename.
            # output_filename = os.path.join(output_directory, "mirrored_" + filename)  # Change output filename
            output_filename = os.path.join(output_directory, filename)
            
            # Saves the modified data dictionary to a new '.pt' file in the output directory.
            torch.save(data, output_filename)

# Let's me know when the script is done running.
print(f"Mirrored coordinates saved to {output_filename}")