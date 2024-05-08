#!/usr/bin/env python

###################
# Script Overview #
###################
# This is a custom script to view contents of 'xyz' dictionaries within the .pt tensors by removing the nan mask. 

###########
# Modules #
###########
import argparse
import torch

############
# Argparse #
############
def get_args():
    parser = argparse.ArgumentParser(description='Process .pt tensor files')
    parser.add_argument('input_file', type=str, help='Path to the input .pt file')
    return parser.parse_args()

args = get_args()
input_file = args.input_file

########
# Main #
########
data = torch.load(input_file)

# Process the loaded data
if isinstance(data, dict):
    for key, value in data.items():
        print(f"Key: {key}, Value: {value}")

# Find NaN values in 'xyz' tensor
xyz_tensor = data['xyz']
nan_mask = torch.isnan(xyz_tensor)
nan_indices = torch.nonzero(nan_mask, as_tuple=False)

# Print specific NaN values
print("Specific NaN Values:")
for idx in nan_indices:
    print(f"Index: {idx}, Value: {xyz_tensor[tuple(idx)]}")

# Ex. Bash Command: /projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/nan_mask_removal_script.py /projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02_sample/pdb/l3/1l3a_A.pt