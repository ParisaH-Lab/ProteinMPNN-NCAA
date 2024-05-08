#!/usr/bin/env python

###################
# Script Overview #
###################
# This is a custom script to view the contents of the tensors with the .pt files. It also checks that the nan values for the 'xyz' coordinates within a tensor have data.

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

print(data['xyz'])

if 'xyz' in data:
    results = torch.sum(data['xyz'].isnan()) == data['xyz'].flatten().size(0)
    print(results)
else:
    print("Key 'xyz' not found in the loaded data.")

# Ex. Bash Command: /projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/tensor_viewing_script.py /projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02_sample/pdb/l3/1l3a_A.pt