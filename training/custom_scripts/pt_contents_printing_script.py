#!/usr/bin/env python

import torch

# Path to your .pt file
file_path = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02_sample/pdb/l3/1l3a_A.pt'

# Load the .pt file
data = torch.load(file_path)

# Print the keys in the .pt file
print("Keys in the .pt file:")
print(data.keys())

# Check for potential element-related data
element_keys = ['element', 'elements', 'atom_types', 'atom_names', 'atom_symbols']

# Print the contents under each key
for key, value in data.items():
    print(f"\nKey: {key}")
    print(f"Type of data: {type(value)}")
    print(f"Content:\n{value}")
