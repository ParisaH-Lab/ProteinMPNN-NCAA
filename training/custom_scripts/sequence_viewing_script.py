#!/usr/bin/env python

import torch

def view_sequences(pt_file):
    # Load the .pt file
    data = torch.load(pt_file)
    
    # Check if the 'seq' key exists and print the sequence
    if 'seq' in data:
        sequence = data['seq']
        print(f"Sequence in {pt_file}:")
        print(sequence)
    else:
        print(f"'seq' key not found in {pt_file}")

# Specify the .pt files you want to inspect
unmirrored_pt_file = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/labeled_pdb_2021aug02_sample/pdb/1l3a_A.pt'
mirrored_pt_file = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/labeled_mirrored_pdb_2021aug02_sample/pdb/1l3a_A.pt'

# View the sequences in both files
view_sequences(unmirrored_pt_file)
view_sequences(mirrored_pt_file)
