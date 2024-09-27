#!/usr/bin/env python
import torch

def check_for_nan(pt_file):
    data = torch.load(pt_file)
    coordinates = data['xyz']

    # Check for NaN values
    if torch.isnan(coordinates).any():
        print(f"NaN values found in {pt_file}")
        nan_indices = torch.isnan(coordinates).nonzero(as_tuple=True)
        print(f"NaN indices: {nan_indices}")
    else:
        print(f"No NaN values in {pt_file}")

check_for_nan('/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02_sample/pdb/l3/1l3a_A.pt')
check_for_nan('/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/mirrored_pdb_2021aug02_sample/pdb/1l3a_A.pt')