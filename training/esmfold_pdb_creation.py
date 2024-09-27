#!/usr/bin/env python

# Imports
import torch
from esm.esmfold.v1.misc import output_to_pdb

# Load the .pt file
data = torch.load('/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02_sample/pdb/l3/1l3a_A.pt')  # Replace with your actual file path

# Extract the relevant parts
sequence = data['seq']  # This is the sequence as a string
xyz = data['xyz']  # The tensor of XYZ coordinates
mask = data['mask']  # The mask indicating valid coordinates
bfac = data['bfac']  # B-factors (not used in this script but available)
occ = data['occ']  # Occupancy (not used in this script but available)

# Apply mask to filter out invalid coordinates
valid_mask = mask > 0  # Boolean tensor where True indicates valid entries

# Mask the XYZ tensor to only keep valid coordinates
xyz_valid = xyz[valid_mask].view(-1, 3)  # Reshape the valid coordinates into (N, 3)

# Prepare the amino acid types in a format compatible with output_to_pdb
aatype = torch.tensor([[ord(res) - ord('A') for res in sequence]])  # Convert sequence to indices

# Construct the output dictionary expected by output_to_pdb
output_dict = {
    "positions": xyz_valid.unsqueeze(0),  # Add batch dimension
    "aatype": aatype,  # Use the converted sequence indices
    "atom14_atom_exists": valid_mask.unsqueeze(0),  # Use the valid_mask to determine existing atoms
}

# Convert to PDB format
pdb_strings = output_to_pdb(output_dict)

# Save the PDB string to a file
with open("output.pdb", "w") as pdb_file:
    pdb_file.write(pdb_strings[0])  # If you only have one structure