#!/usr/bin/env python

# Imports
import torch
from esmfold import ESMFold
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

# Load ESMFold model
model = ESMFold()

# Prepare inputs for the model
aa_tensor = torch.tensor([[ord(aa) - ord('A') for aa in sequence]], dtype=torch.int64)
mask_tensor = torch.ones_like(aa_tensor)  # Since we are not masking any residues, we use ones

# Run the forward pass to get the structure
with torch.no_grad():
    structure = model.forward(aa_tensor, mask=mask_tensor)

# Structure now contains various outputs like positions, angles, etc.
# You can use 'structure["positions"]' and 'structure["aatype"]' to generate PDB file.

# Construct the output dictionary for output_to_pdb
output_dict = {
    "positions": structure["positions"],
    "aatype": structure["aatype"],
    "atom14_atom_exists": structure["atom14_atom_exists"],
    "residue_index": structure["residue_index"],  # This is necessary to ensure residue indexing is correct
}

# Convert to PDB format using ESMFold's helper function
pdb_strings = output_to_pdb(output_dict)

# Save the PDB string to a file
with open("output.pdb", "w") as pdb_file:
    pdb_file.write(pdb_strings[0])  # Assuming single structure output