#!/usr/bin/env python

import torch
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom

def pt_to_pdb(pt_file, pdb_file):
    # Load .pt file
    data = torch.load(pt_file)
    
    # Extract coordinates and amino acid residues from the protein
    coordinates = data['xyz']  # Coordinates
    sequence = data['seq']     # Amino acid sequence
    
    # Initialize PDB structure
    structure = Structure.Structure("structure")
    model = Model.Model(0)
    chain = Chain.Chain('A')
    
    # Populate the PDB structure
    for i, (residue_name, coords) in enumerate(zip(sequence, coordinates)): # Interate over each residue and its sequence/coordinates and puts them in a tuple to build protein structure (residue_name (i.e. ALA), 3D coordinates for amino acid)
            residue = Residue.Residue((' ', i, ' '), residue_name, '') # Creates residue object for each amino acid in the protein structure (standard residue ' ', identifier (i), insertion code ' ') (residue_name (i.e. ALA), amino acid sequence)
            for j, atom_coord in enumerate(coords): # Iterate over each set of coordinates within the current residue coordinates  
                if not torch.isnan(atom_coord).any():  # Skip if the atom coordinates are NaN
                    atom_name = f"CA" if j == 0 else f"C{j+1}" # Name the first atom (j = 0) "CA", second atom "C2", etc.
                    element = 'C'  # Assuming all atoms are carbon... could be problematic, but there's no key in the .pt file to help define the atoms more accurately
                    atom = Atom.Atom(atom_name, atom_coord, 1.0, data['occ'][i][j], ' ', atom_name, j, element=element) # creates atom: 1 is the occupancy of the atom, data['occ'][i][j] this is to extract the B-factor
                    residue.add(atom) # Adds the atom object to the residue object
            chain.add(residue) # adds the residue to the chain object in the protein structure
    
    model.add(chain) # adds fully constructed chain to model
    structure.add(model) # add the fully complete model to the structure
    
    # Write the structure to a PDB file
    io = PDBIO() # Initializes PDBIO object to handle input/outpt of pdb file
    io.set_structure(structure) # Uses fully constructed structure as data source
    io.save(pdb_file) # Save the structure to a PDB file at the specified file path.

# Main execution
if __name__ == '__main__':
    # Convert standard .pt file to .pdb file
    pt_to_pdb('/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02_sample/pdb/l3/1l3a_A.pt', '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/outputs/pt_to_pdb_outputs/standard_output.pdb')

    # Convert mirrored .pt file to .pdb file
    pt_to_pdb('/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/mirrored_pdb_2021aug02_sample/pdb/1l3a_A.pt', '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/outputs/pt_to_pdb_outputs/mirrored_output.pdb')