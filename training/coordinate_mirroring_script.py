#!/usr/bin/env python

###################
# Script Overview #
###################
# This is a custom script to mirror coordinates from L-chiral proteins to make complimentary D-chiral proteins.
# To Dos: Read through directory, load data strucuture, extract xyz coordinates, mirror them, load them into a new data structure
# Note: Both input and output files should be in a pdb format.

###########
# Modules #
###########
import os
from Bio import PDB

#############
# Functions #
#############
    
########
# Main #
########

# Load PDB File
parser = PDB.PDBParser(QUIET=True)
structure = parser.get_structure('xyz', '/projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02_sample/pdb/l3')

# Information Extraction
for model in structure:
    for chain in model:
        for residue in chain:
            for atom in residue:
                print(f"Atom: {atom.get_id()}, Coordinates: {atom.get_coord()}")

else:
    print("No structure loaded from PDB file.")