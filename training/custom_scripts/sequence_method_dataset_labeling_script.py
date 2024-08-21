#!/usr/bin/env python

import os
import torch

def recursively_update_sequence(seq, is_mirrored):
    """
    Recursively update the sequence to be lowercase if mirrored or uppercase if not mirrored.
    Handles nested lists of sequences.
    """
    if isinstance(seq, list):
        return [recursively_update_sequence(sub_seq, is_mirrored) for sub_seq in seq]
    elif isinstance(seq, str):
        return seq.lower() if is_mirrored else seq.upper()
    else:
        return seq

def update_sequences(input_dir, output_dir):
    # Determine if the data is mirrored 
    is_mirrored = 'mirrored' in input_dir.lower()
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.pt'):
            # Get input file
            file_path = os.path.join(input_dir, filename)
            data = torch.load(file_path)

            if 'seq' in data:
                original_sequence = data['seq']
                updated_sequence = recursively_update_sequence(original_sequence, is_mirrored)
                
                # Update the sequence
                data['seq'] = updated_sequence
                
                # Save the updated file in the correct output directory
                output_file_path = os.path.join(output_dir, filename)
                torch.save(data, output_file_path)

# Specify the input and output directories
regular_input_dir = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02_sample/pdb/l3'
regular_output_dir = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/labeled_pdb_2021aug02_sample/pdb'
mirrored_input_dir = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/mirrored_pdb_2021aug02_sample/pdb'
mirrored_output_dir = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/labeled_mirrored_pdb_2021aug02_sample/pdb'

# Process L-chiral residue sequence
update_sequences(regular_input_dir, regular_output_dir)

# Process D-chiral residue sequence
update_sequences(mirrored_input_dir, mirrored_output_dir)