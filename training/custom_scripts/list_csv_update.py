#!/usr/bin/env python

import pandas as pd
import hashlib

# Define transformation functions
def generate_new_hash(old_hash):
    """Generates a new hash by appending 'mirror_' prefix to the old hash."""
    return "mirror_" + str(old_hash)

def update_cluster(old_cluster):
    """Prepends 'mirror_' to the old cluster number to generate a new cluster identifier."""
    return "mirror_" + str(old_cluster)

def convert_sequence(sequence):
    """Converts the sequence to lowercase."""
    return sequence.lower()

def update_list_csv(input_file, output_file):
    try:
        # Load the CSV file
        df = pd.read_csv(input_file)
        
        # Apply transformations
        df['HASH'] = df['HASH'].apply(generate_new_hash)
        df['CLUSTER'] = df['CLUSTER'].apply(lambda x: update_cluster(x))
        df['SEQUENCE'] = df['SEQUENCE'].apply(convert_sequence)
        
        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_file, index=False)
        print(f"Updated file saved as {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Specify the paths to the input and output CSV files
input_csv = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/combined_labeled_pdb_2021aug02_sample2/list.csv'
output_csv = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/combined_labeled_pdb_2021aug02_sample2/list_mirror.csv'

# Call the function to update the CSV
update_list_csv(input_csv, output_csv)