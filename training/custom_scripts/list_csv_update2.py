#!/usr/bin/env python
import pandas as pd

def update_identifiers(row):
    """
    Adds 1000 to both the HASH and CLUSTER numbers to create a new mirrored version.
    Assumes HASH and CLUSTER can be converted directly to integers.
    """
    # Convert HASH to an integer, add 1000, and convert back to string if needed
    row['HASH'] = f"{int(row['HASH']) + 1000}"
    # Add 1000 to the cluster number directly
    row['CLUSTER'] += 1000
    # Convert the sequence to lowercase
    row['SEQUENCE'] = row['SEQUENCE'].lower()
    return row

def create_mirrored_file(input_file, output_file):
    """
    Reads a CSV file, applies transformations to HASH and CLUSTER,
    and writes the results to a new CSV file.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(input_file)
        
        # Update HASH and CLUSTER
        df = df.apply(update_identifiers, axis=1)
        
        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_file, index=False)
        print(f"Mirrored file saved as {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Specify the paths to the input and output CSV files
input_csv = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/combined_labeled_pdb_2021aug02_sample2/list.csv'
output_csv = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/combined_labeled_pdb_2021aug02_sample2/list_mirror.csv'

# Call the function to create the mirrored CSV
create_mirrored_file(input_csv, output_csv)