#!/usr/bin/env python

import pandas as pd

def generate_chiral_csv(original_csv, mirrored_csv, output_csv):
    # Load the original and mirrored data
    df_original = pd.read_csv(original_csv)
    df_mirrored = pd.read_csv(mirrored_csv)
    
    # Concatenate the two DataFrames
    combined_df = pd.concat([df_original, df_mirrored])
    
    # Save the combined data to a new CSV file
    combined_df.to_csv(output_csv, index=False)
    print("Combined chiral data CSV generated:", output_csv)

# Specify file paths
original_csv = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/combined_labeled_pdb_2021aug02_sample/list.csv'
mirrored_csv = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/combined_labeled_pdb_2021aug02_sample/list_mirror.csv'
output_csv = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/combined_labeled_pdb_2021aug02_sample/chiral_out.csv'

# Generate the chiral_out.csv
generate_chiral_csv(original_csv, mirrored_csv, output_csv)