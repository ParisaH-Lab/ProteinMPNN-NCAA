#!/usr/bin/env python

import os
import shutil

# Define the directories for mirrored and unmirrored datasets
unmirrored_source_directory = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/labeled_pdb_2021aug02_sample/pdb/l3'  # Modify this to your actual unmirrored dataset directory
mirrored_source_directory = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/labeled_mirrored_pdb_2021aug02_sample/pdb/l3'  # Modify this to your actual mirrored dataset directory

# Define the base output directory for your pdb structure
base_output_directory = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/combined_labeled_pdb_2021aug02_sample/pdb'  # Modify this to the target base directory

# Define the target directories for l3 and l3mirror
l3_directory = os.path.join(base_output_directory, 'l3')
l3mirror_directory = os.path.join(base_output_directory, 'l3mirror')

def move_files(source_dir, target_dir):
    """
    Moves all .pt files from the source directory to the target directory.
    """
    files_moved = 0
    for filename in os.listdir(source_dir):
        if filename.endswith('.pt'):
            src_file = os.path.join(source_dir, filename)
            dest_file = os.path.join(target_dir, filename) # Creating full file path
            shutil.move(src_file, dest_file)
            print(f"Moved {filename} from {source_dir} to {target_dir}")
            files_moved += 1

# Move unmirrored files to l3 directory
move_files(unmirrored_source_directory, l3_directory)

# Move mirrored files to l3mirror directory
move_files(mirrored_source_directory, l3mirror_directory)

