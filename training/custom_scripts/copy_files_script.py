#!/bin/bash

# Source directory containing .pt files
source_dir="/projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02_sample/pdb/l3"

# Destination directory for files without capital letters
destination_dir="/projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/mirrored_pdb_2021aug02_sample/pdb/l3"

# Copy files without capital letters to the destination directory
find "$source_dir" -maxdepth 1 -type f -iname '*.pt' ! -name '*[A-Z]*' -exec cp {} "$destination_dir" \;