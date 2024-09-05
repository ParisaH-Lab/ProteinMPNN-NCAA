#!/usr/bin/env python

import os
import torch
import shutil

input_directory = '/projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02_sample'
output_directory = '/projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/datasets/mirrored_pdb_2021aug02_sample/pdb'

def mirror_xyz(xyz):
    mirrored_xyz = xyz.copy()
    mirrored_xyz[:, :, 0] = -mirrored_xyz[:, :, 0]  # Mirror the x-coordinates
    return mirrored_xyz

def process_file(input_filepath, output_filepath):
    data = torch.load(input_filepath)
    if 'xyz' in data:
        mirrored_xyz = mirror_xyz(data['xyz'].numpy())
        data['xyz'] = torch.tensor(mirrored_xyz)
    torch.save(data, output_filepath)

def process_directory(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            input_filepath = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_dir)
            output_root = os.path.join(output_dir, relative_path)
            os.makedirs(output_root, exist_ok=True)

            if file.endswith('.pt'):
                # Create the mirrored filename
                name, ext = os.path.splitext(file)
                output_filename = f"{name}mirror{ext}"
                output_filepath = os.path.join(output_root, output_filename)
                
                # Process the file
                process_file(input_filepath, output_filepath)
            else:
                output_filepath = os.path.join(output_root, file)
                shutil.copyfile(input_filepath, output_filepath)

process_directory(input_directory, output_directory)
print("All files have been processed and mirrored coordinates saved.")