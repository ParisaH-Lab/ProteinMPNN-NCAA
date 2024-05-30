#!/usr/bin/env python

import os
import torch
import numpy as np
import shutil

input_directory = '/projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/datasets/pdb_2021aug02'
output_directory = '/projects/bgmp/lmjone/internship/ProteinMPNN-PH/training/mirrored_pdb_2021aug02/pdb'

def mirror_xyz(xyz):
    mirrored_xyz = xyz.copy()
    mirrored_xyz[:, :, 0] = -mirrored_xyz[:, :, 0]
    return mirrored_xyz

def process_file(input_filepath, output_filepath):
    data = torch.load(input_filepath)
    if 'xyz' in data:
        mirrored_xyz = mirror_xyz(data['xyz'].numpy())
        data['xyz'] = torch.tensor(mirrored_xyz)
        torch.save(data, output_filepath)
    else:
        shutil.copyfile(input_filepath, output_filepath)

def process_directory(input_dir, output_dir):
    for item in os.listdir(input_dir):
        input_item_path = os.path.join(input_dir, item)
        output_item_path = os.path.join(output_dir, item)
        if os.path.isdir(input_item_path):
            os.makedirs(output_item_path, exist_ok=True)
            process_directory(input_item_path, output_item_path)
        elif item.endswith('.pt'):
            process_file(input_item_path, output_item_path)
        else:
            shutil.copyfile(input_item_path, output_item_path)

process_directory(input_directory, output_directory)
print("All files have been processed and mirrored coordinates saved.")