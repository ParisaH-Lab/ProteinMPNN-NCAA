#!/usr/bin/env python

import torch

# # Ensure that CUDA is available
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')
#     print("CUDA is not available. Loading on CPU.")

# Load the checkpoint with CPU mapping
checkpoint = torch.load('/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/vanilla_sample_training_output/model_weights/epoch_last.pt', 'cuda' if torch.cuda.is_available() else 'cpu')

# Check for keys
print("Checkpoint keys:", checkpoint.keys())

# Example keys: 'epoch', 'model_state_dict', 'optimizer_state_dict'
if 'epoch' in checkpoint:
    print(f"Epoch: {checkpoint['epoch']}")

if 'model_state_dict' in checkpoint:
    print("Model state dictionary loaded.")

if 'optimizer_state_dict' in checkpoint:
    print("Optimizer state dictionary loaded.")

# Alternate script if you want to see what is in the dictionaries:
# # Check for keys
# print("Checkpoint keys:", checkpoint.keys())

# # Print details for each key
# if 'epoch' in checkpoint:
#     print(f"Epoch: {checkpoint['epoch']}")

# if 'step' in checkpoint:
#     print(f"Step: {checkpoint['step']}")

# if 'num_edges' in checkpoint:
#     print(f"Number of edges: {checkpoint['num_edges']}")

# if 'noise_level' in checkpoint:
#     print(f"Noise level: {checkpoint['noise_level']}")

# if 'model_state_dict' in checkpoint:
#     print("Model state dictionary contents:")
#     for key, value in checkpoint['model_state_dict'].items():
#         print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else type(value)}")

# if 'optimizer_state_dict' in checkpoint:
#     print("Optimizer state dictionary contents:")
#     for key, value in checkpoint['optimizer_state_dict'].items():
#         print(f"{key}: {value}")