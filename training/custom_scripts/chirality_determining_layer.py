#!/usr/bin/env python

import torch
import torch.nn as nn

# Define the tensor shapes
vanilla_tensor_shape = (15, 574, 21)
dchiral_tensor_shape = (15, 574, 21)

class ChiralDetermine(nn.Module):
    def __init__(self, input_size: int, out1: int):
        super(ChiralDetermine, self).__init__()
        # Define the first linear layer for the combined tensors
        self.lin1_combined = nn.Linear(input_size * 2, out1)
        # Define the Sigmoid activation function
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, vanilla_tensor: torch.Tensor, dchiral_tensor: torch.Tensor) -> torch.Tensor:
        # Concatenate the tensors along the last dimension
        combined_tensor = torch.cat((vanilla_tensor, dchiral_tensor), dim=-1)
        # Pass the combined tensor through the linear layer
        combined_out = self.lin1_combined(combined_tensor)
        # Apply Sigmoid activation
        output = self.sigmoid(combined_out)
        return output

# Example usage
input_size = vanilla_tensor_shape[-1]  # This should be 21, as the feature size is 21 (amino acid probabilities)
out1 = 2  # Example output size for the linear layers, setting out1 to 42 for 42 features

# Instantiate the model
model = ChiralDetermine(input_size, out1)

# Generate some example input data
vanilla_tensor = torch.randn(vanilla_tensor_shape)
dchiral_tensor = torch.randn(dchiral_tensor_shape)

# Perform a forward pass
output = model(vanilla_tensor, dchiral_tensor)
print(output.shape)  # This should print torch.Size([15, 574, 42])

# Print the probabilities of each amino acid for the first sample and first position
print("Probabilities of each amino acid:")
print(output[0, 0, :])