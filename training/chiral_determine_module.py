#!/usr/bin/env python

import torch
import torch.nn as nn

class ChiralDetermine(nn.Module):
    def __init__(self, input_size: int, out1: int):
        super(ChiralDetermine, self).__init__()
        self.lin1_combined = nn.Linear(input_size * 2, out1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, vanilla_tensor: torch.Tensor, dchiral_tensor: torch.Tensor) -> torch.Tensor:
        combined_tensor = torch.cat((vanilla_tensor, dchiral_tensor), dim=-1)
        combined_out = self.lin1_combined(combined_tensor)
        output = self.sigmoid(combined_out)
        return output

# Example usage
# if __name__ == "__main__":
#     input_size = 21 # Input size
#     out1 = 2 # Output size for binary classification

#     # Create the model
#     model = ChiralDetermine(input_size, out1)

#     # Example input tensorsß
#     vanilla_tensor = torch.randn(15, 574, input_size)
#     dchiral_tensor = torch.randn(15, 574, input_size)
    
#     # Forward 
#     output = model(vanilla_tensor, dchiral_tensor)
#     print("Output shape:", output.shape)
#     print("Output example:", output[0])