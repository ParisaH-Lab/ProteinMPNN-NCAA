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

# if __name__ == "__main__":
#     # Example usage
#     input_size = 21
#     out1 = 2
#     vanilla_tensor = torch.randn(15, 574, input_size)
#     dchiral_tensor = torch.randn(15, 574, input_size)
    
#     model = ChiralDetermine(input_size, out1)
#     output = model(vanilla_tensor, dchiral_tensor)
    
#     print("Output shape:", output.shape)
#     print("Output example:", output[0])