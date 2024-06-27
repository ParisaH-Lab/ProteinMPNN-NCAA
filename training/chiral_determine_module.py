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