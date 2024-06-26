#!/usr/bin/env python

import torch
import torch.nn as nn

class NewComboChiral(nn.Module):
    def __init__(self, 
                 edge_features: int, 
                 hidden_dim: int, 
                 num_encoder_layers: int, 
                 num_decoder_layers: int, 
                 dropout: float, 
                 k_neighbors: int, 
                 augment_eps: float, 
                 input_size: int, 
                 out1: int):
        super(NewComboChiral, self).__init__()

    vanilla_train_weights = torch.load("/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/vanilla_sample_training_output/model_weights")
    
    self.vanilla.load_state_dict(vanilla_train_weights["vanilla_trained_weights"])
    self.dchiral.load_state_dict(["dchiral_trained_weights"])

    self.vanilla.train(False)
    self.dchiral.train(False)

def forward(self, x):
      vanilla_out = self.vanilla(x)
      dchiral_out = self.dchiral(x)
      combine = torch.cat((vanilla_out, dchiral_out), dim = 1)
      return self.chiral_determine(combine)