#!/usr/bin/env python

import torch
import torch.nn as nn
from chiral_determine_module import ChiralDetermine
from training import ProteinMPNN

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

        # Instantiate vanilla, dchiral, and chiraldetermine
        self.vanilla = ProteinMPNN(node_features=hidden_dim, 
                                   edge_features=edge_features, 
                                   hidden_dim=hidden_dim, 
                                   num_encoder_layers=num_encoder_layers, 
                                   num_decoder_layers=num_decoder_layers, 
                                   k_neighbors=k_neighbors, 
                                   dropout=dropout, 
                                   augment_eps=augment_eps)

        self.dchiral = ProteinMPNN(node_features=hidden_dim, 
                                   edge_features=edge_features, 
                                   hidden_dim=hidden_dim, 
                                   num_encoder_layers=num_encoder_layers, 
                                   num_decoder_layers=num_decoder_layers, 
                                   k_neighbors=k_neighbors, 
                                   dropout=dropout, 
                                   augment_eps=augment_eps) 
        
        self.chiraldetermine = ChiralDetermine(input_size=input_size,
                                               out1=out1)

        vanilla_train_weights = torch.load("/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/vanilla_sample_training_output/model_weights/epoch200_step564.pt", map_location=torch.device('cpu'))
        # print(vanilla_train_weights.keys())
        self.vanilla.load_state_dict(vanilla_train_weights["model_state_dict"])

        dchiral_train_weights = torch.load("/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/mirrored_sample_training_output/model_weights/epoch200_step606.pt", map_location=torch.device('cpu'))
        # print(dchiral_train_weights.keys())
        self.dchiral.load_state_dict(dchiral_train_weights["model_state_dict"])

        self.vanilla.train(False)
        self.dchiral.train(False)

    def forward(self, x):
        vanilla_out = self.vanilla(x)
        dchiral_out = self.dchiral(x)

        new_combo = self.chiraldetermine(vanilla_out, dchiral_out)
        
        return new_combo

if __name__ == "__main__":
    # Example usage within the main block
    edge_features = 64
    hidden_dim = 128
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.1
    k_neighbors = 10
    augment_eps = 0.05
    input_size = 21
    out1 = 2

    # Instantiate the model
    model = NewComboChiral(edge_features=edge_features,
                           hidden_dim=hidden_dim,
                           num_encoder_layers=num_encoder_layers,
                           num_decoder_layers=num_decoder_layers,
                           dropout=dropout,
                           k_neighbors=k_neighbors,
                           augment_eps=augment_eps,
                           input_size=input_size,
                           out1=out1)

    # Prepare your input data 'x'
    x = torch.randn(1, input_size)  # Example: create a dummy input tensor

    # Perform the forward pass
    output = model(x)

    # Now 'output' contains the result of your model computation
    print(output)