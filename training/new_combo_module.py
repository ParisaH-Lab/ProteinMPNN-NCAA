#!/usr/bin/env python

import torch
import torch.nn as nn
from chiral_determine_module import ChiralDetermine
from model_utils import ProteinMPNN

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

        # # Load pretrained weights
        # vanilla_train_weights = torch.load("/mnt/c/Users/thefr/Downloads/epoch200_step564.pt", map_location=torch.device('cpu'))
        # dchiral_train_weights = torch.load("/mnt/c/Users/thefr/Downloads/epoch200_step606.pt", map_location=torch.device('cpu'))
        #
        #
        # # Extract model state dictionaries
        # vanilla_model_state_dict = vanilla_train_weights['model_state_dict']
        # dchiral_model_state_dict = dchiral_train_weights['model_state_dict']
        #
        # # Load the state dictionaries
        # self.vanilla.load_state_dict(vanilla_model_state_dict)
        # self.dchiral.load_state_dict(dchiral_model_state_dict)
        #
        # # Freeze the weights in the vanilla and d-chiral models 
        # for param in list(self.vanilla.parameters()) + list(self.dchiral.parameters()):
        #     param.requires_grad = False
        #
        # # Set models to evaluation mode
        # self.vanilla.eval()
        # self.dchiral.eval()

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        # Pass input through vanilla and dchiral models
        vanilla_out = self.vanilla(X, S, mask, chain_M, residue_idx, chain_encoding_all)
        dchiral_out = self.dchiral(X, S, mask, chain_M, residue_idx, chain_encoding_all)

        new_combo = self.chiraldetermine(vanilla_out, dchiral_out)
        # print(f"new_combo shape before reduction: {new_combo.shape}")
    
        return new_combo, vanilla_out, dchiral_out

# if __name__ == "__main__":
#     # Example usage within the main block
#     edge_features = 128
#     hidden_dim = 128
#     num_encoder_layers = 3
#     num_decoder_layers = 3
#     dropout = 0.1
#     k_neighbors = 10
#     augment_eps = 0.05
#     input_size = 21
#     out1 = 2

#     # Instantiate the model
#     model = NewComboChiral(edge_features=edge_features,
#                            hidden_dim=hidden_dim,
#                            num_encoder_layers=num_encoder_layers,
#                            num_decoder_layers=num_decoder_layers,
#                            dropout=dropout,
#                            k_neighbors=k_neighbors,
#                            augment_eps=augment_eps,
#                            input_size=input_size,
#                            out1=out1)

#     # Create dummy input tensors
#     S = torch.randint(0, 20, (2, 1746), dtype=torch.long)  # Example: create a dummy input tensor for sequence with appropriate type
#     X = torch.randn([2, 1746, 4, 3])                        # Ensure last dimension is 3 for cross product
#     mask = torch.ones([2, 1746])                            # Example: create a dummy mask tensor
#     chain_M = torch.ones([2, 1746])                         # Example: create a dummy chain mask tensor
#     residue_idx = torch.ones([2, 1746], dtype=torch.long)   # Example: create a dummy residue index tensor
#     chain_encoding_all = torch.zeros([2, 1746])

#     print(f"X shape: {X.shape}")
#     print(f"S shape: {S.shape}")
#     print(f"mask shape: {mask.shape}")
#     print(f"chain_M shape: {chain_M.shape}")
#     print(f"residue_idx shape: {residue_idx.shape}")
#     print(f"chain_encoding_all shape: {chain_encoding_all.shape}")

#     # Perform the forward pass
#     output = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)

<<<<<<< HEAD
    print(f"Output shape: {output.shape}")
    print(output)
=======
#     print(f"Output shape: {output.shape}")
#     print(output)
>>>>>>> origin/LJ-branch
