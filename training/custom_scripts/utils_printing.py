#!/usr/bin/env python

import torch
import numpy as np
import time

def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=10):
    pdb_dict_list = []
    t0 = time.time()

    for step, t in enumerate(data_loader):
        print(f"Data at step {step}: {t}")  # Add this line to check the structure of t

        for idx in range(repeat):
            if not isinstance(t, dict):
                print(f"Skipping idx {step} because t is not a dictionary.")
                continue

            if 'masked' not in t or 'seq' not in t or 'xyz' not in t or 'mask' not in t or 'label' not in t:
                print(f"Skipping idx {step} due to missing keys in t.")
                continue

            res = torch.nonzero(t['masked'][0, :].cpu().numpy() == 0).view(-1)
            print(f"idx: {step}, res shape: {res.shape}")
            if len(res) < 200 or res.shape[0] < 1:
                print(f"Skipping idx {step} due to insufficient length")
                continue

            initial_sequence = "".join(map(str, t['seq'][0, res].tolist()))
            if len(initial_sequence) > max_length:
                print(f"Skipping idx {step} due to sequence length {len(initial_sequence)}")
                continue

            pdb_dict_list.append({
                'seq': initial_sequence,
                'xyz': t['xyz'][0, res, :].cpu().numpy(),
                'mask': t['mask'][0, res].cpu().numpy(),
                'label': t['label'][0, res].cpu().numpy()
            })

            if len(pdb_dict_list) >= num_units:
                break

    print(f"Generated pdb_dict_list with length: {len(pdb_dict_list)}")
    return pdb_dict_list