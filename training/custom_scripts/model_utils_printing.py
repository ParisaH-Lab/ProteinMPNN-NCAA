#!/usr/bin/env python

from __future__ import print_function
import json, time, os, sys, glob
import shutil
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import torch.utils
import torch.utils.checkpoint

import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools

def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Debug print statements
    print(f"S shape: {S.shape}")
    print(f"S_onehot shape: {S_onehot.shape}")
    print(f"log_probs shape: {log_probs.shape}")

    if S_onehot.size() != log_probs.size():
        raise ValueError(f"Shape mismatch: S_onehot {S_onehot.size()} vs log_probs {log_probs.size()}")

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / 2000.0  # Adjusted for your specific case
    return loss, loss_av

if __name__ == "__main__":
    # Example usage within main block
    # Define some example tensors
    S = torch.randint(0, 21, (6, 1502))  # Example shape, adjust as per your actual data
    log_probs = torch.randn((6, 1502, 21))  # Example shape, adjust as per your actual data
    mask = torch.randint(0, 2, (6, 1502))  # Example shape, adjust as per your actual data

    # Call the function and print shapes
    loss, loss_av = loss_smoothed(S, log_probs, mask)
    print(f"Total loss shape: {loss.shape}")
    print(f"Average loss shape: {loss_av.shape}")
