#!/usr/bin/env python

import torch
import numpy as np

class MyDatasetClass:
    def __init__(self, IDs, train_dict, loader, params):
        self.IDs = IDs
        self.train_dict = train_dict
        self.loader = loader
        self.params = params

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.train_dict[ID]))
        out = self.loader(self.train_dict[ID][sel_idx], self.params)
        
        # Debugging: Print the contents of 'out'
        print(f"Index: {index}, ID: {ID}, Selected Index: {sel_idx}, Output: {out}")

        # Check if 'out' is None or invalid
        if out is None:
            print(f"Warning: Loader returned None for index {index}, ID: {ID}")
            return None

        # If 'out' is empty or has other issues, print a warning
        if not out:  # This checks if 'out' is an empty list, dict, or other iterable
            print(f"Warning: Loader returned empty data for index {index}, ID: {ID}")
            return None

        return out

# Example loader function
def example_loader(data, params):
    # Simulates loading and processing the data
    return {"data": data, "params": params}

if __name__ == "__main__":
    # Example data to initialize the dataset
    IDs = ["id1", "id2", "id3"]
    train_dict = {
        "id1": [np.array([1, 2, 3])],
        "id2": [np.array([4, 5, 6])],
        "id3": [np.array([7, 8, 9])]
    }
    params = {"param1": 0.1, "param2": 0.2}

    # Create the dataset instance
    dataset = MyDatasetClass(IDs, train_dict, example_loader, params)

    # Test the __getitem__ method for a specific index
    print("Testing __getitem__ for index 0:")
    output = dataset.__getitem__(0)
    print(f"Output for index 0: {output}")

    print("Testing __getitem__ for index 1:")
    output = dataset.__getitem__(1)
    print(f"Output for index 1: {output}")