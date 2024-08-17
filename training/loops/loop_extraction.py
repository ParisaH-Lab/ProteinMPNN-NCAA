#!/usr/bin/env python
#
# @Author: Andrew Powers (apowers@flatironinstitute.org & apowers4@uoregon.edu)
# @brief: Extract secondary structure information from PDBs and then pull out loop structures into a dataset
#

####
# Import 
####

# base packages
import os, sys
from collections import defaultdict
import argparse
# import mpi

# bio packages
import biopandas as biopd


######
# Functions
######

def load_function(path: str):
    """Take in a directory path to our pdb data. Then generate a list of paths
    to these files in a list

    PARAMS
    ------
    path: str
        Dir path that holds all of our pdbs

    RETURNS
    -------
    pdb_path: list[str]
        A list of pdb relative paths from the script to pdbs
    """
    # Init the output pdb list
    pdb_path = list()

    # generate a dict of subdirectories within the path
    dir_dict = {x[0]: x[2] for x in os.walk(path)}

    # Iter through dictionary of key, value pairs
    for k in dir_dict.keys():
        # Generate a list of correct paths
        out = [os.path.join(k, out) for out in dir_dict[k]]
        # extend the initial list and add all
        pdb_path.extend(out)

    return pdb_path
        


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Parameters for Loop Extraction and Dataset Creation.")
    p.add_argument("--path", type=str, help="Path to root direcotry of pdb directoires/files.")
    args = p.parse_args()

    print(load_function(path=args.path))
