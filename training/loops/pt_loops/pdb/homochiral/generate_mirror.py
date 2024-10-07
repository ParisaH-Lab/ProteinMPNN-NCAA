#!/usr/bin/env python

# IMPORT
import torch
import re
import numpy as np
import os, sys, glob
import argparse
import random
from tqdm import tqdm
from typing import Dict, List, Tuple
from collections import defaultdict

###### Functions
def read_list(f: str) -> Dict[str, Tuple[str,str,str]]:
    """Create a file: hash pair dictionary that will be used for adding on the new entries to
    the list.csv file

    PARAMETERS
    ----------
    f: str
        This will be a file str that needs to be read into memory or parsed still

    RETURNS
    -------
    file_hash_dict: Dict[str, Tuple(str,str,str)]
        Dictionary that holds the file as key and the new hash as the value
    """
    # init dict
    file_hash_dict = {}
    old_to_new = defaultdict(str)
    # In same location open test and valid files
    # valid_file = open(re.sub("list.csv", "valid_clusters.txt", f), 'a')
    # test_file = open(re.sub("list.csv", "test_clusters.txt", f), 'a')
    # open file
    with open(f, 'r') as f:
        # get rid of the heads
        f.readline()
        # now sequentially read in the files
        for line in f:
            if line.strip() == "":
                break
            # split line by commas
            f_split = line.split(',')
            # grab important elements
            file, hash_num, seq = f_split[0].split('/')[-1], f_split[3], f_split[-1]
            # If file has no hash_num then generate hash and cluster
            if old_to_new[hash_num] == '':
                old_to_new[hash_num] = (
                    "".join(
                        random.choice(
                            '0123456789'
                        ) for _ in range(15)
                    ),
                    "".join(
                        random.choice(
                            '0123456789'
                        ) for _ in range(15)
                    )
                )
            # Write to test or validate
            # 20% of the time write to test or validate
            # if np.random.choice([0,1], p=[0.8, 0.2], size=1):
            #     if np.random.choice([0,1], p=[0.5,0.5], size=1):
            #         print(old_to_new[hash_num][1], file=valid_file)
            #     else:
            #         print(old_to_new[hash_num][1], file=test_file)
            # # Add information to dictionary
            # file_hash_dict[file] = (old_to_new[hash_num][0],
            #                         old_to_new[hash_num][1],
            #                         seq)

    # close files
    # test_file.close()
    # valid_file.close()
    return file_hash_dict

def load_data(f:object, list_f: object, out:str, valid_file: str, test_file: str):
    """Load in .pt file Check if it is _{letter} or not type file. Send to appropriate functions

    PARAMETERS
    ----------
    f: object

    RETURNS
    -------
    """
    f_split = (f.split('/')[-1])
    # File specific check list
    true_check = re.search(r'_[A-Z]+\.pt', f_split)
    if true_check:
        chain_file(f, list_f, out, valid_file, test_file)
    else:
        non_chain_file(f, out)
    return 0

def non_chain_file(f: str, out:str):
    # grab file name
    f_new = re.sub(".pt$", "", (f.split('/')[-1]))

    # f_new _rev
    f_new = f_new + "mirror.pt"

    # load in .pt
    data = torch.load(f)

    # grab seq
    seq = data['seq']

    # change sequences
    for i, s in enumerate(seq):
        for n, k in enumerate(s):
            seq[i][n] = gen_seq(k)

    # Add back to pt
    data['seq'] = seq

    torch.save(data, os.path.join(out, f_new))
    return 0

def chain_file(f: str, list_f: object, out:str, valid_f: object, test_f: object):
    # Grab the name
    f_rm = re.sub(".pt$", "", (f.split('/')[-1]))

    # f_new _rev
    f_new = f_rm.split('_')[0] + "mirror_" + f_rm.split('_')[1] + '.pt'

    # load in .pt
    data = torch.load(f)

    # grab sequence and xyz from _A file
    seq = data['seq']
    xyz = data['xyz']

    if (np.isnan(xyz).all()).item():
        print(f'* -------------- Skipping {f} -------------- *')
        return 0

    # mirror image
    inv_seq = gen_seq(seq)
    chiral_xyz = xyz[:, 0] * -1.0

    # reassign data
    data['xyz'] = chiral_xyz
    data['seq'] = inv_seq

    # Now output as rev file to data
    torch.save(data, os.path.join(out, f_new))

    # write to list file
    write_list(list_f, f_new, f_rm, test_f, valid_f, inv_seq)

    return 0

def write_list(f: object, pt_file_new: str, pt_file: str, test_f:object, valid_f:object, sequence: str) -> None:
    """Write out new .pt rev file to list.csv file

    PARAMETERS
    ----------
    f: object
        opened file object to append to

    pt_file: str
        New pt file name that will be added to the file
    """
    test_file = open(test_f, 'a')
    valid_file = open(valid_f, 'a')
    hash = "".join(
        random.choice(
            '0123456789'
        ) for _ in range(10)
    )
    cluster = "".join(
        random.choice(
            '0123456789'
        ) for _ in range(10)
    )
    
    new_file = re.sub(".pt", "", pt_file_new)
    try:
        # out_line = f"{new_file},2024-07-10,0.0,{file_hash_dict[pt_file][0]},{file_hash_dict[pt_file][1]},{gen_seq(file_hash_dict[pt_file][2])}"
        out_line = f"{new_file},2024-07-10,0.0,{hash},{cluster},{sequence}"
        f.write(out_line+"\n")
        if np.random.choice([0,1], p=[0.8, 0.2], size=1):
            if np.random.choice([0,1], p=[0.5,0.5], size=1):
                print(cluster, file=valid_file)
            else:
                print(cluster, file=test_file)
    except:
        print('Skipping File: ', pt_file)
    test_file.close()
    valid_file.close()
    return 0

def gen_seq(seq: str)-> list:
    """Take in a column of sequences and return the lower/upper inverse
    sequences

    PARAMETERS
    ----------
    seq: str
        string of cyclic peptide sequences
        Ex (aAWFNPDg)

    RETURNS
    -------
    inverse_seq: str
        string of inverse cyclic peptide sequences
        compared to the input.
        Ex (AawfnpdG)
    """
    # create inverse seq
    inverse_seq = ''.join([x.upper() if x.islower() else x.lower() for x in seq])

    # get rid of artifact
    inverse_seq = inverse_seq.replace('g', "G")

    return inverse_seq

def main():

    p = argparse.ArgumentParser()
    p.add_argument("--input-pt-dir", help="Path to pt file paths (default: None)", type=str, required=True)
    p.add_argument("--output-dir", help="Where to put output data (default: None)",
                   type=str, required=True)
    p.add_argument("--metadata-path", help="Path to list.csv, valid_ & test_clusters.txt", type=str,
                   required=True)
    args = p.parse_args()

    # extract global variables
    INPUT = args.input_pt_dir
    OUTPUT = args.output_dir
    METAPATH = args.metadata_path

    # list file location
    list_file = os.path.join(METAPATH,"list.csv")

    # file for appending to list.csv
    append_list = open(list_file, 'a')

    # glob the .pt files
    pt_files = glob.glob(os.path.join(INPUT, "*.pt"))

    for file in tqdm(pt_files):
        load_data(file, append_list, OUTPUT,
                  os.path.join(METAPATH, "valid_clusters.txt"),
                  os.path.join(METAPATH, "test_clusters.txt"),
                  )

    # close file
    append_list.close()
    return 0


if __name__ == "__main__":
    main()
