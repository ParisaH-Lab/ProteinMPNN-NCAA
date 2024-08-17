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
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.DSSP import dssp_dict_from_pdb_file


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

def dssp_label_residues(file: str):
    """Use DSSP within BioPandas to label secondary structures elements within a PDB

    PARAMS
    ------
    file: str
        Path to pdb file

    RETURNS
    -------
    pdb_dict: Dict
        PDB secondary structure string, index, and phi, psi of the loop regions
    """
    # init Parser
    _parser = PDBParser()

    # Extract pdb name
    structure_name = file.split("/")[-1].split(".pdb")[0]

    # Extract information from pdb
    structure = _parser.get_structure(structure_name, file)

    # Generate DSSP Output per input
    _dssp = DSSP(structure[0], file, dssp="mkdssp")

    # Residue key list
    residue_key = list(_dssp.keys())

    # Secondary structure string
    tmp = ''

    # init lists for information
    secondary_structure_idx = list()
    phi_list = list()
    psi_list = list()

    # Loop through the residues and grab important information
    for ele in residue_key:
        # Extract out dssp information for resiude ele
        dict_out = _dssp[ele]
        # Generate the secondary string
        tmp += dict_out[2]
        # Now we check if dict_out[2] is a loop character
        if dict_out[2] in ["T", "S", "-"]:
            # Append SS idx for loops and phi/psi of those
            secondary_structure_idx.append(ele[1][1])
            phi_list.append(dict_out[4])
            psi_list.append(dict_out[5])

    # generate out pdb dict
    pdb_dict = {
        "ss_pdb": tmp,
        "ss_loop_index": secondary_structure_idx,
        "phi_loop": phi_list,
        "psi_loop": psi_list,
    }
    return pdb_dict

def correct_pdb(pdb_file: str, dir_path: str):
    """Only needed to be run once. This is to fix the PDBs, so the DSSP
    runs correctly.

    PARAMS
    ------
    pdb_file: str
        PDB File path, so we cal load read/write
    dir_path: str
        The relative path for our output file
    """
    # open file to read
    open_pdb = open(pdb_file, 'r')
    # Generate file to write to
    file_name = pdb_file.split("/")[-1]
    out_file = open(
        os.path.join(dir_path, file_name),
        'w'
    )

    # write first line
    out_file.write("HEADER\n")

    # Loop through file lines
    for line in open_pdb:
        if not line.startswith("REMARK"):
            out_file.write(line)

    # Close files
    open_pdb.close()
    out_file.close()

    return 0



if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Parameters for Loop Extraction and Dataset Creation.")
    p.add_argument("--path", type=str, help="Path to root direcotry of pdb directoires/files.")
    p.add_argument("--pdb", type=str, help="Single Path to PDB for testing")
    p.add_argument("--fix-pdb", action="store_true", help="Use this flag to convert my test data to work with DSSP")
    p.add_argument("--fix-out-dir", type=str, help="Path to directory (make if not made already) where fixed pdbs are stored")
    args = p.parse_args()

    if args.fix_pdb:
        # Generate our file list first
        list_current = load_function(args.path)

        # Check if dir exists. If not then make it
        if not os.path.exists(args.fix_out_dir):
            os.mkdir(args.fix_out_dir)

        # Loop through and fix files
        for pdb_file in list_current:
            correct_pdb(pdb_file, args.fix_out_dir)

    else:
        pass
