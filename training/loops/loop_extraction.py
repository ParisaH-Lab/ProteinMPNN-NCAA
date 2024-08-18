#!/usr/bin/env python
#
# @Author: Andrew Powers (apowers@flatironinstitute.org & apowers4@uoregon.edu)
# @brief: Extract secondary structure information from PDBs and then pull out loop structures into a dataset
#

####
# Import 
####

# base packages
import os, sys, time
from collections import defaultdict
from functools import partial
import argparse
import re
from multiprocessing.pool import ThreadPool

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

    # ss track consecutive
    ss_track = 0
    # ss begin
    ss_previous = 0

    # init lists for information
    ss_ss_list = list()
    secondary_structure_idx = list()
    sub_ss_idx = list()
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

            # If this is a new loop section
            if (ele[1][1] != ss_previous + 1) & (ss_previous != 0):
                # If a ss is larger then 3 residues add it to sub_ss_idx
                if ss_track >= 3:
                    sub_ss_idx.append(ss_ss_list)
                # Reset Track
                ss_track = 0
                # Now clear and set up previous 
                ss_ss_list = list()

            # Update the ss previous residue
            ss_previous = ele[1][1]

            # ss track
            ss_track += 1
            ss_ss_list.append(ele[1][1])

            # Append SS idx for loops and phi/psi of those
            secondary_structure_idx.append(ele[1][1])
            phi_list.append(dict_out[4])
            psi_list.append(dict_out[5])

    # generate out pdb dict
    pdb_dict = {
        "ss_pdb": tmp,
        "sub_ss_list": sub_ss_idx,
        "ss_loop_index": secondary_structure_idx,
        "phi_loop": phi_list,
        "psi_loop": psi_list,
    }
    return pdb_dict

def extract_pdb(pdb_dict: dict, pdb_file: str, out_path: str):
    """Extract out loop regions into separate pdb_files

    PARAMS
    ------
    pdb_dict: Dict
        PDB secondary structure string, index, and phi, psi of the loop regions
    pdb_file: str
        Path to the pdb file the pdb_dict is from
    out_path: str
        Path to directory that the loops pdbs will be held in
    """
    # open file and generate dict of information
    file = open(pdb_file, 'r')
    file_dict = defaultdict(list)
    # Extract file name
    file_name = pdb_file.split('/')[-1].split(".pdb")[0]
    index_num = 0

    # Iter through PDB file and grab lines based on residue index
    for line in file:
        # Skip uninportant lines
        if (line.startswith("HEADER")) or (line.startswith("TER")) or (line.startswith("HET")):
            pass
        # Add everything else
        elif line.startswith("ATOM"):
            broken_line = re.sub("\s+", ",", line).split(",")
            # Since the PDBs have weird collision of columns we need these if else lines
            if ("." in broken_line[5]) & ("." in broken_line[4]):
                key = clean_key(broken_line[3])
            elif "." in broken_line[5]:
                key = clean_key(broken_line[4])
            else:
                key = int(broken_line[5])
            # Add the pdb lines to this particular dict 
            file_dict[key].append(line)


    # iter through loops
    for loops in pdb_dict["sub_ss_list"]:
        # Generate a pdb file
        out_file = open(os.path.join(
                        out_path,
                        f"{file_name}_{index_num}.pdb"), 'w')

        # write first line
        out_file.write("HEADER\n")

        # for resi in loops
        for resi in loops:
            out_file.write("".join(file_dict[resi]))

        # close particular file
        out_file.close()
        # increment idnex_num
        index_num += 1

    return 0

def clean_key(key:str):
    """Clean up the key, so that it is an int

    PARAMS
    ------
    key: str
        Possibly an int that has a letter in it.

    RETURNS
    -------
    clean_key: int
        An integar instead of a str
    """
    # we use re to remove an letters
    clean_key = re.sub('[A-Za-z]', '', key)
    return int(clean_key)

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

def extract_all_loops(pdb: str, out_loop_dir: str):
    """Function for passing to multithreading process

    PARAMS
    ------
    pdb: str
        PDB file path
    out_loop_dir: str
        Path to the out dir for loops
    """
    # Generate our out dict
    pdb_dict = dssp_label_residues(pdb)
    # Now generate the loop pdbs each given an input pdb
    extract_pdb(pdb_dict, pdb, out_loop_dir)
    return 0



if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Parameters for Loop Extraction and Dataset Creation.")
    p.add_argument("--path", type=str, help="Path to root direcotry of pdb directoires/files.")
    p.add_argument("--pdb", type=str, help="Single Path to PDB for testing")
    p.add_argument("--test", action="store_true", help="For testing with a single PDB")
    p.add_argument("--loop-out-dir", type=str, help="Path to directory (make if not made already) where loop pdbs are stored.")
    p.add_argument("--fix-pdb", action="store_true", help="Use this flag to convert my test data to work with DSSP")
    p.add_argument("--fix-out-dir", type=str, help="Path to directory (make if not made already) where fixed pdbs are stored")
    p.add_argument("--cpu-count", type=int, default=4, help="Number of CPU counts for multithreading (Default: 4)")
    args = p.parse_args()

    # If the pdbs needs to be fixed then this will run. 
    if args.fix_pdb:
        # Generate our file list first
        list_current = load_function(args.path)

        # Check if dir exists. If not then make it
        if not os.path.exists(args.fix_out_dir):
            os.mkdir(args.fix_out_dir)

        # Loop through and fix files
        for pdb_file in list_current:
            correct_pdb(pdb_file, args.fix_out_dir)

    # If I am testing the script or a problem PDB then this will run
    elif args.test:
        extract_loops = partial(extract_all_loops, out_loop_dir = args.loop_out_dir)
        extract_loops(args.pdb)

    # As long as the fix-pdb isn't True then this will run
    else:
        # Generate our file of pdbs first
        pdb_list = load_function(args.path)
        extract_loops = partial(extract_all_loops, out_loop_dir = args.loop_out_dir)
        # generate our loops
        with ThreadPool(args.cpu_count) as p:
            p.map(
                extract_loops,
                pdb_list
            )
