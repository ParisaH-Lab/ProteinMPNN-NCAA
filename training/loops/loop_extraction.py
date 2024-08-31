#!/usr/bin/env python
#
# @Author: Andrew Powers (apowers@flatironinstitute.org & apowers4@uoregon.edu)
# @brief: Extract secondary structure information from PDBs and then pull out loop structures into a dataset
#

####
# Import 
####

# base packages
import os, sys, time, random
from collections import defaultdict
from functools import partial
import argparse
import re
from multiprocessing.pool import ThreadPool
import warnings
from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed

# bio packages
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio import BiopythonWarning


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
        loop region infromation
        - PDB secondary structure (ss) string
        - Chiral label for loop
        - List of ss lists for pdb extraction
        - index of ss
        - phi
        - psi 
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
    aa_sub_list = list()
    aa_list = list()
    chiral_out_list = list()
    chiral_sub_list = list()
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
                    chiral_out_list.append(chiral_sub_list)
                    aa_list.append(aa_sub_list)
                # Reset Track
                ss_track = 0
                # Now clear and set up previous 
                ss_ss_list = list()
                chiral_sub_list = list()
                aa_sub_list = list()

            # Update the ss previous residue
            ss_previous = ele[1][1]

            # ss track
            ss_track += 1
            ss_ss_list.append(ele[1][1])
            aa_sub_list.append(dict_out[1]) # All upper still here (since model is all upper still)

            # Chiral determine labels
            if dict_out[4] > 0.0:
                chiral_sub_list.append('D')
            else:
                chiral_sub_list.append('L')

            # Append SS idx for loops and phi/psi of those
            secondary_structure_idx.append(ele[1][1])
            phi_list.append(str(dict_out[4]))
            psi_list.append(str(dict_out[5]))

    # generate out pdb dict
    pdb_dict = {
        "ss_pdb": tmp,
        "sub_ss_list": sub_ss_idx,
        "amino_acid_list": aa_list,
        "chiral_out": chiral_out_list,
        "ss_loop_index": secondary_structure_idx,
        "phi_loop": phi_list,
        "psi_loop": psi_list,
    }
    return pdb_dict

def flip_sign(match):
    """Convert match object from re and flip sign

    PARAMS
    ------
    match: re.object
        Matched pattern should be string that is actually a float value

    RETURNS
    -------
    float
    """
    value = -1.0 * float(match)
    if float(match) > 0:
        out = f"{value:.3f}"
    else:
        out = f"  {value:.3f}"
    return out
    # return f" {value:.3f}"

def extract_pdb(pdb_dict: dict, pdb_file: str, out_path: str, out_supp_dir: str):
    """Extract out loop regions into separate pdb_files

    PARAMS
    ------
    pdb_dict: Dict
        PDB secondary structure string, index, and phi, psi of the loop regions
    pdb_file: str
        Path to the pdb file the pdb_dict is from
    out_path: str
        Path to directory that the loops pdbs will be held in
    out_supp_dir: str
        Path to list.csv, valid, test.txt, chiral_out.csv
    """
    # open file and generate dict of information
    file = open(pdb_file, 'r')
    file_dict = defaultdict(list)
    # Extract file name
    file_name = pdb_file.split('/')[-1].split(".pdb")[0]
    chain = file_name[-1]
    index_num = 0

    # Iter through PDB file and grab lines based on residue index
    for line in file:
        # Skip uninportant lines
        if (line.startswith("HEADER")) or (line.startswith("TER")) or (line.startswith("HET")):
            pass
        elif line.startswith("ANISOU"):
            print("--------------------------")
            print("--------------------------")
            print("ANISOU WITHIN FILE SKIPPING: ", file_name)
            print("--------------------------")
            print("--------------------------")
            return 0
        # Add everything else
        elif line.startswith("ATOM"):
            broken_line = re.sub("\s+", ",", line).split(",")
            # Since the PDBs have weird collision of columns we need these if else lines
            if ("." in broken_line[5]) & ("." in broken_line[4]):
                key = clean_key(broken_line[3])
            elif "." in broken_line[5]:
                key = clean_key(broken_line[4])
            elif bool(re.search("[A-Z]", broken_line[5])):
                key = clean_key(broken_line[5])
            else:
                try:
                    key = int(broken_line[5])
                    # chain = broken_line[4]
                except:
                    print(broken_line)
                    print(file_name)
                    raise Exception("THIS FILE DIDNT WORK:", file_name, "\n", broken_line)
            # Add the pdb lines to this particular dict 
            file_dict[key].append(line)


    # iter through loops
    for num, loops in enumerate(pdb_dict["sub_ss_list"]):
        # Standar file
        pdb_file_name = f"{file_name}{num}"
        pdb_file_mirror_name = f"{file_name}mirror{num}"
        # Generate a pdb file
        out_file = open(os.path.join(
                        out_path,
                        f"{pdb_file_name}_{chain}.pdb"), 'w')
        out_mirror_file = open(os.path.join(
                        out_path,
                        f"{pdb_file_mirror_name}_{chain}.pdb"), 'w')

        # write first line
        out_file.write("HEADER\n")
        out_mirror_file.write("HEADER\n")

        # for resi in loops
        for resi in loops:
            out_file.write("".join(file_dict[resi]))

            # Regex pattern for capturing the x dimesnion coords
            pattern = r"(-?\d+\.\d+)(\s*\d*\.\d*\s*\d*\.\d*\s*[A-Z])"
            # write by flipping the sign of x
            out_mirror_file.write(
                "".join(
                    [
                        # re.sub(pattern, lambda m: m.group(1) + flip_sign(m), line) for line in file_dict[resi]
                        re.sub(pattern, lambda m: flip_sign(m.group(1)) + m.group(2), line) for line in file_dict[resi]
                    ]
                )
            )

        # close particular file
        out_file.close()
        out_mirror_file.close()

        # Generate supplementary standard files
        generate_supplementary_files(
            out_path = out_supp_dir,
            seq = "".join(pdb_dict["amino_acid_list"][num]),
            chiral_seq = "".join(pdb_dict["chiral_out"][num]),
            # chain = chain_list[num],
            chain = chain,
            file_name = pdb_file_name,
            hash = str(random.randint(100_000, 999_999)),
            cluster = str(random.randint(20_000, 30_000)),
            phi_list = pdb_dict["phi_loop"],
            psi_list = pdb_dict["psi_loop"],
        )

        # Generate supllementary mirror files
        generate_supplementary_files(
            out_path = out_supp_dir,
            # seq = "".join([x.lower() if x.isupper() else x.upper() for x in pdb_dict["amino_acid_list"][num]]),
            seq = "".join(pdb_dict["amino_acid_list"][num]),
            chiral_seq = "".join(["L" if i == "D" else "D" for i in pdb_dict["chiral_out"][num]]),
            # chain = chain_list[num],
            chain = chain,
            file_name = pdb_file_mirror_name,
            hash = str(random.randint(100_000, 999_999)),
            cluster = str(random.randint(20_000, 30_000)),
        )

    return 1

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

def generate_supplementary_files(
    out_path: str,
    seq: str,
    chiral_seq: str,
    chain: str,
    file_name: str,
    hash: int,
    cluster: int,
    phi_list: list = None,
    psi_list: list = None,
    ):
    """Generate MPNN supplementary list.csv, valid, and test .txt files
    Also, for our datasets chiral_out.csv

    PARAMS
    ------
    out_path: str
        The path to the output directory
    seq: str
        Sequence of the loop
    chiral_seq: str
        Sequence of chirality per residue
    chain: str
        Chain of the sequence
    file_name: str
        Name of the output file for the loop
    hash: int
        Hash that will be associated with the loop (6digit hash)
    cluster: int
        Cluster that will be associated with this loop (since we are not using seqid=30%)
    phi_list: list
        List of original non-mirror phi values
    psi_list: list
        List of original non-mirrror psi values
    """
    # generate data information that is need
    date = "2024-08-25"
    resolution = 1.0

    # seq and chirality change
    seq_converted = ''
    for i, s in enumerate(seq):
        if chiral_seq[i] == "D":
            seq_converted += s.lower()
        else:
            seq_converted += s.upper()

    # Now check if list/valid/test are generated (if one generated then all made)
    if os.path.exists(os.path.join(out_path, "list.csv")):
        # Append to the file
        list_out = open(os.path.join(out_path, "list.csv"), 'a')
        valid_out = open(os.path.join(out_path, "valid_clusters.txt"), 'a')
        test_out = open(os.path.join(out_path, "test_clusters.txt"), 'a')
        chiral_out = open(os.path.join(out_path, "chiral_out.csv"), 'a')
        metadata_out = open(os.path.join(out_path, "metadata.csv"), 'a')

    else:
        # Create the file
        list_out = open(os.path.join(out_path, "list.csv"), 'w')
        valid_out = open(os.path.join(out_path, "valid_clusters.txt"), 'w')
        test_out = open(os.path.join(out_path, "test_clusters.txt"), 'w')
        chiral_out = open(os.path.join(out_path, "chiral_out.csv"), 'w')
        metadata_out = open(os.path.join(out_path, "metadata.csv"), 'w')

        # Write out HEADER
        list_out.write("CHAINID,DEPOSITION,RESOLUTION,HASH,CLUSTER,SEQUENCE\n")
        chiral_out.write("CHAINID,CHIRALSEQ\n")
        metadata_out.write("CHAINID,PHI,PSI\n")

    # generate a list of important information
    list_line = f"{file_name}_{chain},{date},{resolution},{hash},{cluster},{seq_converted}\n"
    valid_line = f"{cluster}\n"
    test_line = f"{cluster}\n"
    chiral_line = f"{file_name}_{chain},{chiral_seq}\n"
    if (phi_list != None) or (psi_list != None):
        metadata_out.write(f"{file_name}_{chain},{'_'.join(phi_list)},{'_'.join(psi_list)}\n")

    # Write to the files
    list_out.write(list_line)
    chiral_out.write(chiral_line)

    # Determine if test or valid
    value = random.randint(1, 100)
    # This means it is a test or valid line
    if value <= 20:
        # These determine if it is valid or test
        if value <= 10:
            valid_out.write(valid_line)
        else:
            test_out.write(test_line)

    # Close all files
    list_out.close()
    valid_out.close()
    test_out.close() 
    chiral_out.close()
    metadata_out.close()

    return 0


def extract_all_loops(pdb: str, out_loop_dir: str, out_supp_dir: str):
    """Function for passing to multithreading process

    PARAMS
    ------
    pdb: str
        PDB file path
    out_loop_dir: str
        Path to the out dir for loops
    out_supp_dir: str
        Path to out dir for supplementary files
    """
    # Generate our out dict
    try:
        pdb_dict = dssp_label_residues(pdb)
    except Exception:
        print("-----------------------")
        print(pdb, " SKIPPING")
        print("-----------------------")
        return 0
    
    # try:
    # Now generate the loop pdbs each given an input pdb
    result_num = extract_pdb(pdb_dict, pdb, out_loop_dir, out_supp_dir)
    # except:
    #     # print the problem file
    #     print("---------------------")
    #     print("---------------------")
    #     print("---------------------")
    #     print("---------------------")
    #     print("---------------------")
    #     print(f"This file {pdb} gave an error!!!!")
    #     print("---------------------")
    #     print("---------------------")
    #     print("---------------------")
    #     print("---------------------")
    #     print("---------------------")
    return result_num



if __name__ == "__main__":
    warnings.simplefilter('ignore', BiopythonWarning)

    p = argparse.ArgumentParser(description="Parameters for Loop Extraction and Dataset Creation.")
    p.add_argument("--path", type=str, help="Path to root direcotry of pdb directoires/files.")
    p.add_argument("--pdb", type=str, help="Single Path to PDB for testing")
    p.add_argument("--test", action="store_true", help="For testing with a single PDB")
    p.add_argument("--loop-out-dir", type=str, help="Path to directory (make if not made already) where loop pdbs are stored.")
    p.add_argument("--supplementary-out-path", type=str, help="Path to directory for supplementary files")
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
        extract_loops = partial(extract_all_loops, out_loop_dir = args.loop_out_dir, out_supp_dir = "./")
        extract_loops(args.pdb)

    # As long as the fix-pdb isn't True then this will run
    else:
        # Generate our file of pdbs first
        pdb_list = load_function(args.path)
        extract_loops = partial(extract_all_loops, out_loop_dir = args.loop_out_dir, out_supp_dir = args.supplementary_out_path)
        # generate our loops
        # with ThreadPool(args.cpu_count) as p:
        #     p.map(
        #         extract_loops,
        #         pdb_list
        #     )

        with ThreadPool(args.cpu_count) as p:
            futures = list(tqdm(p.imap(extract_loops, pdb_list), total=len(pdb_list)))

        print("--------------------- SCRIPT FINISHED ---------------------")
        print("--------------------- REPORT OF RUN   ---------------------")
        print("Number of PDBs Converted: ", sum(futures), ", Percentage of Whole: ", sum(futures)/ len(pdb_list))
