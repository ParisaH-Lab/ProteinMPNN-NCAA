#!/usr/bin/env python

###################
# Script Overview #
###################
# This is a custom script to mirror coordinates from L-chiral proteins to make complimentary D-chiral proteins.
# To Dos: Read through directory, load data strucuture, extract xyz coordinates, mirror them, load them into a new data structure
# Note: Both input and output files should be in a pdb format.

############
# Modules #
############
#import argparse
import os
import sys
import urllib.request
import Bio
import Bio.PDB
import Bio.SeqRecord

#############
# Functions #
#############
def read_pdb(pdbcode, pdbfilenm):
    """
    Read a PDB structure from a file.
    :param pdbcode: A PDB ID string
    :param pdbfilenm: The PDB file
    :return: a Bio.PDB.Structure object or None if something went wrong
    """
    try:
        pdbparser = Bio.PDB.PDBParser(QUIET=True)   # suppress PDBConstructionWarning
        struct = pdbparser.get_structure(pdbcode, pdbfilenm)
        return struct
    except Exception as err:
        print(str(err), file=sys.stderr)
        return None 