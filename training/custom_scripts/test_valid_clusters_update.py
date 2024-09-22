#!/usr/bin/env python

def update_clusters(input_file, output_file, prefix="mirror_"):
    """
    Updates cluster numbers by prepending a prefix to each cluster number.
    Writes the updated clusters to a new file.
    """
    with open(input_file, 'r') as file:
        clusters = file.read().splitlines()
    
    # # Prepend the specified prefix to each cluster number
    # mirrored_clusters = [prefix + cluster for cluster in clusters]
    # Add the specified increment to each cluster number
    mirrored_clusters = [str(int(cluster) + 1000) for cluster in clusters]

    with open(output_file, 'w') as file:
        for cluster in mirrored_clusters:
            file.write(f"{cluster}\n")

# Paths to the original and mirrored test cluster files
input_test_clusters = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/combined_labeled_pdb_2021aug02_sample2/test_clusters.txt'
output_test_clusters_mirror = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/combined_labeled_pdb_2021aug02_sample2/test_clusters_mirror.txt'

# Paths to the original and mirrored validation cluster files
input_valid_clusters = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/combined_labeled_pdb_2021aug02_sample2/valid_clusters.txt'
output_valid_clusters_mirror = '/projects/parisahlab/lmjone/internship/ProteinMPNN-PH/training/datasets/combined_labeled_pdb_2021aug02_sample2/valid_clusters_mirror.txt'

# Call the function to update the test and validation cluster files
update_clusters(input_test_clusters, output_test_clusters_mirror)
update_clusters(input_valid_clusters, output_valid_clusters_mirror)