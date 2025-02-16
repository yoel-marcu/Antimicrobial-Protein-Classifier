import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Define input and output directories
DATASET_DIR = "/sci/labs/asafle/yoel.marcu2003/dataset/splits"
OUTPUT_DIR = "/sci/labs/asafle/yoel.marcu2003/dataset/ohe_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Amino acid dictionary for one-hot encoding
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
SEQ_LENGTH = 150  # Max sequence length (assuming sequences are in range 50-150)

# Function to one-hot encode sequences
def one_hot_encode(seq):
    ohe = np.zeros((SEQ_LENGTH, len(AMINO_ACIDS)), dtype=np.float32)
    for i, aa in enumerate(seq[:SEQ_LENGTH]):  # Truncate longer sequences
        if aa in AA_TO_INDEX:
            ohe[i, AA_TO_INDEX[aa]] = 1
    return ohe

# Process each dataset split
splits = ["pos_train.fasta", "pos_val.fasta", "pos_test.fasta",
          "neg_train.fasta", "neg_val.fasta", "neg_test.fasta"]

for split in splits:
    fasta_path = os.path.join(DATASET_DIR, split)
    output_file = os.path.join(OUTPUT_DIR, split.replace(".fasta", "_ohe.npy"))

    print(f"Processing {split} -> {output_file}")

    sequences = []
    with open(fasta_path, "r") as f:
        lines = f.readlines()

    # Extract sequences from FASTA
    for i in range(0, len(lines), 2):  # FASTA format: header + sequence
        sequence = lines[i+1].strip()
        sequences.append(one_hot_encode(sequence))

    # Save OHE as numpy array
    np.save(output_file, np.array(sequences))
    print(f"Saved {output_file} ({len(sequences)} sequences)")
