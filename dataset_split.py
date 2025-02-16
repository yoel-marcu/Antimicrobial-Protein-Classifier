import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

# File paths for positive and negative FASTA files
pos_fasta = '/sci/labs/asafle/yoel.marcu2003/dataset/clustered_positive/clusterRes_positive_set_extend_rep_seq.fasta'
neg_fasta = '/sci/labs/asafle/yoel.marcu2003/dataset/clustered_negative/neg_set_clustering_rep_seq.fasta'

# Output directories
output_dir = '/sci/labs/asafle/yoel.marcu2003/dataset/splits'
os.makedirs(output_dir, exist_ok=True)

def read_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        seq = ""
        header = None
        for line in f:
            if line.startswith('>'):
                if header:
                    sequences.append((header, seq))
                header = line.strip()
                seq = ""
            else:
                seq += line.strip()
        if header:
            sequences.append((header, seq))
    return sequences

pos_sequences = read_fasta(pos_fasta)
neg_sequences = read_fasta(neg_fasta)

neg_sample_size = min(len(neg_sequences), len(pos_sequences) * 10)
neg_sequences_sampled = random.sample(neg_sequences, neg_sample_size)

pos_train, pos_test = train_test_split(pos_sequences, test_size=0.2, random_state=42)
pos_train, pos_val = train_test_split(pos_train, test_size=0.125, random_state=42)
neg_train, neg_test = train_test_split(neg_sequences_sampled, test_size=0.2, random_state=42)
neg_train, neg_val = train_test_split(neg_train, test_size=0.125, random_state=42)

def write_fasta(sequences, file_path):
    with open(file_path, 'w') as f:
        for header, seq in sequences:
            f.write(f"{header}\n{seq}\n")

for name, seqs in zip(['pos_train', 'pos_val', 'pos_test', 'neg_train', 'neg_val', 'neg_test'],
                      [pos_train, pos_val, pos_test, neg_train, neg_val, neg_test]):
    write_fasta(seqs, os.path.join(output_dir, f'{name}.fasta'))

with open(os.path.join(output_dir, 'dataset_summary.txt'), 'w') as summary:
    for name, seqs in zip(['pos_train', 'pos_val', 'pos_test', 'neg_train', 'neg_val', 'neg_test'],
                          [pos_train, pos_val, pos_test, neg_train, neg_val, neg_test]):
        lengths = np.array([len(seq) for _, seq in seqs])
        summary.write(f"{name}:\n")
        summary.write(f"  Count: {len(seqs)}\n")
        summary.write(f"  Avg Length: {np.mean(lengths) if lengths.size > 0 else 0:.2f}\n")
        summary.write(f"  Min Length: {np.min(lengths) if lengths.size > 0 else 0}\n")
        summary.write(f"  Max Length: {np.max(lengths) if lengths.size > 0 else 0}\n")
        summary.write(f"  Std Dev: {np.std(lengths) if lengths.size > 0 else 0:.2f}\n\n")

print("Dataset splitting and summary completed.")

