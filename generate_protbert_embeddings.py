from transformers import BertTokenizer, BertModel
import torch
import os

# Load ProtBERT tokenizer and model
print("Loading ProtBERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained("/sci/labs/asafle/yoel.marcu2003/hf_cache/prot_bert_local")
model = BertModel.from_pretrained("/sci/labs/asafle/yoel.marcu2003/hf_cache/prot_bert_local")
print("Model loaded successfully!")

# Function to read sequences from FASTA file
def read_fasta(file_path):
    print(f"Reading sequences from {file_path}...")
    sequences = []
    with open(file_path, 'r') as f:
        seq = ""
        for line in f:
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line.strip()
        if seq:
            sequences.append(seq)
    print(f"Loaded {len(sequences)} sequences from {file_path}.")
    return sequences

# Function to generate ProtBERT embeddings
def generate_embeddings(sequences, output_file):
    print(f"Generating embeddings for {len(sequences)} sequences...")
    embeddings = []
    for i, seq in enumerate(sequences):
        print(f"Processing sequence {i+1}/{len(sequences)} (length: {len(seq)})")
        inputs = tokenizer(" ".join(list(seq)), return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        embeddings.append(embedding.cpu().numpy())

    # Save embeddings
    torch.save(embeddings, output_file)
    print(f"Embeddings saved at {output_file}.")

# Directory paths
input_dir = '/sci/labs/asafle/yoel.marcu2003/dataset/splits'
output_dir = '/sci/labs/asafle/yoel.marcu2003/dataset/embeddings'
os.makedirs(output_dir, exist_ok=True)

# Generate embeddings for each dataset split
for file in os.listdir(input_dir):
    if file.endswith('.fasta') and file not in ('pos_test.fasta','pos_val.fasta'):
        print(f"Processing file: {file}")
        sequences = read_fasta(os.path.join(input_dir, file))
        output_file = os.path.join(output_dir, f"{file.replace('.fasta', '_protbert.pt')}")
        generate_embeddings(sequences, output_file)

print("All embeddings generated successfully!")

