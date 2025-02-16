#!/bin/bash
#SBATCH --job-name=protbert_embeddings
#SBATCH --time=24:0:0              # Set maximum runtime
#SBATCH --mem=512G                 # Memory requirement
#SBATCH --cpus-per-task=8         # Number of CPUs
#SBATCH --output=protbert_embeddings_%j.out  # Save log file
#SBATCH --error=protbert_embeddings_%j.err   # Save error logs


# Activate the virtual environment
source /sci/labs/asafle/yoel.marcu2003/myenv/bin/activate
export TMPDIR=/sci/labs/asafle/yoel.marcu2003/lab_tmp
export PIP_CACHE_DIR=/sci/labs/asafle/yoel.marcu2003/pip_cache
export TRANSFORMERS_CACHE=/sci/labs/asafle/yoel.marcu2003/hf_cache


# Install required Python packages
#pip install torch transformers tqdm

# Navigate to the Python scripts directory
cd /sci/labs/asafle/yoel.marcu2003/PythonScripts

# Run the ProtBERT embedding generation script
python generate_protbert_embeddings.py

# Deactivate the virtual environment
deactivate

# End of script

