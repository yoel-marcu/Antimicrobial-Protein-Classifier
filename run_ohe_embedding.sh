#!/bin/bash
#SBATCH --job-name=ohe_embeddings
#SBATCH --time=2:0:0
#SBATCH --mem=512G
#SBATCH --cpus-per-task=4
#SBATCH --output=ohe_embeddings_%j.out

source /sci/labs/asafle/yoel.marcu2003/myenv/bin/activate
pip install numpy tqdm

cd /sci/labs/asafle/yoel.marcu2003/PythonScripts
python generate_ohe_embeddings.py
deactivate

