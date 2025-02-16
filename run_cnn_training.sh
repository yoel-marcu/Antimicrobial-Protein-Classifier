#!/bin/bash
#SBATCH --job-name=cnn_ohe_gpu
#SBATCH --time=18:0:0
#SBATCH --mem=512G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --output=cnn_ohe_%j.out

# Load necessary modules
module load cuda/12.4.1
module load cudnn/9.1.0

# Activate virtual environment
source /sci/labs/asafle/yoel.marcu2003/myenv/bin/activate

# Ensure required Python packages are installed
pip install --upgrade pip
pip install numpy tensorflow scikit-learn matplotlib seaborn

# Run the CNN training script
python /sci/labs/asafle/yoel.marcu2003/PythonScripts/cnn_on_ohe.py

# Deactivate virtual environment (optional)
deactivate

