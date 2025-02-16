#!/bin/bash
#SBATCH --job-name=dataset_split
#SBATCH --time=2:0:0              # Set maximum runtime
#SBATCH --mem=16G                 # Memory requirement
#SBATCH --cpus-per-task=4         # Number of CPUs
#SBATCH --output=dataset_split_%j.out   # Output log file with job ID

# Activate the virtual environment if necessary
source /sci/labs/asafle/yoel.marcu2003/myenv/bin/activate

# Navigate to the directory with the script
cd /sci/labs/asafle/yoel.marcu2003/PythonScripts

# Run the dataset split Python script
python dataset_split.py

# Deactivate the environment (optional)
deactivate

# End of script

