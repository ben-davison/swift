#!/bin/bash
#SBATCH --job-name=test_link
#SBATCH --output=test_link_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1      
#SBATCH --mem=4G                
#SBATCH --time=00:15:00

# 1. Load module and environment
module load Anaconda3/2024.02-1
source activate swift

# 2. Run the python script
python -u test_link.py

# 3. Upload the output to OSF
set -o allexport
source .env
set +o allexport

echo "Uploading to OSF..."
# Syntax: osf -p <project_id> upload <local_path> <osf_destination_name>
osf -p $OSF_PROJECT upload results/dummy_output.csv dummy_output.csv

echo "Upload complete!"
