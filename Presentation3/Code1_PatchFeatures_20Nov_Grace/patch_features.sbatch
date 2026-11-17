#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks=16                        # n nores
#SBATCH --mem=128G                          # n GB RAM
#SBATCH --job-name=patch_features
#SBATCH --output=patch_features_output.out
#SBATCH --error=patch_features_error.err
#SBATCH --mail-type=ALL                   # Send email on begin, end, and fail
#SBATCH --mail-user=gracefan2027@u.northwestern.edu  # Your NU email

# Load environment (same as your previous jobs)
module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env  # you might want to copy the env and put it in your new project directory, if different from e32998

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run from the directory containing this sbatch file
cd "$SLURM_SUBMIT_DIR"

# Run patch feature script
python3 create_case_patch_tables.py

echo "End time: $(date)"
