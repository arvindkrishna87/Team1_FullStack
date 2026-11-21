#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=1G
#SBATCH --job-name=make_splits_93
#SBATCH --output=/home/chj4322/Presentation3/Presentation3_base_code/logs/%x_%j.out
#SBATCH --error=/home/chj4322/Presentation3/Presentation3_base_code/logs/%x_%j.err
#SBATCH --mail-user=nolanwhite2027@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
cd /home/chj4322/Presentation3/Presentation3_base_code
mkdir -p splits_93

python3 make_splits.py \
    --labels_csv /home/chj4322/Presentation3/Presentation3_base_code/labels_93.csv \
    --patches_dir /projects/e32998/patch_images \
    --save_dir /home/chj4322/Presentation3/Presentation3_base_code/splits_93

echo "End: $(date)"