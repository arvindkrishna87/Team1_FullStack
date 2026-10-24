#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short                 # quick CPU job; no GPU or long queue needed
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1                        # single-process script
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G                          # headroom for loading PNGs + matplotlib
#SBATCH --job-name=replot
#SBATCH --output=replot_%j.out            # %j = job ID, so reruns don't overwrite logs
#SBATCH --error=replot_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nolanwhite2027@u.northwestern.edu                   # Your NU email

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

cd /projects/e32998/MIL_training/final_runs/run_20261022_115349
python3 -u /home/chj4322/Presentation1_base_code/replot.py .

echo "End time: $(date)"