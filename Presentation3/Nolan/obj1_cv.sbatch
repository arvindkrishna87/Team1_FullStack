#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=normal
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --job-name=obj1_cv
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user=nolanwhite2027@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

cd /home/chj4322/Presentation3/Presentation3_base_code
FOLD=$((SLURM_ARRAY_TASK_ID + 1))
SPLIT=no_train_val_leakage_splits/data_splits_new_0${FOLD}.npz

sleep $((SLURM_ARRAY_TASK_ID * 120))
echo "Fold $FOLD  split=$SPLIT  start $(date)"
python3 main.py \
    --labels_csv /home/chj4322/Presentation3/Presentation3_base_code/extra_benign_case_grade_match.csv \
    --patches_dir /home/chj4322/Presentation3/patch_images_93 \
    --analyze_attention \
    --attention_top_n 8 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --load_splits $SPLIT
echo "End $(date)"