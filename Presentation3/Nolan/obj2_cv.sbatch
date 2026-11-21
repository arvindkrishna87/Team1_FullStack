#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=normal
#SBATCH --array=0-44
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=8G
#SBATCH --job-name=obj2_cv
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user=nolanwhite2027@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

cd /home/chj4322/Presentation3/Presentation3_base_code
R=$(( SLURM_ARRAY_TASK_ID / 5 + 1 ))        # repeats 1-9 (repeat 0 = Objective 1)
F=$(( SLURM_ARRAY_TASK_ID % 5 + 1 ))        # folds 1-5
TAG=$(printf "r%02d_f%d" $R $F)
SPLIT=obj2_splits/${TAG}.npz

echo "Task $SLURM_ARRAY_TASK_ID  tag=$TAG  split=$SPLIT  start $(date)"
python3 run_obj2.py --split $SPLIT --tag $TAG \
    --labels_csv /home/chj4322/Presentation3/Presentation3_base_code/extra_benign_case_grade_match.csv \
    --patches_dir /home/chj4322/Presentation3/patch_images_93 \
    --analyze_attention \
    --attention_top_n 8 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5
echo "End $(date)"