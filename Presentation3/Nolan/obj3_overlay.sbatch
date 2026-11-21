#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=short
#SBATCH --array=0-1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=obj3_overlay
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user=nolanwhite2027@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
python3 -c "import cv2" 2>/dev/null || python3 -m pip install --user opencv-python-headless
python3 -c "import tifffile" 2>/dev/null || python3 -m pip install --user tifffile
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

CASES=(18 46)
RUNS=(r01_f4 r01_f2)
CASE=${CASES[$SLURM_ARRAY_TASK_ID]}
RUN=/projects/e32998/MIL_training/final_runs/obj2_chj4322/${RUNS[$SLURM_ARRAY_TASK_ID]}

cd /home/chj4322/Presentation3/Presentation3_base_code/patch_viz_on_slice/sift_method
echo "Case $CASE  run $RUN  start $(date)"
python3 top_patches_overlay.py --run-dir "$RUN" --cases $CASE --k 5 \
    --csv-name top_patches_with_filename.csv \
    --slice-root /home/chj4322/Presentation3/slices \
    --out /home/chj4322/Presentation3/Presentation3_base_code/obj3_overlays/case_$CASE
echo "End $(date)"