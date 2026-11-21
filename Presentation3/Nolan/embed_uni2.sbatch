#!/bin/bash
# Usage:  sbatch embed_uni2.sbatch  (UNI2 weights must already be in ~/.cache/huggingface)
# Each array task encodes every 4th patch. Rerunning skips patches already saved.
#SBATCH --account=e32998
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --job-name=P2_embed_uni2
#SBATCH --output=/home/chj4322/Presentation3/Presentation3_base_code/logs/%x_%A_%a.out
#SBATCH --error=/home/chj4322/Presentation3/Presentation3_base_code/logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nolanwhite2027@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
export HF_HUB_OFFLINE=1

echo "Job ID: $SLURM_JOB_ID  Shard: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"
nvidia-smi -L

cd /home/chj4322/Presentation3/Presentation3_base_code

python3 precompute_pooled_features.py \
    --encoder uni2 \
    --patches_dir /projects/e32998/patches_benign_split \
    --embeddings_dir /home/chj4322/Presentation3/patch_embeddings_UNI2 \
    --batch_size 64 \
    --num_workers 6 \
    --num_shards 4 \
    --shard_index "$SLURM_ARRAY_TASK_ID" \
    --log_every 20

echo "End time: $(date)"
