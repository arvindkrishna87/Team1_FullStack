# Counts how many patch PNGs are smaller than 32x32 px.
# These are the patches skipped by precompute_pooled_features.py (MIN_PATCH_SIZE = 32).
# UPDATE BEFORE RUNNING: PATCH_DIR, if the patch folder moves.
import os
from PIL import Image

PATCH_DIR = "/projects/e32998/patches_benign_split"
MIN_SIZE = 32

files = [f for f in os.listdir(PATCH_DIR) if f.lower().endswith(".png")]
too_small = 0
for f in files:
    w, h = Image.open(os.path.join(PATCH_DIR, f)).size  # reads header only
    if w < MIN_SIZE or h < MIN_SIZE:
        too_small += 1

print(f"total={len(files)}  too_small={too_small}  pct={100*too_small/len(files):.2f}%")
