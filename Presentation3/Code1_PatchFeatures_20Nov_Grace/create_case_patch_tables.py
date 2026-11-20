import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
from config import DATA_PATHS, VALID_CLASSES
from data_utils import load_labels, extract_stain_from_filename
import time
import pyarrow

start = time.time()
processed = 0

### GET PATHS
PATCHES_DIR = DATA_PATHS["patches_dir"]
EMBEDDINGS_DIR = "/projects/e32998/patches_varsize_pooled4096"
#EMBEDDINGS_DIR = "/home/ele0436/STAT390_coder/same_input_comparison/embeddings_uni2_original_size"
SPLITS_DIR = ("/home/cwh6075/Team1_FullStack/Presentation3/Code1_PatchFeatures_20Nov_Grace/no_train_val_leakage_splits")

### GET LABELS
labels = load_labels(DATA_PATHS["labels_csv"])
labels = labels[labels["Class"].isin(VALID_CLASSES)]
grade_map = dict(zip(labels["Case"].astype(int), labels["Class"]))

### PATCH-LEVEL TABLE
patch_table = []
built_at = datetime.now().isoformat(timespec="seconds")
# Test on only 100 files
#all_patches = sorted(
#    f for f in os.listdir(PATCHES_DIR)
#    if f.lower().endswith(".png"))
#test_patches = all_patches[:100]
#for filename in test_patches:
# Test on all files
for filename in sorted(os.listdir(PATCHES_DIR)):
    # Time
    processed = processed + 1
    if processed % 1000 == 0:
        elapsed = time.time() - start
        rate = processed / elapsed
        print(f"Processed {processed:,} patches "
              f"in {elapsed / 60:.1f} minutes"
              f"({rate:.1f} patches/sec)",
              flush = True)
    if not filename.lower().endswith(".png"):
        continue

    # Filename values (patch_id, case_id, patient, slice, and patch number)
    patch_id = os.path.splitext(filename)[0]
    match = re.match(r"case_(\d+)_([a-z]+_?\d+)_.+_patch(\d+)\.png$",
                     filename,
                     flags=re.IGNORECASE)
    case_id = int(match.group(1))
    patient_map = {
        22: 22, 107: 22, 108: 22,
        26: 26, 109: 26, 110: 26, 111: 26,
        24: 24, 118: 24,
        25: 25, 119: 25}
    patient_id = patient_map.get(case_id, case_id)
    slice_raw = match.group(2).lower()
    patch_num = int(match.group(3))
    stain = extract_stain_from_filename(filename)

    # Get file and embedding paths
    png_path = os.path.join(PATCHES_DIR, filename)
    expected_emb = os.path.join(EMBEDDINGS_DIR, os.path.splitext(filename)[0] + ".pt")
    if not os.path.isfile(expected_emb):
        print(f"Missing embedding: {expected_emb}")
    has_embedding = os.path.isfile(expected_emb)
    NEW_HIGH_GRADE_CASES = {8, 20, 33, 35, 62, 102}
    if case_id in NEW_HIGH_GRADE_CASES:
        emb_path = "N/A (new case)"
    elif os.path.isfile(expected_emb):
        emb_path = expected_emb
    else:
        emb_path = None

    # Match current labels
    if case_id not in grade_map:
        print(f"Case {case_id} not found in grade_map (low-grade)")
        continue
    grade = grade_map[case_id]
    label = 0 if grade == 1 else 1

    # PNG dimensions
    with Image.open(png_path) as img:
        width, height = img.size
    patch_size = width * height

    # Put values together
    patch_table.append({
        "patch_full": patch_id,
        "case": case_id,
        "patient": patient_id,
        "slice_raw": slice_raw,
        "stain": stain,
        "patch": patch_num,
        "png_name": filename,
        "png_path": png_path,
        "emb_path": emb_path,
        "grade_raw": grade,
        "label": label,
        "emb_version": "pooled4096",
        "built_at": built_at,
        "width": width, # new
        "height": height, # new
        "patch_size": patch_size # new
    })
# Get df and print
patch_df = pd.DataFrame(patch_table)
patch_df.to_parquet("patches_with_features.parquet", index=False)
print("Patch table:", len(patch_df), " rows")

### CASE-LEVEL TABLE
case_table = []
for case_id, group in patch_df.groupby("case"):
    # Patient, label, grade
    patient_id = group["patient"].iloc[0]
    label = group["label"].iloc[0]
    grade = group["grade_raw"].iloc[0]

    # Counts and sizes
    n_patches = len(group)
    n_slices = group[["stain", "slice_raw"]].drop_duplicates().shape[0]
    n_stains = group["stain"].nunique()
    stains = ", ".join(sorted(group["stain"].unique()))
    n_missing = int(group["emb_path"].isna().sum())
    missing_emb_share = n_missing / n_patches
    patches_per_slice = n_patches / n_slices

    # Sizes
    sizes = group["patch_size"]
    stain_counts = group["stain"].value_counts()
    median_patch_size = sizes.median()
    mean_patch_size = sizes.mean()
    std_patch_size = sizes.std()
    min_patch_size = sizes.min()
    max_patch_size = sizes.max()
    patch_size_iqr = sizes.quantile(0.75) - sizes.quantile(0.25)
    slice_counts = group.groupby(["stain", "slice_raw"]).size()
    median_patches_per_slice = slice_counts.median()
    std_patches_per_slice = slice_counts.std()
    max_patches_per_slice = slice_counts.max()

    # Shares
    h_e_share = stain_counts.get("h&e", 0) / n_patches
    melan_share = stain_counts.get("melan", 0) / n_patches
    sox10_share = stain_counts.get("sox10", 0) / n_patches
    small_patch_share = (((group["width"] < 32) | (group["height"] < 32)).mean())

    # Put values together
    case_table.append({
        "case_id": case_id,
        "patient_id": patient_id,
        "label": label,
        "grade": grade,
        "n_patches": n_patches,
        "n_slices": n_slices,
        "n_stains": n_stains, # new
        "stains": stains,
        "n_missing_emb": n_missing,
        "missing_emb_share": missing_emb_share, # new
        "patches_per_slice": patches_per_slice, # new
        "median_patch_size": median_patch_size, # new
        "mean_patch_size": mean_patch_size, # new
        "std_patch_size": std_patch_size, # new
        "min_patch_size": min_patch_size, # new
        "max_patch_size": max_patch_size, # new
        "patch_size_iqr": patch_size_iqr, # new
        "small_patch_share": small_patch_share, # new
        "median_patches_per_slice": median_patches_per_slice, # new
         "std_patches_per_slice": std_patches_per_slice, # new
        "max_patches_per_slice": max_patches_per_slice, # new
        "h_e_share": h_e_share, # new
        "melan_share": melan_share, # new
        "sox10_share": sox10_share # new
    })
# Get df and print
case_df = pd.DataFrame(case_table).sort_values("case_id")
case_df.to_parquet("cases_with_features.parquet", index=False)
print("Case table:", len(case_df), " rows")