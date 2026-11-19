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
SPLITS_DIR = ("/home/cwh6075/Team1_FullStack/Presentation3/Code1_PatchFeatures_20Nov_Grace/no_train_val_leakage_splits")

### GET LABELS
labels = load_labels(DATA_PATHS["labels_csv"])
labels = labels[labels["Class"].isin(VALID_CLASSES)]
grade_map = dict(zip(labels["Case"].astype(int), labels["Class"]))

### PATCH-LEVEL TABLE
records = []
built_at = datetime.now().isoformat(timespec="seconds")
# Test on only 100 files
#all_patches = sorted(
#    f for f in os.listdir(PATCHES_DIR)
#    if f.lower().endswith(".png"))
#test_patches = all_patches[:100]
#for filename in test_patches:
# Test on all files
for filename in sorted(os.listdir(PATCHES_DIR)):
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

    # Extract values from filename (case, patient, slice, and patch number)
    match = re.match(r"case_(\d+)_([a-z]+_?\d+)_.+_patch(\d+)\.png$",
                     filename,
                     flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unrecognized filename: {filename}")
    case_id = int(match.group(1))
    patient_map = {
        22: 22, 107: 22, 108: 22,
        26: 26, 109: 26, 110: 26, 111: 26,
        24: 24, 118: 24,
        25: 25, 119: 25}
    patient = patient_map.get(case_id, case_id)
    slice_raw = match.group(2).lower()
    patch_num = int(match.group(3))
    # slice_id = re.sub(r"([a-z])(\d+)$", r"\1_\2", slice_raw) # unused
    stain = extract_stain_from_filename(filename)
    if stain is None: raise ValueError(f"Unknown stain: {filename}")

    # Match current labels
    grade = grade_map[case_id]
    label = 0 if grade == 1 else 1

    # Get file path
    png_path = os.path.join(PATCHES_DIR, filename)
    expected_emb = os.path.join(EMBEDDINGS_DIR, os.path.splitext(filename)[0] + ".pt")
    if not os.path.isfile(expected_emb):
        print(f"Missing embedding: {expected_emb}")

    # Read original PNG dimensions
    with Image.open(png_path) as img:
        width, height = img.size

    # Embedding
    has_embedding = os.path.isfile(expected_emb)
    NEW_HIGH_GRADE_CASES = {8, 20, 33, 35, 62, 102}
    if case_id in NEW_HIGH_GRADE_CASES:
        emb_path = "new case, no embedding yet"
    elif os.path.isfile(expected_emb):
        emb_path = expected_emb
    else:
        emb_path = None

    # Put values together
    records.append({
        "patch_id": os.path.splitext(filename)[0],
        "case": case_id,
        "patient": patient,
        "slice_raw": slice_raw,
        #"slice_id": slice_id,
        "stain": stain,
        "patch": patch_num,
        "png_name": filename,
        "png_path": png_path,
        "emb_path": emb_path,
        #"exclude_reason": None if has_embedding else "missing_embedding", # deleted
        "grade_raw": grade,
        "label": label,
        #"fold": fold,
        "emb_version": "uni2",
        #"dataset": "current", # deleted
        "built_at": built_at,
        "width": width, # new
        "height": height, # new
        "patch_size": width * height # new
    })
# Get df and print
patch_df = pd.DataFrame(records)
if patch_df.empty:
    raise ValueError("No labeled PNG patches found")
patch_df.to_parquet("patches_with_features_test.parquet", index=False)
print("Patch table:", len(patch_df), " rows")

### CASE-LEVEL TABLE
case_records = []
for case_id, group in patch_df.groupby("case"):
    # Calculating new columns
    n_patches = len(group)
    n_slices = n_slices = group[["stain", "slice_raw"]].drop_duplicates().shape[0]
    sizes = group["patch_size"]
    stain_counts = group["stain"].value_counts()
    n_missing = group["emb_path"].isna().sum()

    # Put values together
    case_records.append({
        "case_id": case_id,
        "patient_id": group["patient"].iloc[0],
        "label": group["label"].iloc[0],
        "grade": group["grade_raw"].iloc[0],
        #"fold": fold,
        "n_patches": n_patches,
        "n_slices": n_slices,
        "stains": ", ".join(sorted(group["stain"].unique())),
        "n_missing_emb": int(n_missing),
        #"note": None, # deleted
        "patches_per_slice": n_patches / n_slices, # new
        "median_patch_size": sizes.median(), # new
        "patch_size_iqr": sizes.quantile(0.75) - sizes.quantile(0.25), # new
        "h&e_share": stain_counts.get("h&e", 0) / n_patches, # new
        "melan_share": stain_counts.get("melan", 0) / n_patches, # new
        "sox10_share": stain_counts.get("sox10", 0) / n_patches # new
    })
# Get df and print
case_df = pd.DataFrame(case_records).sort_values("case_id")
case_df.to_parquet("cases_with_features_test.parquet", index=False)
print("Case table:", len(case_df), " rows")