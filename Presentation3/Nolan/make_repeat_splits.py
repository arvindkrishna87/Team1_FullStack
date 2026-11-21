"""Objective 2: build R reshuffled 5-fold rotations over the 93-case set.
 
Each repeat regroups the cases into 5 folds (patient families kept together,
benign / high-grade balanced in every fold), then writes 5 split files in the
same .npz format main.py loads with --load_splits:
    test = fold k, val = fold k-1, train = the other three folds.
Repeat 0 is the Objective 1 run, so this script makes repeats 1..9 by default.
"""
import argparse, os
import numpy as np
import pandas as pd
 
EXCLUDE = {8, 20, 33, 35, 62, 102}                     # no Team 2 embeddings
FAMILIES = [(22, 107, 108), (24, 118), (25, 119), (26, 109, 110, 111)]
 
ap = argparse.ArgumentParser()
ap.add_argument("--template", default="no_train_val_leakage_splits/data_splits_new_01.npz")
ap.add_argument("--labels_csv", default="extra_benign_case_grade_match.csv")
ap.add_argument("--out_dir", default="obj2_splits")
ap.add_argument("--first_repeat", type=int, default=1)
ap.add_argument("--last_repeat", type=int, default=9)
ap.add_argument("--k", type=int, default=5)
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
 
t = np.load(args.template, allow_pickle=True)
dtype = t["test_cases"].dtype
extra = set(t.files) - {"train_cases", "val_cases", "test_cases"}
if extra:
    print("WARNING: template has extra keys not written here:", extra)
cases = sorted({int(c) for k in ("train_cases", "val_cases", "test_cases") for c in t[k]} - EXCLUDE)
 
lab = pd.read_csv(args.labels_csv)
y = {int(c): (0 if float(k) == 1.0 else 1) for c, k in zip(lab.Case, lab.Class)}
missing = [c for c in cases if c not in y]
assert not missing, f"cases with no label: {missing}"
 
# Units that must stay together: a patient family, or a single case
fam_of = {c: f for f in FAMILIES for c in f}
units, seen = [], set()
for c in cases:
    if c in seen:
        continue
    u = tuple(x for x in fam_of.get(c, (c,)) if x in cases)
    assert len({y[x] for x in u}) == 1, f"family {u} mixes grades"
    seen |= set(u)
    units.append(u)
 
B, H = sum(y[c] == 0 for c in cases), sum(y[c] == 1 for c in cases)
print(f"{len(cases)} cases ({B} benign / {H} high-grade), {len(units)} units, k={args.k}")
 
for r in range(args.first_repeat, args.last_repeat + 1):
    rng = np.random.default_rng(1000 + r)
    folds = [[] for _ in range(args.k)]
    for cls in (0, 1):
        us = [u for u in units if y[u[0]] == cls]
        us = [us[i] for i in rng.permutation(len(us))]
        us.sort(key=len, reverse=True)                 # families first, random order otherwise
        for u in us:
            cnt = [sum(y[c] == cls for c in f) for f in folds]
            tot = [len(f) for f in folds]
            tie = rng.random(args.k)
            i = min(range(args.k), key=lambda j: (cnt[j], tot[j], tie[j]))
            folds[i] += u
 
    # sanity checks
    flat = [c for f in folds for c in f]
    assert sorted(flat) == cases, "every case must be in exactly one fold"
    for f in FAMILIES:
        present = [c for c in f if c in cases]
        assert len({i for i, fo in enumerate(folds) for c in present if c in fo}) <= 1
 
    desc = []
    for i in range(args.k):
        test = folds[i]
        val = folds[(i - 1) % args.k]
        train = [c for j in range(args.k) if j not in (i, (i - 1) % args.k) for c in folds[j]]
        assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
        np.savez(os.path.join(args.out_dir, f"r{r:02d}_f{i + 1}.npz"),
                 train_cases=np.array(sorted(train), dtype=dtype),
                 val_cases=np.array(sorted(val), dtype=dtype),
                 test_cases=np.array(sorted(test), dtype=dtype))
        b = sum(y[c] == 0 for c in test)
        desc.append(f"{len(test)}({b}B/{len(test) - b}H)")
    print(f"repeat {r:2d}: test folds " + "  ".join(desc))
print(f"Saved {args.k * (args.last_repeat - args.first_repeat + 1)} split files to {args.out_dir}/")