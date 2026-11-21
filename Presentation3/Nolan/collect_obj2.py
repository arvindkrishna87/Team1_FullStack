"""Objective 2: count how often each case is misclassified across R repeats.

Repeat 0 = Objective 1 (obj1_all_predictions.csv); repeats 1-9 = obj2 runs.
Writes obj2_long.csv (one row per case per repeat) and obj2_wrong_counts.csv.
"""
import glob, os, re, sys
import pandas as pd

ROOT = "/projects/e32998/MIL_training/final_runs/obj2_chj4322"
HARD_MIN = 6                                   # wrong in >= 6 of 10 repeats -> hard

o1 = pd.read_csv("obj1_all_predictions.csv")
o1["repeat"] = 0
frames, missing = [o1[["repeat", "fold", "case_id", "true_label", "predicted_label", "prob_high_grade"]]], []
for r in range(1, 10):
    for f in range(1, 6):
        d = os.path.join(ROOT, f"r{r:02d}_f{f}")
        preds = sorted(glob.glob(f"{d}/predictions*.csv"))
        if not preds:
            missing.append(f"r{r:02d}_f{f}")
            continue
        df = pd.read_csv(preds[0]).rename(columns={"prob_class1": "prob_high_grade"})
        df["repeat"], df["fold"] = r, f
        frames.append(df[["repeat", "fold", "case_id", "true_label", "predicted_label", "prob_high_grade"]])

if missing:
    print(f"WARNING: {len(missing)} runs have no predictions yet: {', '.join(missing)}\n")

long = pd.concat(frames, ignore_index=True)
long["wrong"] = long.true_label != long.predicted_label
long.to_csv("obj2_long.csv", index=False)

per_rep = long.groupby("repeat").case_id.agg(["count", "nunique"])
bad = per_rep[(per_rep["count"] != 93) | (per_rep["nunique"] != 93)]
if len(bad):
    print("WARNING: repeats without exactly 93 test cases:\n", bad, "\n")

wide = long.pivot_table(index="case_id", columns="repeat", values="wrong", aggfunc="first")
out = pd.DataFrame({
    "true_grade": long.groupby("case_id").true_label.first().map({0: "benign", 1: "high-grade"}),
    "n_repeats": wide.notna().sum(axis=1),
    "n_wrong": wide.sum(axis=1).astype(int),
    "mean_prob_high_grade": long.groupby("case_id").prob_high_grade.mean().round(3),
    "pattern": wide.apply(lambda row: " ".join("." if v is False or v == 0 else ("X" if v else "-")
                                               for v in row), axis=1),
})
out["category"] = pd.cut(out.n_wrong, [-1, 0, HARD_MIN - 1, 99], labels=["never", "unstable", "hard"])
out = out.sort_values(["n_wrong", "case_id"], ascending=[False, True])
out.to_csv("obj2_wrong_counts.csv")

print(f"Repeats loaded: {sorted(int(r) for r in long.repeat.unique())}   (pattern columns = repeats 0..9, X = wrong, - = missing)")
print(out.category.value_counts().to_string(), "\n")
print(out[out.n_wrong > 0].to_string())
