import glob, re, sys, pandas as pd

run_dirs = {}                                   # fold -> run_dir; later job IDs override earlier ones
for job in sys.argv[1:]:
    for log in sorted(glob.glob(f"logs/obj1_cv_{job}_*.out")):
        fold = int(re.search(r"_(\d+)\.out$", log).group(1)) + 1
        m = re.search(r"Created run directory: (\S+)", open(log).read())
        if m: run_dirs[fold] = m.group(1)

if len(set(run_dirs.values())) < len(run_dirs):
    print("WARNING: two folds share a run directory:", run_dirs)

rows = []
for fold, run_dir in sorted(run_dirs.items()):
    preds = sorted(glob.glob(f"{run_dir}/predictions*.csv"))
    if not preds: print(f"fold {fold}: no predictions CSV in {run_dir}"); continue
    df = pd.read_csv(preds[0]); df.insert(0, "fold", fold); df["run_dir"] = run_dir
    rows.append(df)

df = pd.concat(rows, ignore_index=True)
df["correct"] = df.true_label == df.predicted_label
df["true_grade"] = df.true_label.map({0: "benign", 1: "high-grade"})
df = df.rename(columns={"prob_class0": "prob_benign", "prob_class1": "prob_high_grade"})
df.to_csv("obj1_all_predictions.csv", index=False)
wrong = df[~df.correct].sort_values(["fold", "case_id"])
wrong.to_csv("obj1_misclassified.csv", index=False)

def summ(d):
    b, h = d[d.true_label == 0], d[d.true_label == 1]
    return pd.Series({"n": len(d), "accuracy": d.correct.mean(),
                      "benign_recall": b.correct.mean(), "high_grade_recall": h.correct.mean(),
                      "n_wrong": int((~d.correct).sum())})
tab = pd.DataFrame({f: summ(g) for f, g in df.groupby("fold")}).T
tab.loc["all"] = summ(df)
print(tab.round(3).to_string())
print(f"\nCases tested: {df.case_id.nunique()} (rows: {len(df)})  |  misclassified: {len(wrong)}")
print(wrong[["fold", "case_id", "true_grade", "prob_high_grade"]].to_string(index=False))