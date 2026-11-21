"""Objectives 3 and 4: top-attention patches and the direction each one pushes.

For each hard case (and a few correctly classified cases of the same grade from
the same model) this script:
  * reads every patch's effective attention a_i from attention_analysis,
  * recomputes h_i = ReLU(W_p x_i + b_p) from the patch's UNI2 embedding,
  * computes E_i = a_i (W_1 - W_0)^T h_i  (slide 41; E_i > 0 pushes to high-grade),
  * checks that sum_i E_i + (b_1 - b_0) equals the model's logit difference,
  * saves tables and figures to obj34_out/.
Run from ~/Presentation3/Presentation3_base_code:  python obj34_analysis.py
"""
import glob, os, re
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------- settings
RUNS_ROOT = "/projects/e32998/MIL_training/final_runs/obj2_chj4322"
EMB_DIR = "/home/chj4322/Presentation3/team2_patch_embeddings_UNI2"
HARD = {18: "r01_f4", 46: "r01_f2", 78: "r03_f5"}   # run where each was most confidently wrong
N_COMPARE = 3          # correctly classified cases of the same grade, same run
TOP_K = 8              # top-attention patches to show / count
OUT = "obj34_out"

HG_COLOR, BEN_COLOR = "#e34948", "#2a78d6"            # red = toward high-grade, blue = toward benign
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})
GRADE = {0: "benign", 1: "high-grade"}
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- helpers
def load_weights(run_dir):
    ck = torch.load(os.path.join(run_dir, "checkpoints", "best.pth"), map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    sd = {re.sub(r"^(module\.|model\.)", "", k): v for k, v in sd.items()}
    Wp, bp = sd["patch_projector.0.weight"].float(), sd["patch_projector.0.bias"].float()
    ck_w = [k for k in sd if k.startswith("classifier.") and k.endswith(".weight")]
    assert len(ck_w) == 1, f"expected one classifier Linear, found {ck_w}"
    Wc, bc = sd[ck_w[0]].float(), sd[ck_w[0].replace("weight", "bias")].float()
    assert Wc.shape[0] == 2, Wc.shape
    return Wp, bp, Wc, bc


def contributions(rows, Wp, bp, Wc, bc):
    w_diff = Wc[1] - Wc[0]
    X = []
    for f in rows.patch_file:
        x = torch.load(os.path.join(EMB_DIR, os.path.splitext(f)[0] + ".pt"), map_location="cpu")
        X.append(torch.as_tensor(x).float().flatten())
    H = torch.relu(torch.stack(X) @ Wp.T + bp)                      # (P, 512)
    a = torch.tensor(rows.effective_weight.values, dtype=torch.float32)
    E = a * (H @ w_diff)                                             # (P,)
    out = rows.copy()
    out["E"] = E.numpy()
    return out, float(bc[1] - bc[0])


def montage(top, title, path):
    n = len(top); cols = min(4, n); rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(2.6 * cols, 3.3 * rws), gridspec_kw={"hspace": 0.45})
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (_, r) in zip(np.atleast_1d(axes).ravel(), top.iterrows()):
        try:
            ax.imshow(Image.open(r.patch_path).convert("RGB"))
        except Exception:
            ax.text(0.5, 0.5, "image\nnot found", ha="center", va="center", color=MUTED)
        col = HG_COLOR if r.E > 0 else BEN_COLOR
        for s in ax.spines.values():
            s.set_visible(True); s.set_edgecolor(col); s.set_linewidth(4)
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        arrow = "→ high-grade" if r.E > 0 else "→ benign"
        ax.set_title(f"{r.stain} · {r.slice_id} · p{re.search(r'patch(\d+)', r.patch_file).group(1)}\n"
                     f"a={r.effective_weight:.3f}  E={r.E:+.3f} {arrow}", fontsize=8, color=INK)
    fig.suptitle(title, fontsize=11, color=INK, y=1.0)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def contrib_bars(df, title, path, n=20):
    top = df.nlargest(n, "effective_weight").iloc[::-1]
    labels = [f"{r.stain} {r.slice_id} p{re.search(r'patch(\d+)', r.patch_file).group(1)}  (a={r.effective_weight:.3f})"
              for _, r in top.iterrows()]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(top) + 1.2))
    ax.barh(range(len(top)), top.E, color=[HG_COLOR if e > 0 else BEN_COLOR for e in top.E],
            height=0.7, edgecolor="white", linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.set_xlabel("E_i  (← pushes toward benign  |  pushes toward high-grade →)")
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- main loop
wrong_counts = pd.read_csv("obj2_wrong_counts.csv", index_col=0) if os.path.exists("obj2_wrong_counts.csv") else None
all_rows, summary, outline = [], [], []

for hard_case, tag in HARD.items():
    run_dir = os.path.join(RUNS_ROOT, tag)
    att = pd.read_csv(os.path.join(run_dir, "attention_analysis", "patch_attention_by_filename.csv"))
    preds = pd.read_csv(sorted(glob.glob(os.path.join(run_dir, "predictions*.csv")))[0])
    Wp, bp, Wc, bc = load_weights(run_dir)

    true = int(preds.loc[preds.case_id == hard_case, "true_label"].iloc[0])
    same = preds[(preds.true_label == true) & (preds.true_label == preds.predicted_label) & (preds.case_id != hard_case)]
    conf_col = "prob_class1" if true == 1 else "prob_class0"
    compare = same.sort_values(conf_col, ascending=False).case_id.head(N_COMPARE).tolist()

    for case in [hard_case] + compare:
        role = "hard" if case == hard_case else f"correct (vs {hard_case})"
        rows = att[att.case_id == case].reset_index(drop=True)
        df, bias = contributions(rows, Wp, bp, Wc, bc)
        df.insert(0, "run", tag); df.insert(1, "role", role)
        all_rows.append(df)

        p = preds[preds.case_id == case].iloc[0]
        logit = float(np.log(max(p.prob_class1, 1e-12)) - np.log(max(p.prob_class0, 1e-12)))
        recon = df.E.sum() + bias
        wrong_sign = 1 if true == 0 else -1                           # sign of E that pushes to the wrong class
        top = df.nlargest(TOP_K, "effective_weight")
        stain_E = df.groupby("stain").E.sum()
        summary.append({
            "case_id": case, "role": role, "run": tag, "true_grade": GRADE[true],
            "pred_grade": GRADE[int(p.predicted_label)], "prob_high_grade": round(float(p.prob_class1), 4),
            "n_wrong_of_10": int(wrong_counts.loc[case, "n_wrong"]) if wrong_counts is not None and case in wrong_counts.index else None,
            "n_patches": len(df), "sum_attention": round(df.effective_weight.sum(), 4),
            f"top{TOP_K}_toward_wrong": int((np.sign(top.E) == wrong_sign).sum()),
            f"top{TOP_K}_sum_E": round(top.E.sum(), 3),
            "all_patches_sum_E": round(df.E.sum(), 3),
            "share_of_|E|_toward_wrong": round(df.E[np.sign(df.E) == wrong_sign].abs().sum() / df.E.abs().sum(), 3),
            **{f"sum_E_{s}": round(v, 3) for s, v in stain_E.items()},
            "bias_diff": round(bias, 3), "recon_logit": round(recon, 3), "model_logit": round(logit, 3),
            "recon_matches": abs(recon - logit) < 1e-2 * max(1, abs(logit)),
        })
        for rank, (_, r) in enumerate(top.iterrows(), 1):
            outline.append({"case_id": case, "role": role, "run": tag, "rank": rank, "stain": r.stain,
                            "slice_id": r.slice_id, "patch_file": r.patch_file, "patch_path": r.patch_path,
                            "effective_weight": r.effective_weight, "E": r.E,
                            "pushes_toward": "high-grade" if r.E > 0 else "benign"})

        name = f"case{case}_{tag}" + ("_HARD" if role == "hard" else "")
        head = (f"Case {case} ({GRADE[true]}, predicted {GRADE[int(p.predicted_label)]}, "
                f"P(high-grade)={p.prob_class1:.3f}) — {role}, model {tag}")
        montage(top, f"Top-{TOP_K} attention patches — {head}", os.path.join(OUT, f"{name}_top_patches.png"))
        contrib_bars(df, head, os.path.join(OUT, f"{name}_contributions.png"))

pd.concat(all_rows, ignore_index=True).to_csv(os.path.join(OUT, "patch_contributions.csv"), index=False)
pd.DataFrame(outline).to_csv(os.path.join(OUT, "top_patches_for_slice_outline.csv"), index=False)
S = pd.DataFrame(summary)
S.to_csv(os.path.join(OUT, "summary.csv"), index=False)

# Comparison figure: net push of the top-K patches, hard case next to its correct same-grade cases
fig, axes = plt.subplots(1, len(HARD), figsize=(4 * len(HARD), 3.2), sharey=True)
for ax, (hc, tag) in zip(np.atleast_1d(axes), HARD.items()):
    g = S[S.run == tag]
    x = range(len(g))
    ax.bar(x, g[f"top{TOP_K}_sum_E"], color=[HG_COLOR if v > 0 else BEN_COLOR for v in g[f"top{TOP_K}_sum_E"]],
           width=0.6, edgecolor="white", linewidth=2)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{c}\n{'HARD' if r == 'hard' else 'correct'}" for c, r in zip(g.case_id, g.role)], fontsize=9)
    ax.set_title(f"Hard case {hc} ({g.true_grade.iloc[0]}), model {tag}", fontsize=10, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
np.atleast_1d(axes)[0].set_ylabel(f"Sum of E_i, top-{TOP_K} patches\n(+ high-grade / − benign)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "comparison_topK_push.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

pd.set_option("display.width", 200)
cols = ["case_id", "role", "true_grade", "pred_grade", "prob_high_grade", "n_wrong_of_10",
        f"top{TOP_K}_toward_wrong", f"top{TOP_K}_sum_E", "share_of_|E|_toward_wrong", "recon_logit", "model_logit", "recon_matches"]
print(S[cols].to_string(index=False))
if not S.recon_matches.all():
    print("\nWARNING: sum of E_i + bias does not match the model logit for some cases (see summary.csv).")
print(f"\nSaved tables and figures to {OUT}/")
