
"""
 Usage (in main.py, after trainer.evaluate):
    from selected_case_viz import analyze_selected_cases
    analyze_selected_cases(trainer.model, test_loader, run_dir, top_n=args.attention_top_n)
"""
 
import os
import csv
from typing import Any, Dict
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
 
from attention_analysis import (
    _lookup_png_for_patch,
    _safe_get_slice_tensor,
    _plot_feature_heatmap,
)
 
HIGH_GRADE_LABEL = 1
 
CATEGORIES = [
    "correct_high_grade",
    "correct_benign",
    "incorrect_high_grade",
    "incorrect_benign",
]
 
def _predict(model, case_data):
    logits, attn = model(case_data["stain_slices"], return_attn_weights=True)
    logits = logits.detach()
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    probs = torch.softmax(logits, dim=1)
    pred = int(torch.argmax(probs, dim=1)[0].item())
    pred_prob = float(probs[0, pred].item())
 
    label = case_data.get("label")
    true = int(label.view(-1)[0].item()) if isinstance(label, torch.Tensor) else int(label)
    return true, pred, pred_prob, attn
 
 
def _category(true: int, pred: int) -> str:
    grade = "high_grade" if true == HIGH_GRADE_LABEL else "benign"
    return f"{'correct' if true == pred else 'incorrect'}_{grade}"
 
 
def select_cases(model, test_loader) -> Dict[str, Any]:
    """Pass 1: classify every test case, pick the most confident case per category."""
    buckets = {c: [] for c in CATEGORIES}
    with torch.no_grad():
        for batch in test_loader:
            case_data = batch[0]
            true, pred, pred_prob, _ = _predict(model, case_data)
            buckets[_category(true, pred)].append((pred_prob, case_data["case_id"], true, pred))
 
    selected = {}
    print("\nCase selection (most confident case per category):")
    for cat in CATEGORIES:
        items = buckets[cat]
        if not items:
            print(f"  {cat:22s}: NONE in test set ({len(items)} cases)")
            continue
        prob, cid, true, pred = max(items, key=lambda x: x[0])
        selected[cat] = cid
        print(f"  {cat:22s}: case {cid} (true={true}, pred={pred}, p={prob:.3f}; {len(items)} candidates)")
    return selected
 
 
def _render_patch(ax, case_id, stain, slice_idx, patch_idx, slice_tensor):
    png_path = _lookup_png_for_patch(case_id, stain, slice_idx, patch_idx)
    if png_path and os.path.exists(png_path):
        try:
            ax.imshow(Image.open(png_path).convert("RGB"))
            return png_path
        except Exception:
            pass
    if isinstance(slice_tensor, torch.Tensor) and slice_tensor.dim() == 2 and patch_idx < slice_tensor.shape[0]:
        _plot_feature_heatmap(ax, slice_tensor[patch_idx].detach().cpu().numpy().astype(np.float32))
    else:
        ax.text(0.5, 0.5, "No image found", ha="center", va="center", fontsize=9)
    return ""
 
 
def visualize_case_all_slices(case_id, category, stain_slices, attn, output_dir, top_n=5):
    case_dir = os.path.join(output_dir, "selected_cases", f"{category}_case_{case_id}")
    os.makedirs(case_dir, exist_ok=True)
 
    stain_order = attn.get("stain_order", [])
    case_w = attn["case_weights"].detach().cpu().numpy()
    stain_w = {s: float(case_w[i]) for i, s in enumerate(stain_order)}
 
    csv_rows = []
    for stain, wd in attn.get("stain_weights", {}).items():
        slice_w = wd["slice_weights"].detach().cpu().numpy()
        for slice_idx, pw in enumerate(wd.get("patch_weights", [])):
            if pw is None or slice_idx >= len(slice_w):
                continue
            pw = pw.detach().cpu().numpy()
            n = len(pw)
            if n == 0:
                continue
 
            k = min(top_n, n)
            order = np.argsort(pw)
            groups = [("High", order[-k:][::-1]), ("Low", order[:k])]
            slice_tensor = _safe_get_slice_tensor(stain_slices, stain, slice_idx)
 
            fig, axes = plt.subplots(2, k, figsize=(3 * k, 6.6), squeeze=False)
            for row, (level, idxs) in enumerate(groups):
                for col, pidx in enumerate(idxs):
                    pidx = int(pidx)
                    ax = axes[row][col]
                    ax.axis("off")
                    png = _render_patch(ax, case_id, stain, slice_idx, pidx, slice_tensor)
                    eff = stain_w.get(stain, np.nan) * float(slice_w[slice_idx]) * float(pw[pidx])
                    ax.set_title(
                        f"{level} #{col + 1}  (patch {pidx})\nattn={pw[pidx]:.4f}  eff={eff:.2e}",
                        fontsize=9,
                    )
                    csv_rows.append({
                        "category": category, "case_id": case_id, "stain": stain,
                        "slice_idx": slice_idx, "level": level, "rank": col + 1,
                        "patch_idx": pidx, "patch_attn_weight": float(pw[pidx]),
                        "slice_attn_weight": float(slice_w[slice_idx]),
                        "stain_attn_weight": stain_w.get(stain, np.nan),
                        "effective_weight": eff, "png_path": png,
                    })
 
            fig.suptitle(
                f"{category.replace('_', ' ')} | case {case_id} | {stain} | slice {slice_idx} "
                f"({n} patches)\nstain attn={stain_w.get(stain, np.nan):.3f}, "
                f"slice attn={slice_w[slice_idx]:.3f}   |   top row: high attention, bottom row: low",
                fontsize=11,
            )
            plt.tight_layout()
            safe_stain = stain.replace("&", "and").replace("/", "_").replace(" ", "_")
            fig.savefig(os.path.join(case_dir, f"{safe_stain}_slice{slice_idx}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
 
    if csv_rows:
        with open(os.path.join(case_dir, "plotted_patches.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"  Saved {category} (case {case_id}) -> {case_dir}")
 
 
def analyze_selected_cases(model, test_loader, output_dir, top_n=5):
    print("\n" + "=" * 60)
    print("SELECTED-CASE ATTENTION VISUALIZATION")
    print("=" * 60)
    model.eval()
 
    selected = select_cases(model, test_loader)
    if not selected:
        print("No cases selected; nothing to plot.")
        return
 
    cid_to_cat = {cid: cat for cat, cid in selected.items()}
    with torch.no_grad():
        for batch in test_loader:  # pass 2: plot only the selected cases
            case_data = batch[0]
            cid = case_data["case_id"]
            if cid not in cid_to_cat:
                continue
            _, _, _, attn = _predict(model, case_data)
            visualize_case_all_slices(cid, cid_to_cat[cid], case_data["stain_slices"],
                                      attn, output_dir, top_n=top_n)
