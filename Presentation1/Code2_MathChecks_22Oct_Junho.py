# =============================================================================
# Code2_MathChecks_22Oct_Junho.py
# Author: Junho Hong (Team1_FullStack, Mathlete)
#
# Purpose
#   Reproduces two numbers used on the Mathletes slides.
#   1) Parameter counts of HierarchicalAttnMIL (and of the frozen KimiaNet encoder)
#   2) That weighted cross-entropy with batch size 1 ignores the class weight
#
# UPDATE BEFORE RUNNING
#   BASE_CODE_DIR  -> folder that contains models.py and config.py of the base code
#
# Run on Quest (after activating the course environment):
#   module purge
#   module load python-anaconda3/2019.10
#   eval "$(conda shell.bash hook)"
#   conda activate /projects/e32998/env
#   python3 Code2_MathChecks_22Oct_Junho.py
# =============================================================================

import sys

import torch
import torch.nn as nn
import torchvision.models as tvm

# ----------------------------------------------------------------------------
# Paths (UPDATE if the base code moves)
# ----------------------------------------------------------------------------
BASE_CODE_DIR = "/projects/e32998/STAT390_Fall2026/Team6_InstructionalTeam/Code/Presentation1_base_code"
sys.path.insert(0, BASE_CODE_DIR)

from models import create_model  # noqa: E402  (import after sys.path update)


def count(module):
    """Return (total, trainable) parameter counts of a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


# ----------------------------------------------------------------------------
# 1) Parameter counts
# ----------------------------------------------------------------------------
model = create_model(num_classes=2, embed_dim=512, dropout=0.2599)

print("=== HierarchicalAttnMIL parameter counts ===")
for name in ["patch_projector", "patch_attention", "stain_attention", "case_attention", "classifier"]:
    total, _ = count(getattr(model, name))
    print(f"{name:16s} {total:>12,d}")
total, trainable = count(model)
print(f"{'TOTAL':16s} {total:>12,d}   (trainable: {trainable:,d})")
print("Expected total: 2,493,061")

# KimiaNet uses the DenseNet-121 architecture; only its convolutional part is used
encoder = tvm.densenet121(weights=None).features
enc_total, _ = count(encoder)
print(f"\nDenseNet-121 features (KimiaNet encoder, frozen): {enc_total:,d}")

# ----------------------------------------------------------------------------
# 2) Weighted cross-entropy with one case per batch
# ----------------------------------------------------------------------------
print("\n=== Class weight check (batch size 1) ===")
w = torch.tensor([2.5285, 1.0])        # config.py class_weights (benign, high-grade)
logits = torch.tensor([[0.3, -0.2]])   # any logits for one case
label = torch.tensor([0])              # a benign case

ce_plain = nn.CrossEntropyLoss()(logits, label).item()
ce_weighted_mean = nn.CrossEntropyLoss(weight=w)(logits, label).item()            # what trainer.py uses
ce_weighted_sum = nn.CrossEntropyLoss(weight=w, reduction="sum")(logits, label).item()  # proposed fix

print(f"CE without weights          : {ce_plain:.6f}")
print(f"CE weighted, reduction=mean : {ce_weighted_mean:.6f}   <- identical, weight cancels")
print(f"CE weighted, reduction=sum  : {ce_weighted_sum:.6f}   <- 2.5285 x plain CE")
