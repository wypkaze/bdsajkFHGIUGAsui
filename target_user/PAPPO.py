#!/usr/bin/env python3
"""
Per-user PAPPO on the task_LoRA merged model.

PAPPO aggregates the reference anchors into a single per-triple margin Δ̂_ref with a
ReLU-gated weighted mean (produced by the upstream precompute stage). That margin is
precomputed and stored in the record, so no anchor model is ever forwarded here:

    Δ̂_ref = sp_val_pappo_norm   (--norm, per-token)  |  sp_val_pappo_unorm (--no_norm)
    Δ_θ   = t_w - t_l           (policy margin on the same scale as Δ̂_ref)
    L     = -log sigma( beta * (Δ_θ - Δ̂_ref) )

Inputs
    PAPPO_record/data/<task>/pappo_anchor/dpo_pair.json      preference triples
    PAPPO_record/records/precompute/<task>/<strategy>/       aggregated sp_val per triple
                                                             (default strategy: similar)

The policy starts from the task_LoRA merged model and gets a fresh LoRA adapter per
user. Shared logic (CLI, splits, early stopping, logging, checkpoint layout) lives in
src/training/baseline_trainer.py.

Usage:
    python PAPPO.py --task_name citation --beta 0.1                 # length-normalized
    python PAPPO.py --task_name citation --beta 0.1 --no_norm       # raw sum log-probs
    python PAPPO.py --task_name citation --anchor_selection noisy --gpu_id 0
    python PAPPO.py --task_name citation --max_users 1 --dpo_epoch 1 --dry_run
"""

import os
import sys

_gpu_id = 0
for i, arg in enumerate(sys.argv):
    if arg == '--gpu_id' and i + 1 < len(sys.argv):
        _gpu_id = sys.argv[i + 1]
        break
os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_id)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # PAPPO/
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.training.baseline_trainer import main

if __name__ == "__main__":
    main("pappo")
