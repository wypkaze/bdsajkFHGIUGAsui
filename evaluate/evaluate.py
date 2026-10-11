#!/usr/bin/env python3
"""
Evaluation entry point for the personalized baselines.

Thin wrapper (mirrors target_user/PAPPO.py): pins the GPU from --gpu_id before any
CUDA import, wires up sys.path, and delegates to src.evaluation.runner.main.

For every test user's queries it (1) saves the model answers under
    records/PAPPO_baselines/<METHOD>/<task>/responses/<eval_suffix>/<run_tag>/user_<uid>.json
and (2) computes per-user metrics + a macro average under
    records/PAPPO_baselines/<METHOD>/<task>/metrics/<eval_suffix>/<run_tag>/{user_<uid>,average}.json
Each run is versioned by <run_tag> (timestamp) so runs never overwrite each other.

Usage:
    python evaluate.py --task_name citation --method pappo --k 1
    python evaluate.py --task_name citation --method oppu --k 1
    python evaluate.py --task_name citation --method pappo --pag
    python evaluate.py --task_name citation --method pappo --k 0
    python evaluate.py --task_name citation --method base --gpu_id 1
    python evaluate.py --task_name scholarly_title --method task_lora --batch_size 4
    # recompute metrics only from the latest responses run:
    python evaluate.py --task_name citation --method pappo --k 1 --metric_only
"""

import os
import sys

_gpu_id = 0
for _i, _arg in enumerate(sys.argv):
    if _arg == '--gpu_id' and _i + 1 < len(sys.argv):
        _gpu_id = sys.argv[_i + 1]
        break
os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_id)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # PAPPO/
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.evaluation.runner import main

if __name__ == "__main__":
    main()
