"""
Training utilities for PAPPO.

Provides the anchor-user trainer (upstream pipeline stage) and the PAPPO
per-user preference-optimization driver + loss. The other-method losses
(DPO / SimPO / IPO / HyPO / MRPO) and the legacy `dpo_losses` / `pair_generator`
modules have been removed; preference-pair generation now lives in
`src.generation` and is consumed by pair_gen/gen_pairs.py.
"""

from .anchor_trainer import AnchorTrainer
from .dpo_utils import compute_log_prob
from .baseline_losses import compute_pappo_loss, BASELINE_LOSS_FN_MAP
from .baseline_trainer import (
    build_arg_parser,
    resolve_args,
    run_baseline,
    main as run_baseline_main,
)

__all__ = [
    'AnchorTrainer',
    'compute_log_prob',
    'compute_pappo_loss',
    'BASELINE_LOSS_FN_MAP',
    'build_arg_parser',
    'resolve_args',
    'run_baseline',
    'run_baseline_main',
]
