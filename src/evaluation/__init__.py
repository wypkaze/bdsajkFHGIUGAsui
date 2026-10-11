"""
Evaluation package for the personalized baselines.

  * metrics.py : per-task metric primitives + per-user computation + averaging
  * runner.py  : CLI, path/version resolution, generation and metric phases

Entry point lives at PAPPO/evaluate/evaluate.py.
"""

from .metrics import (
    DISCRIMINATIVE_TASKS,
    GENERATIVE_TASKS,
    REGRESSION_TASKS,
    BertScorer,
    average_metrics,
    compute_user_metrics,
    task_family,
)
from .runner import main

__all__ = [
    "main",
    "compute_user_metrics",
    "average_metrics",
    "task_family",
    "BertScorer",
    "DISCRIMINATIVE_TASKS",
    "REGRESSION_TASKS",
    "GENERATIVE_TASKS",
]
