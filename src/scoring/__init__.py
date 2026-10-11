"""Scoring utilities (thin batch layer over dpo_utils.compute_log_prob)."""

from .logprob import compute_log_prob_batch, score_candidates

__all__ = ["compute_log_prob_batch", "score_candidates"]
