"""
Task-specific evaluation metrics for the personalized baselines.

Mirrors the metric semantics of the reference evaluation script but is
organised so metrics can be computed *per test user* and then averaged:

  * discriminative (citation, movie_tagging) -> accuracy, macro_f1
  * regression      (product_rating)          -> mae, rmse
  * generative      (news/scholarly/tweet)    -> rouge1, rougeL, bertscore_{p,r,f1}

`compute_user_metrics` returns ONLY the numeric metric dict for one user (no
metadata), so the caller can freely average the values across users.
"""

import re
from typing import Dict, List, Optional, Sequence

import numpy as np

DISCRIMINATIVE_TASKS = {"citation", "movie_tagging"}
REGRESSION_TASKS = {"product_rating"}
GENERATIVE_TASKS = {"news_headline", "scholarly_title", "tweet_paraphrase"}

# Metric keys each task family can produce (used for stable averaging/ordering).
METRIC_KEYS = {
    "discriminative": ["accuracy", "macro_f1"],
    "regression": ["mae", "rmse"],
    "generative": ["rouge1", "rougeL",
                   "bertscore_precision", "bertscore_recall", "bertscore_f1"],
}


def task_family(task_name: str) -> str:
    """Return 'discriminative' | 'regression' | 'generative' for a task."""
    if task_name in DISCRIMINATIVE_TASKS:
        return "discriminative"
    if task_name in REGRESSION_TASKS:
        return "regression"
    if task_name in GENERATIVE_TASKS:
        return "generative"
    raise ValueError(f"No metric family defined for task: {task_name}")


# ============================ output parsing ============================
def parse_citation_output(output: str) -> str:
    m = re.search(r'\[([12])\]', output)
    if m:
        return f"[{m.group(1)}]"
    if '1' in output and '2' not in output:
        return "[1]"
    if '2' in output and '1' not in output:
        return "[2]"
    return output.strip()


def parse_movie_output(output: str) -> str:
    return output.strip().lower()


def parse_rating_output(output: str) -> int:
    m = re.search(r'([1-5])', output)
    if m:
        return int(m.group(1))
    return -1


# ============================ metric primitives ============================
def compute_acc_f1(preds: Sequence, labels: Sequence):
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    acc = correct / len(labels) if labels else 0.0

    classes = sorted(set(labels) | set(preds))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_true = np.array([cls_to_idx[l] for l in labels])
    y_pred = np.array([cls_to_idx[p] for p in preds])

    f1_scores = []
    for c in classes:
        idx = cls_to_idx[c]
        tp = ((y_pred == idx) & (y_true == idx)).sum()
        fp = ((y_pred == idx) & (y_true != idx)).sum()
        fn = ((y_pred != idx) & (y_true == idx)).sum()
        if tp + fp + fn == 0:
            continue
        f1_scores.append(2 * tp / (2 * tp + fp + fn))

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return float(acc), macro_f1


def compute_mae_rmse(preds: Sequence, labels: Sequence):
    preds = np.array(preds, dtype=float)
    labels = np.array(labels, dtype=float)
    mae = float(np.mean(np.abs(preds - labels)))
    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
    return mae, rmse


def compute_rouge(preds: Sequence[str], labels: Sequence[str]):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    r1_scores, rl_scores = [], []
    for p, l in zip(preds, labels):
        s = scorer.score(l, p)
        r1_scores.append(s['rouge1'].fmeasure)
        rl_scores.append(s['rougeL'].fmeasure)
    return float(np.mean(r1_scores)), float(np.mean(rl_scores))


class BertScorer:
    """Lazily-loaded, reusable BERTScore wrapper (loads DeBERTa only once).

    The per-user metric loop would otherwise reload the encoder for every user;
    a single cached `bert_score.BERTScorer` instance avoids that. `available`
    stays False when the dependency/model is missing so callers can skip it.
    """

    def __init__(self, deberta_path: str, num_layers: int = 24, device: str = "cuda"):
        self.deberta_path = deberta_path
        self.num_layers = num_layers
        self.device = device
        self._scorer = None
        self._failed = False

    @property
    def available(self) -> bool:
        if self._scorer is None and not self._failed:
            try:
                from bert_score import BERTScorer
                self._scorer = BERTScorer(
                    model_type=self.deberta_path, lang="en",
                    device=self.device, num_layers=self.num_layers,
                )
            except Exception:
                self._failed = True
        return self._scorer is not None

    def score(self, preds: Sequence[str], labels: Sequence[str]) -> Optional[Dict[str, float]]:
        if not self.available:
            return None
        # bert_score logs a WARNING for every empty candidate ("Empty candidate
        # sentence detected; setting raw BERTscores to 0."). Its numeric handling
        # (score 0) is fine, but under our stdout/stderr->logger redirection that
        # warning is noisy, so silence the bert_score loggers for this call only.
        import logging
        bs_logger = logging.getLogger("bert_score")
        prev_level, prev_propagate = bs_logger.level, bs_logger.propagate
        bs_logger.setLevel(logging.ERROR)
        bs_logger.propagate = False
        try:
            P, R, F1 = self._scorer.score(list(preds), list(labels), verbose=False)
        finally:
            bs_logger.setLevel(prev_level)
            bs_logger.propagate = prev_propagate
        return {
            "bertscore_precision": float(P.mean()),
            "bertscore_recall": float(R.mean()),
            "bertscore_f1": float(F1.mean()),
        }


# ============================ per-user metrics ============================
def compute_user_metrics(
    task_name: str,
    preds: List[str],
    labels: List,
    bert_scorer: Optional[BertScorer] = None,
) -> Dict[str, float]:
    """Compute the numeric metric dict for ONE user's (pred, label) pairs.

    Returns only metric keys (see METRIC_KEYS); empty pairs -> empty dict.
    """
    if not preds or not labels:
        return {}

    family = task_family(task_name)
    metrics: Dict[str, float] = {}

    if family == "discriminative":
        parser = parse_citation_output if task_name == "citation" else parse_movie_output
        parsed = [parser(p) for p in preds]
        acc, f1 = compute_acc_f1(parsed, labels)
        metrics["accuracy"] = acc
        metrics["macro_f1"] = f1

    elif family == "regression":
        parsed = [parse_rating_output(p) for p in preds]
        valid = [(p, float(l)) for p, l in zip(parsed, labels) if p != -1]
        if valid:
            vp, vl = zip(*valid)
            mae, rmse = compute_mae_rmse(vp, vl)
            metrics["mae"] = mae
            metrics["rmse"] = rmse

    else:  # generative
        try:
            r1, rl = compute_rouge(preds, labels)
            metrics["rouge1"] = r1
            metrics["rougeL"] = rl
        except ImportError:
            pass
        if bert_scorer is not None:
            bs = bert_scorer.score(preds, labels)
            if bs:
                metrics.update(bs)

    return metrics


def average_metrics(per_user: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Macro-average each metric key across users that reported it."""
    keys = []
    for m in per_user.values():
        for k in m:
            if k not in keys:
                keys.append(k)
    avg = {}
    for k in keys:
        vals = [m[k] for m in per_user.values() if k in m and m[k] is not None]
        if vals:
            avg[k] = float(np.mean(vals))
    return avg
