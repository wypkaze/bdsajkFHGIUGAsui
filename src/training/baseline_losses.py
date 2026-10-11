"""
PAPPO loss function.

Interface (unchanged from the shared baseline driver):

    loss_fn(model, tokenizer, batch_items, beta=..., cut_off=..., norm=...,
            no_grad=False, verbose=False, logger=None)
        -> (loss | None, stats)

Items come from `src.data.precompute_loader.build_user_items` and carry:
    prompt, y_w, y_l, len_w, len_l, sp_val_pappo_norm, sp_val_pappo_unorm, ...

PAPPO objective on the precomputed aggregated reference margin Δ̂_ref:

    Δ_θ       = t_w − t_l          (per-token if norm else raw sum log-probs)
    Δ̂_ref     = sp_val_pappo_<norm|unorm>   (read from the precompute record)
    L         = −log σ( β · (Δ_θ − Δ̂_ref) )

Δ̂_ref is the ReLU-gated weighted mean of the per-anchor margins that
precompute.py already aggregated, so no anchor model is ever forwarded here.

Trimmed: the DPO / SimPO / IPO / HyPO / MRPO objectives are gone.
"""

import torch
import torch.nn.functional as F

from .dpo_utils import compute_log_prob

# log-prob sentinel: anything <= this is treated as an invalid/empty completion.
_NEG_INF = -1e8


def _emit(msg, logger=None):
    """Per-pair detail line: INFO level so it lands in the run log."""
    if logger is not None:
        logger.info(msg)
    else:
        print(msg, flush=True)


def _scalar(value):
    """Float view of a value that may be a 0-dim tensor or a plain float."""
    return value.item() if isinstance(value, torch.Tensor) else float(value)


def _policy_logps(model, tokenizer, item, cut_off, no_grad, device=None):
    """Sum log-probs of y_w / y_l under the policy (grad flows unless no_grad).

    `compute_log_prob` returns a float when no_grad=True and a 0-dim tensor
    otherwise; both are normalised to tensors so the arithmetic below is uniform.
    """
    lp_w = compute_log_prob(model, tokenizer, item["prompt"], item["y_w"],
                            cut_off, no_grad=no_grad)
    lp_l = compute_log_prob(model, tokenizer, item["prompt"], item["y_l"],
                            cut_off, no_grad=no_grad)
    if _scalar(lp_w) <= _NEG_INF or _scalar(lp_l) <= _NEG_INF:
        return None, None
    if not isinstance(lp_w, torch.Tensor):
        lp_w = torch.tensor(float(lp_w), device=device)
    if not isinstance(lp_l, torch.Tensor):
        lp_l = torch.tensor(float(lp_l), device=device)
    return lp_w, lp_l


def _normalized(lp, length):
    """Average per-token log-prob (0-dim tensor when lp carries grad)."""
    return lp / max(int(length), 1)


def _new_stats():
    return {"valid_pairs": 0, "skipped_pairs": 0, "sum_margin": 0.0}


def _finalize(total, stats, beta):
    n = stats["valid_pairs"]
    if n == 0:
        return None, {**stats, "loss": None, "margin": None, "beta": beta}
    loss = total / n
    stats["loss"] = _scalar(loss)
    stats["margin"] = stats["sum_margin"] / n
    stats["beta"] = beta
    return loss, stats


def compute_pappo_loss(model, tokenizer, batch_items, *, beta, cut_off, norm=True,
                       gamma=None, no_grad=False, verbose=False, logger=None, **kwargs):
    """PAPPO: DPO-style loss against the ReLU-gated aggregated reference margin.

    `gamma` is accepted (and ignored) so the call surface matches the shared
    driver; PAPPO has no target-margin term.
    """
    device = next(model.parameters()).device
    total = torch.tensor(0.0, device=device)
    stats = _new_stats()
    sp_field = "sp_val_pappo_norm" if norm else "sp_val_pappo_unorm"

    for item in batch_items:
        lp_w, lp_l = _policy_logps(model, tokenizer, item, cut_off, no_grad, device)
        if lp_w is None:
            stats["skipped_pairs"] += 1
            continue

        t_w = _normalized(lp_w, item["len_w"]) if norm else lp_w
        t_l = _normalized(lp_l, item["len_l"]) if norm else lp_l
        delta_theta = t_w - t_l
        delta_ref_hat = float(item.get(sp_field, 0.0) or 0.0)

        log_ratio = beta * (delta_theta - delta_ref_hat)
        pair_loss = -F.logsigmoid(log_ratio)

        total = total + pair_loss
        stats["valid_pairs"] += 1
        stats["sum_margin"] += _scalar(delta_theta) - delta_ref_hat

        if verbose:
            _emit(f"  [PAPPO] qid={item.get('qid')} norm={norm} "
                  f"t_w={_scalar(t_w):.4f} t_l={_scalar(t_l):.4f} "
                  f"delta_theta={_scalar(delta_theta):.4f} "
                  f"delta_ref_hat({sp_field})={delta_ref_hat:.4f} "
                  f"log_ratio={_scalar(log_ratio):.4f} loss={_scalar(pair_loss):.4f}", logger)

    return _finalize(total, stats, beta)


BASELINE_LOSS_FN_MAP = {
    "pappo": compute_pappo_loss,
}
