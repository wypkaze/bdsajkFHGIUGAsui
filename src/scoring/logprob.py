"""
Batched log-probability scoring.

This is a *thin* wrapper that preserves the exact semantics of
`src.training.dpo_utils.compute_log_prob` (sum of log-softmax over the
completion tokens only), but processes many (prompt, completion) pairs in a
single batched forward pass with LEFT padding.

Left padding is required because the same tokenizer/model is loaded with
`padding_side="left"` (see sft_target_user.py / models.loader.load_model), and
generation-style batching relies on it. With left padding, each sample's real
tokens are right-aligned, so the completion-token absolute positions are
`pad_i + prompt_len_i ... pad_i + full_len_i - 1`.

Why sum (not average): for a given query the gold completion y_w is identical
across all anchors, so ranking anchors by sum log p_anchor(y_w | prompt) is
equivalent to ranking by average log-prob. We keep `sum` to match the reference.
"""

from typing import List, Sequence

import torch
import torch.nn.functional as F

from ..utils.text_utils import split_batch

# Same sentinel the reference compute_log_prob returns for an empty completion.
_EMPTY_SCORE = -1e9


def _encode_one(tokenizer, prompt: str, completion: str, cutoff_len: int):
    """
    Tokenize a single (prompt, completion) exactly like dpo_utils.compute_log_prob.

    Returns (full_ids: List[int], prompt_len: int, completion_len: int).
    completion_len <= 0 signals an empty completion (caller assigns sentinel).
    """
    full_ids = tokenizer(
        prompt + completion, truncation=True, max_length=cutoff_len,
        return_tensors="pt",
    ).input_ids[0]
    prompt_len = tokenizer(
        prompt, truncation=True, max_length=cutoff_len,
        return_tensors="pt",
    ).input_ids.shape[1]
    completion_len = full_ids.shape[0] - prompt_len

    if completion_len <= 0:
        # Mirror the reference fallback: insert a space so the completion is
        # tokenized to at least one token.
        full_ids = tokenizer(
            prompt + " " + completion, truncation=True, max_length=cutoff_len,
            return_tensors="pt",
        ).input_ids[0]
        prompt_len = tokenizer(
            prompt, truncation=True, max_length=cutoff_len,
            return_tensors="pt",
        ).input_ids.shape[1]
        completion_len = full_ids.shape[0] - prompt_len

    return full_ids.tolist(), prompt_len, completion_len


@torch.no_grad()
def _score_batch(model, tokenizer, prompts, completions, cutoff_len,
                 length_normalized=False):
    """Score one batch; returns a list of float log-probs (same order).

    When length_normalized is True each score is divided by the number of
    scored completion tokens (average log-prob per token), which removes the
    length bias that summed log-probs impose on candidates of different token
    lengths (e.g. "sci-fi" vs "based on a book").
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    encoded = [_encode_one(tokenizer, p, c, cutoff_len)
               for p, c in zip(prompts, completions)]

    scores = [_EMPTY_SCORE] * len(encoded)
    valid = [i for i, (_, _, clen) in enumerate(encoded) if clen > 0]
    if not valid:
        return scores

    max_len = max(len(encoded[i][0]) for i in valid)

    input_ids = torch.full((len(valid), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(valid), max_len), dtype=torch.long)
    for row, i in enumerate(valid):
        ids = encoded[i][0]
        L = len(ids)
        pad = max_len - L  # left padding
        input_ids[row, pad:] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, pad:] = 1

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs_all = F.log_softmax(logits, dim=-1)  # [B, T, V]

    for row, i in enumerate(valid):
        ids, prompt_len, completion_len = encoded[i]
        pad = max_len - len(ids)
        # Absolute index of the first completion token in the padded sequence.
        start = pad + prompt_len
        # Logits at positions [start-1 : start-1+completion_len] predict the
        # completion tokens at [start : start+completion_len].
        pred = log_probs_all[row, start - 1:start - 1 + completion_len, :]
        target = input_ids[row, start:start + completion_len].unsqueeze(-1)
        token_lp = pred.gather(1, target).squeeze(-1)
        total = token_lp.sum()
        if length_normalized and completion_len > 0:
            total = total / completion_len
        scores[i] = total.item()

    return scores


def compute_log_prob_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    completions: Sequence[str],
    cutoff_len: int,
    batch_size: int = 32,
    length_normalized: bool = False,
) -> List[float]:
    """
    Batched version of dpo_utils.compute_log_prob.

    Args:
        model: HuggingFace model or PeftModel (eval mode recommended).
        tokenizer: Tokenizer with padding_side="left".
        prompts: Prompt strings.
        completions: Completion strings (aligned with prompts).
        cutoff_len: Max total sequence length (prompt + completion).
        batch_size: Number of pairs per forward pass.
        length_normalized: if True, return the average log-prob per completion
            token (sum / token_count) instead of the sum. Default False keeps
            the exact dpo_utils.compute_log_prob semantics used by Pass A.

    Returns:
        List of log-probabilities over completion tokens, one per input pair,
        in the same order. Empty completions get the -1e9 sentinel.
    """
    assert len(prompts) == len(completions), "prompts/completions length mismatch"
    was_training = model.training
    model.eval()

    results: List[float] = []
    prompt_batches = split_batch(list(prompts), batch_size)
    # split_batch drops nothing but we must chunk completions identically.
    comp_batches = split_batch(list(completions), batch_size)
    for pb, cb in zip(prompt_batches, comp_batches):
        results.extend(_score_batch(model, tokenizer, pb, cb, cutoff_len,
                                    length_normalized))

    if was_training:
        model.train()
    return results


def score_candidates(
    model,
    tokenizer,
    prompt: str,
    candidates: Sequence[str],
    cutoff_len: int,
    batch_size: int = 32,
    length_normalized: bool = False,
) -> List[float]:
    """
    Score multiple candidate completions against a single shared prompt.

    Used by the movie_tagging branch: rank the 15 tags by log p_anchor(tag|prompt).
    With length_normalized=True the ranking uses the average log-prob per token
    so that longer tags are not penalised purely for having more tokens.
    Returns a list of log-probs aligned with `candidates`.
    """
    prompts = [prompt] * len(candidates)
    return compute_log_prob_batch(
        model, tokenizer, prompts, list(candidates), cutoff_len, batch_size,
        length_normalized,
    )
