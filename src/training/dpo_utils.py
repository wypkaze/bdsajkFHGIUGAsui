"""
Shared log-probability utility for PAPPO training.

Only `compute_log_prob` survives. The reference-anchor / hidden-state /
average-log-prob helpers belonged to the removed DPO / HyPO / beta-DPO /
SimPO loss variants; PAPPO reads its aggregated reference margin from the
precompute records and only needs the policy log-probs of y_w / y_l.
"""

import torch
import torch.nn.functional as F


def compute_log_prob(model, tokenizer, prompt, completion, cutoff_len, no_grad=True):
    """
    Compute log-probability of a completion given a prompt under the model.

    Concatenates prompt + completion, runs a single forward pass, and sums
    the log-softmax scores over only the completion tokens.

    Args:
        model: HuggingFace model or PeftModel (in eval mode for reference models)
        tokenizer: Tokenizer instance
        prompt: Prompt string (e.g. paper title for citation)
        completion: Completion string (e.g. the citation output)
        cutoff_len: Maximum total sequence length
        no_grad: If True (default), run under torch.no_grad() and return a Python float.
                 If False, preserve gradients and return a 0-dimensional tensor.

    Returns:
        Sum of log-probabilities over completion tokens (float if no_grad=True, tensor otherwise),
        or -1e9 if completion is empty
    """
    full_text = prompt + completion
    full_ids = tokenizer(
        full_text, truncation=True, max_length=cutoff_len,
        return_tensors="pt",
    ).input_ids.to(model.device)
    prompt_ids = tokenizer(
        prompt, truncation=True, max_length=cutoff_len,
        return_tensors="pt",
    ).input_ids.to(model.device)

    prompt_len = prompt_ids.shape[1]
    completion_len = full_ids.shape[1] - prompt_len

    if completion_len <= 0:
        full_text = prompt + " " + completion
        full_ids = tokenizer(
            full_text, truncation=True, max_length=cutoff_len,
            return_tensors="pt",
        ).input_ids.to(model.device)
        prompt_len = prompt_ids.shape[1]
        completion_len = full_ids.shape[1] - prompt_len

    if completion_len <= 0:
        return torch.tensor(-1e9, device=model.device) if not no_grad else -1e9

    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        outputs = model(full_ids)
        logits = outputs.logits
        log_probs_all = F.log_softmax(logits, dim=-1)
        completion_log_probs = log_probs_all[0, prompt_len - 1:full_ids.shape[1] - 1, :]
        target_ids = full_ids[0, prompt_len:]
        token_log_probs = completion_log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
        total = token_log_probs.sum()

    if no_grad:
        return total.item()
    return total
