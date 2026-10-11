"""
Target-user prompt construction shared by precompute and per-user training.

Moved out of `precompute/precompute.py` so the training scripts rebuild the
exact same prompt that the reference anchor log-probs were computed on:
OPPU_input template prefixed with the target user's BM25-retrieved history
built from profile[:entry_idx] (causal), history fields truncated to 368
tokens. Behaviour is byte-identical to the original implementation.
"""

from rank_bm25 import BM25Okapi

from ..utils.text_utils import get_first_k_tokens

# History entries are truncated to this many tokens before formatting, matching
# pappo.py / precompute.py so prompts stay identical across stages.
HISTORY_TRUNC_TOKENS = 368


def format_history_lines(profile_list, retrieval_fmt, trunc_tokens=HISTORY_TRUNC_TOKENS):
    """Pre-format every profile entry once (index-aligned with profile_list).

    A None marks an entry whose retrieval_history formatting failed; those are
    dropped later, exactly like pappo.py's `continue`.
    """
    lines = []
    for h in profile_list:
        h_copy = {
            kk: (get_first_k_tokens(vv, trunc_tokens) if isinstance(vv, str) else vv)
            for kk, vv in h.items()
        }
        try:
            lines.append(retrieval_fmt.format(**h_copy))
        except (KeyError, IndexError):
            lines.append(None)
    return lines


def build_target_prompt(task_name, x_value, visible_lines, prompt_tmpl, k, x_field):
    """OPPU_input(x) prefixed with the BM25-retrieved target-user history."""
    x_value = str(x_value)
    if task_name == "tweet_paraphrase":
        prompt = prompt_tmpl["OPPU_input"] + x_value
    else:
        prompt = prompt_tmpl["OPPU_input"].format(**{x_field: x_value})

    if k > 0 and visible_lines:
        try:
            bm25 = BM25Okapi([doc.split(" ") for doc in visible_lines])
            query = prompt_tmpl.get("retrieval_query", "").format(**{x_field: x_value})
            retrieved = bm25.get_top_n(query.split(" "), visible_lines, n=k)
            prompt = f"### User History:\n{''.join(retrieved)}\n\n" + prompt
        except Exception:
            pass
    return prompt
