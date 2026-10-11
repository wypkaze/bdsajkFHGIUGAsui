"""
Batched generation for Pass B (y_l) and Pass 0 (tweet formalization).

Reuses:
  - the BM25 anchor-history retrieval + prompt formatting logic from
    `src.training.pair_generator.PairGenerator.build_pair_prompt` (refactored so
    each anchor builds its BM25 index ONCE and reuses it across all queries);
  - the generation parameters from `PairGenerator._do_generate` (passed in via
    TASK_CONFIG[task]["gen"]);
  - the output cleaning from `src.utils.clean_output.extract_real_output`, plus
    the citation special-case (`_post_process_citation`).

Batched generation uses LEFT padding (tokenizer.padding_side == "left"), and the
full sequence (prompt + generated) is decoded before cleaning so that
extract_real_output behaves exactly like the single-sample reference path.
"""

from typing import Dict, List, Optional, Sequence

import torch
from rank_bm25 import BM25Okapi

from ..utils.clean_output import extract_real_output
from ..utils.text_utils import get_first_k_tokens, split_batch


class AnchorPromptBuilder:
    """
    Builds `prompt_pair.json`-style prompts for ONE anchor across many queries.

    The anchor's history lines and BM25 index are constructed once in the
    constructor; `build(x_text)` then only performs the per-query retrieval.
    This mirrors PairGenerator.build_pair_prompt but caches the index per anchor.
    """

    def __init__(
        self,
        anchor_user: Dict,
        task_name: str,
        pair_prompt_template: Dict,
        k: int = 1,
        trunc_tokens: int = 368,
    ):
        self.task_name = task_name
        self.k = k
        pair_tmpl = pair_prompt_template[task_name]
        self.retrieval_fmt = pair_tmpl["retrieval_history"]
        self.profile_prompt = pair_tmpl["profile_prompt"]

        anchor_profile = anchor_user.get("profile", []) or []

        history_list: List[str] = []
        for p in anchor_profile:
            # Truncate each field (do NOT mutate the source dict).
            trunc = {
                kk: (get_first_k_tokens(vv, trunc_tokens) if isinstance(vv, str) else vv)
                for kk, vv in p.items()
            }
            try:
                history_list.append(self.retrieval_fmt.format(**trunc))
            except (KeyError, IndexError):
                continue

        self.history_list = history_list
        self.bm25 = BM25Okapi([doc.split(" ") for doc in history_list]) if history_list else None

    def build(self, x_text: str) -> str:
        """Build the prompt for a single query input `x_text`."""
        x_text = str(x_text)
        if not self.history_list or self.bm25 is None:
            return self.profile_prompt.format("", x_text)
        retrieved = self.bm25.get_top_n(x_text.split(" "), self.history_list, n=self.k)
        history_string = "".join(retrieved)
        return self.profile_prompt.format(history_string, x_text)


def clean_generation(text: str, task_name: str, prompt_template: Dict) -> str:
    """
    Clean a raw decoded generation, mirroring PairGenerator._do_generate.

    citation is handled specially (split on "Answer:", strip leading
    "Paper Title:"); all other tasks defer to extract_real_output.
    """
    if task_name == "citation":
        if "Answer:" in text:
            text = text.split("Answer:")[-1]
        text = text.strip()
        if text.lower().startswith("paper title:"):
            text = text[len("paper title:"):].strip()
        return text
    return extract_real_output(text, task_name, prompt_template)


@torch.no_grad()
def batch_generate_raw(
    model,
    tokenizer,
    prompts: Sequence[str],
    gen_params: Dict,
    batch_size: int = 8,
) -> List[str]:
    """
    Batched generation returning the FULL decoded text (prompt + completion).

    gen_params keys: max_new_tokens, do_sample, temperature, top_p, top_k,
    repetition_penalty (any may be None). Matches _do_generate's generate() call.
    """
    model.config.use_cache = True
    model.eval()

    results: List[str] = []
    for batch in split_batch(list(prompts), batch_size):
        inputs = tokenizer(list(batch), return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_params.get("max_new_tokens", 64),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=gen_params.get("do_sample", False),
            top_p=gen_params.get("top_p"),
            top_k=gen_params.get("top_k"),
            temperature=gen_params.get("temperature"),
            repetition_penalty=gen_params.get("repetition_penalty"),
        )
        # Decode the whole sequence (pads are special tokens -> skipped), so the
        # downstream marker-based cleaning sees the same text as the reference.
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(texts)
    return results


def generate_batch(
    model,
    tokenizer,
    prompts: Sequence[str],
    task_name: str,
    gen_params: Dict,
    prompt_template: Dict,
    batch_size: int = 8,
) -> List[str]:
    """Batch-generate then clean each output for a pair-generation task."""
    raw = batch_generate_raw(model, tokenizer, prompts, gen_params, batch_size)
    return [clean_generation(t, task_name, prompt_template) for t in raw]
