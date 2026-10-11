#!/usr/bin/env python3
"""
Pass 0 (tweet_paraphrase only): formalize each real tweet into formal English.

Uses the UNTRAINED base llama (no TaskLoRA, no anchor LoRA) loaded in FULL
precision (bf16, NOT quantized) purely for inference rewriting, following the
`tweet_formalize.json` prompt. The formalized text becomes the query input `x`
for the tweet task; the raw tweet stays as `y_w`.

Output: PAPPO_record/data/tweet_paraphrase/<out_subdir>/formalized_x.json
    { qid: {user_id, entry_idx, raw_tweet, formal_x} }
Supports resume: already-present qids are skipped.

Usage:
    python formalize_tweets.py --gpu_id 0
    python formalize_tweets.py --gpu_id 0 --max_users 2 --batch_size 8
"""

import os
import sys

# --- env MUST be set before importing torch (mirror sft_target_user.py L15-22) ---
_gpu_id = 0
for i, arg in enumerate(sys.argv):
    if arg == '--gpu_id' and i + 1 < len(sys.argv):
        _gpu_id = sys.argv[i + 1]
        break
os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_id)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_ROOT = Path(__file__).resolve().parent.parent  # PAPPO/
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.pipeline.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_PROMPT_DIR,
    load_json,
    save_json,
    out_dir_for,
)
from src.generation.generator import batch_generate_raw
from src.models.loader import download_model_from_modelscope
from src.utils.text_utils import split_batch


def parse_arguments():
    p = argparse.ArgumentParser(description="Pass 0: formalize tweets (tweet_paraphrase only)")
    p.add_argument('--task_name', type=str, default='tweet_paraphrase',
                   help='Only tweet_paraphrase is supported by Pass 0')
    p.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    p.add_argument('--base_model', type=str,
                   default='modelscope://LLM-Research/Meta-Llama-3-8B',
                   help='Untrained base model (modelscope:// URI resolved at runtime)')
    p.add_argument('--batch_size', type=int, default=8, help='Generation batch size')
    p.add_argument('--max_new_tokens', type=int, default=128, help='Max new tokens per rewrite')
    p.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument('--prompt_dir', type=str, default=str(DEFAULT_PROMPT_DIR))
    p.add_argument('--out_subdir', type=str, default='pappo_anchor',
                   help='Product subdir under <data_dir>/<task_name>/')
    p.add_argument('--max_users', type=int, default=None, help='Limit test users (debug)')
    p.add_argument('--save_every', type=int, default=20,
                   help='Flush formalized_x.json every N batches (resume safety)')
    return p.parse_args()


def clean_formal(text: str) -> str:
    """Extract content after the last 'Formal:' marker (see tweet_formalize.json)."""
    if "Formal:" in text:
        text = text.split("Formal:")[-1]
    return text.strip()


def collect_targets(task_name, data_dir, out_dir, max_users):
    """
    Build the list of (qid, user_id, entry_idx, raw_tweet) to formalize,
    skipping qids already present in an existing formalized_x.json (resume).
    """
    test_path = Path(data_dir) / task_name / "test_100" / "user_test_100.json"
    test_users = load_json(str(test_path))
    if max_users is not None:
        test_users = test_users[:max_users]

    out_path = out_dir / "formalized_x.json"
    existing = {}
    if out_path.exists():
        existing = load_json(str(out_path))
        print(f"[resume] loaded {len(existing)} existing formalized entries")

    targets = []
    for user in test_users:
        user_id = str(user.get("user_id", "unknown"))
        profile = user.get("profile", []) or []
        for entry_idx, entry in enumerate(profile):
            qid = f"{user_id}::{entry_idx}"
            if qid in existing:
                continue
            raw = str(entry.get("text", "") or "").strip()
            if not raw:
                continue
            targets.append((qid, user_id, entry_idx, raw))
    return targets, existing, out_path


def main():
    args = parse_arguments()
    if args.task_name != 'tweet_paraphrase':
        raise SystemExit("Pass 0 only applies to --task_name tweet_paraphrase")

    print(f"Using GPU: {args.gpu_id} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')})")

    out_dir = out_dir_for(args.data_dir, args.task_name, args.out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets, formalized, out_path = collect_targets(
        args.task_name, args.data_dir, out_dir, args.max_users
    )
    print(f"Targets to formalize: {len(targets)} (already done: {len(formalized)})")
    if not targets:
        save_json(formalized, out_path)
        print("Nothing to do. Done.")
        return

    # --- Load UNTRAINED base llama in FULL precision bf16 (no quantization) ---
    model_path = download_model_from_modelscope(args.base_model)
    print(f"Loading base model (bf16, no quant): {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left", use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    print("Base model loaded.")

    formalize_prompt = load_json(str(Path(args.prompt_dir) / "tweet_formalize.json"))["formalize_prompt"]
    gen_params = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "repetition_penalty": None,
    }

    from tqdm import tqdm
    batches = split_batch(targets, args.batch_size)
    for bi, batch in enumerate(tqdm(batches, desc="Formalize")):
        prompts = [formalize_prompt.format(raw) for (_, _, _, raw) in batch]
        raw_outs = batch_generate_raw(model, tokenizer, prompts, gen_params, args.batch_size)
        for (qid, user_id, entry_idx, raw), out in zip(batch, raw_outs):
            formal_x = clean_formal(out)
            if not formal_x:
                # Fall back to the raw tweet so the query is never empty.
                formal_x = raw
            formalized[qid] = {
                "user_id": user_id,
                "entry_idx": entry_idx,
                "raw_tweet": raw,
                "formal_x": formal_x,
            }
        if (bi + 1) % args.save_every == 0:
            save_json(formalized, out_path)

    save_json(formalized, out_path)
    print(f"Done. Wrote {len(formalized)} entries -> {out_path}")


if __name__ == "__main__":
    main()
