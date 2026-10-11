#!/usr/bin/env python3
"""
Pass 0.5 (tweet_paraphrase only): de-emotionalize the already-formalized text.

Second processing stage. It takes the `formal_x` produced by Pass 0
(`formalize_tweets.py` -> formalized_x.json) as INPUT and rewrites only the
emotional / expressive parts (interjections, exclamation marks, "lol", emphatic
ALL-CAPS, emotional intensifiers) into formal, neutral expressions, while
keeping every factual detail unchanged, preserving @usernames verbatim, and
never inventing information. See `tweet_detone.json`.

Uses the same UNTRAINED base llama (no TaskLoRA, no anchor LoRA) in FULL
precision (bf16, NOT quantized) for inference rewriting.

Input : PAPPO_record/data/tweet_paraphrase/<out_subdir>/formalized_x.json
Output: PAPPO_record/data/tweet_paraphrase/<out_subdir>/detoned_x.json
    { qid: {user_id, entry_idx, raw_tweet, formal_x, detoned_x} }
Supports resume: qids already present in detoned_x.json are skipped.

Usage:
    python detone_tweets.py --gpu_id 0
    python detone_tweets.py --gpu_id 0 --max_users 2 --batch_size 8
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
from src.models.loader import download_model_from_huggingface
from src.utils.text_utils import split_batch


def parse_arguments():
    p = argparse.ArgumentParser(description="Pass 0.5: de-emotionalize formalized tweets")
    p.add_argument('--task_name', type=str, default='tweet_paraphrase',
                   help='Only tweet_paraphrase is supported')
    p.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    # Base model. Alternatives:
    #   - "meta-llama/Meta-Llama-3-8B"          (default, gated on HF Hub)
    #   - "mistralai/Mistral-7B-Instruct-v0.3"  (Mistral 7B instruct, 128k ctx)
    #   - "mistralai/Mistral-7B-Instruct-v0.2"  (Mistral 7B instruct, 32k ctx)
    # Native HF Hub IDs are resolved by `transformers.from_pretrained`; prefix with
    # "huggingface://" to force an explicit snapshot_download first.
    p.add_argument('--base_model', type=str,
                   default='meta-llama/Meta-Llama-3-8B',
                   help='Untrained base model (native HF Hub ID or huggingface:// URI). '
                        'Alternative: mistralai/Mistral-7B-Instruct-v0.3')
    p.add_argument('--batch_size', type=int, default=8, help='Generation batch size')
    p.add_argument('--max_new_tokens', type=int, default=128, help='Max new tokens per rewrite')
    p.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument('--prompt_dir', type=str, default=str(DEFAULT_PROMPT_DIR))
    p.add_argument('--out_subdir', type=str, default='pappo_anchor',
                   help='Product subdir under <data_dir>/<task_name>/')
    p.add_argument('--in_file', type=str, default='formalized_x.json',
                   help='Pass 0 product used as the input (formal_x field)')
    p.add_argument('--out_file', type=str, default='detoned_x.json',
                   help='Second-pass product written under out_subdir')
    p.add_argument('--max_users', type=int, default=None, help='Limit distinct users (debug)')
    p.add_argument('--save_every', type=int, default=20,
                   help='Flush output every N batches (resume safety)')
    return p.parse_args()


def clean_formal(text: str) -> str:
    """Extract content after the last 'Formal:' marker (see tweet_detone.json)."""
    if "Formal:" in text:
        text = text.split("Formal:")[-1]
    return text.strip()


def collect_targets(in_path, out_path, max_users):
    """
    Build the list of (qid, user_id, entry_idx, raw_tweet, formal_x) to detone
    from the Pass 0 product, skipping qids already present in the output (resume).
    """
    if not Path(in_path).exists():
        raise SystemExit(
            f"Input not found: {in_path}\nRun Pass 0 (formalize_tweets.py) first."
        )
    inp = load_json(str(in_path))  # {qid: {user_id, entry_idx, raw_tweet, formal_x}}

    existing = {}
    if Path(out_path).exists():
        existing = load_json(str(out_path))
        print(f"[resume] loaded {len(existing)} existing detoned entries")

    targets = []
    seen_users = []
    for qid, rec in inp.items():
        if qid in existing:
            continue
        formal_x = str(rec.get("formal_x", "") or "").strip()
        if not formal_x:
            continue
        user_id = str(rec.get("user_id", "unknown"))
        if max_users is not None:
            if user_id not in seen_users:
                if len(seen_users) >= max_users:
                    continue
                seen_users.append(user_id)
        targets.append((qid, user_id, rec.get("entry_idx"),
                        str(rec.get("raw_tweet", "") or ""), formal_x))
    return targets, existing


def main():
    args = parse_arguments()
    if args.task_name != 'tweet_paraphrase':
        raise SystemExit("Pass 0.5 only applies to --task_name tweet_paraphrase")

    print(f"Using GPU: {args.gpu_id} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')})")

    out_dir = out_dir_for(args.data_dir, args.task_name, args.out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_path = out_dir / args.in_file
    out_path = out_dir / args.out_file

    targets, detoned = collect_targets(in_path, out_path, args.max_users)
    print(f"Input : {in_path}")
    print(f"Targets to detone: {len(targets)} (already done: {len(detoned)})")
    if not targets:
        save_json(detoned, out_path)
        print("Nothing to do. Done.")
        return

    # --- Load UNTRAINED base llama in FULL precision bf16 (no quantization) ---
    model_path = download_model_from_huggingface(args.base_model)
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

    detone_prompt = load_json(str(Path(args.prompt_dir) / "tweet_detone.json"))["detone_prompt"]
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
    for bi, batch in enumerate(tqdm(batches, desc="Detone")):
        prompts = [detone_prompt.format(formal_x) for (_, _, _, _, formal_x) in batch]
        raw_outs = batch_generate_raw(model, tokenizer, prompts, gen_params, args.batch_size)
        for (qid, user_id, entry_idx, raw_tweet, formal_x), out in zip(batch, raw_outs):
            detoned_x = clean_formal(out)
            if not detoned_x:
                # Fall back to the Pass 0 text so the query is never empty.
                detoned_x = formal_x
            detoned[qid] = {
                "user_id": user_id,
                "entry_idx": entry_idx,
                "raw_tweet": raw_tweet,
                "formal_x": formal_x,
                "detoned_x": detoned_x,
            }
        if (bi + 1) % args.save_every == 0:
            save_json(detoned, out_path)

    save_json(detoned, out_path)
    print(f"Done. Wrote {len(detoned)} entries -> {out_path}")


if __name__ == "__main__":
    main()
