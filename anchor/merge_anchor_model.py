#!/usr/bin/env python3
"""
Merge anchor LoRA adapter into the merged_model (base) for all tasks.

Usage:
    python merge_anchor_model.py
    python merge_anchor_model.py --task citation
    GPU_id=0 python merge_anchor_model.py
"""

import os
import sys
import argparse

# CUDA_VISIBLE_DEVICES must be set BEFORE importing torch
gpu_id = os.environ.get("GPU_id", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_save(model_dir: str, lora_dir: str, output_dir: str) -> None:
    """
    Load base model, attach LoRA, merge, and save.

    Args:
        model_dir: Path to the base merged_model directory.
        lora_dir: Path to the anchor_lora_merged directory.
        output_dir: Path to save the merged model.
    """
    print(f"\n{'=' * 60}")
    print(f"Loading base model from: {model_dir}")
    print(f"Loading LoRA adapter from: {lora_dir}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, lora_dir)
    print("LoRA adapter loaded successfully.")

    # Merge and unload
    model = model.merge_and_unload()
    print("LoRA merged successfully.")

    # Save merged model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    print(f"Merged model saved to: {output_dir}")

    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to: {output_dir}")

    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Merge anchor LoRA into merged model")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific task to process (e.g. citation). If not set, processes all tasks.",
    )
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "PAPPO_record", "ckpts", "anchor-train",
        ),
        help="Root directory containing <task>-anchor_train/ checkpoints "
             "(defaults to <repo_root>/PAPPO_record/ckpts/anchor-train).",
    )
    args = parser.parse_args()

    CKPT_ROOT = args.ckpt_root

    TASKS = [
        "citation",
        "movie_tagging",
        "news_headline",
        "product_rating",
        "scholarly_title",
        "tweet_paraphrase",
    ]

    if args.task:
        if args.task not in TASKS:
            print(f"Unknown task: {args.task}. Available: {TASKS}")
            sys.exit(1)
        tasks_to_process = [args.task]
    else:
        tasks_to_process = TASKS

    print(f"GPU_id={gpu_id}, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Tasks to process: {tasks_to_process}")

    for task in tasks_to_process:
        task_dir = os.path.join(CKPT_ROOT, f"{task}-anchor_train")
        model_dir = os.path.join(task_dir, "merged_model")
        lora_dir = os.path.join(task_dir, "anchor_lora_merged")
        output_dir = os.path.join(task_dir, "merged_anchor_model")

        # Skip if output already exists
        if os.path.exists(output_dir) and os.listdir(output_dir):
            print(f"\n[{task}] merged_anchor_model already exists, skipping.")
            continue

        if not os.path.exists(model_dir):
            print(f"\n[{task}] merged_model not found at {model_dir}, skipping.")
            continue

        if not os.path.exists(lora_dir):
            print(f"\n[{task}] anchor_lora_merged not found at {lora_dir}, skipping.")
            continue

        try:
            merge_and_save(model_dir, lora_dir, output_dir)
        except Exception as e:
            print(f"\n[{task}] ERROR: {e}")
            torch.cuda.empty_cache()
            continue

    print(f"\n{'=' * 60}")
    print("All tasks completed.")


if __name__ == "__main__":
    main()