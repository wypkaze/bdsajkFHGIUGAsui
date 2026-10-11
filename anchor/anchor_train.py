#!/usr/bin/env python3
"""
PAPPO Anchor User Personalized Training Script

为每个锚点用户训练个性化的 UserLoRA。

流程:
    Base(bf16) + TaskLoRA → merge_and_unload() → 保存 → 4-bit 加载 → UserLoRA(训练)

Usage:
    python anchor_train.py --task_name citation --task_lora_path "./ckpt-task_LoRA/citation/checkpoint-1000" --k 1
"""

import os
import sys
import gc

# Set CUDA_VISIBLE_DEVICES BEFORE importing torch
# This ensures only the specified GPU is visible to the program
_gpu_id = 0
for i, arg in enumerate(sys.argv):
    if arg == '--gpu_id' and i + 1 < len(sys.argv):
        _gpu_id = sys.argv[i + 1]
        break
os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_id)

import torch
from datetime import datetime
from pathlib import Path
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# Add src to path for importing utilities
_ROOT = Path(__file__).resolve().parent.parent  # PAPPO/
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.utils import (
    setup_logging,
    parse_anchor_train_arguments,
    print_trainable_parameters,
)
from src.data import (
    load_json_file,
    AnchorUserLoader,
)
from src.training import AnchorTrainer


def load_prompt_template(prompt_dir: str) -> dict:
    """Load prompt template from file."""
    prompt_path = Path(prompt_dir) / 'prompt.json'
    print(f"Loading prompt template: {prompt_path}")
    return load_json_file(str(prompt_path))


def main():
    """Main training workflow for anchor users."""
    # Parse arguments
    args = parse_anchor_train_arguments()

    # Note: CUDA_VISIBLE_DEVICES is already set at module import time
    print(f"Using GPU: {args.gpu_id} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")

    # Setup directories
    middle_dir = f"{args.task_name}-anchor_train"
    ckpt_dir = Path(args.ckpt_dir) / middle_dir
    output_dir = Path(args.output_dir) / middle_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    experiment_name = f"{args.task_name}_k{args.k}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logging(str(ckpt_dir), experiment_name)

    logger.info(f"Task: {args.task_name}, k={args.k}, epochs={args.max_epoch}")
    logger.info(f"TaskLoRA path: {args.task_lora_path}")
    logger.info(f"Checkpoint directory: {ckpt_dir}")

    # ========== Step 1: Load tokenizer ==========
    logger.info("Loading tokenizer...")

    # Handle ModelScope download
    model_name = args.model_name
    if model_name.startswith("modelscope://"):
        from modelscope import snapshot_download
        model_name = snapshot_download(model_name.replace("modelscope://", ""))
        logger.info(f"ModelScope model downloaded to: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=args.access_token,
        trust_remote_code=True,
        padding_side="left",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")

    # ========== Step 2: Load base model and TaskLoRA ==========
    logger.info("=" * 60)

    # Create quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ===== Merge TaskLoRA → Save → Reload 4-bit =====
    merged_model_dir = args.merged_model_save_path
    merged_path = Path(merged_model_dir)

    if merged_path.exists() and (merged_path / "config.json").exists():
        logger.info(f"Merged model already exists at: {merged_model_dir}, skipping merge step")
    else:
        logger.info(f"Phase 1: Merging TaskLoRA into base model (full precision)...")
        logger.info(f"TaskLoRA path: {args.task_lora_path}")

        # Load base model in bfloat16 (NO quantization — required for merge_and_unload)
        print(f"\nLoading base model (bf16, no quantization): {model_name}")
        merge_base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        merge_base.config.pad_token_id = tokenizer.pad_token_id

        # Load TaskLoRA on top
        print(f"Loading TaskLoRA: {args.task_lora_path}")
        peft_model = PeftModel.from_pretrained(
            merge_base,
            args.task_lora_path,
            is_trainable=False,
        )

        # Merge and unload (returns a regular model, not PeftModel)
        print("Merging TaskLoRA into base model...")
        merged_model = peft_model.merge_and_unload()

        # Save merged model to disk
        print(f"Saving merged model to: {merged_model_dir}")
        merged_path.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(merged_model_dir)
        tokenizer.save_pretrained(merged_model_dir)

        # Free memory before reloading with 4-bit
        del merge_base, peft_model, merged_model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"Merged model saved to: {merged_model_dir}")

    # Phase 2: Load merged model with 4-bit quantization
    logger.info(f"Phase 2: Loading merged model with 4-bit quantization...")
    print(f"\nLoading merged model (4-bit): {merged_model_dir}")

    model_with_task_lora = AutoModelForCausalLM.from_pretrained(
        merged_model_dir,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model_with_task_lora.config.use_cache = False
    model_with_task_lora.config.pad_token_id = tokenizer.pad_token_id
    print(f"Merged model loaded on {next(model_with_task_lora.parameters()).device}")

    # Enable gradient checkpointing and k-bit training
    model_with_task_lora.gradient_checkpointing_enable()
    model_with_task_lora = prepare_model_for_kbit_training(model_with_task_lora)

    logger.info("Merged model loaded and prepared (4-bit)")

    # ========== Step 3: Load data ==========
    logger.info("Loading anchor users and data...")

    # Load anchor users using AnchorUserLoader
    anchor_loader = AnchorUserLoader(
        task_name=args.task_name,
        data_dir=args.data_dir,
        anchor_dir=args.anchor_dir,
        k=16,  # Default k for anchor selection
        anchor_idx_path=args.anchor_idx_path,
        candidate_path=args.candidate_path,
    )
    anchor_ids, anchor_users = anchor_loader.load_anchor_users()

    # Load prompt template
    prompt_template = load_prompt_template(args.prompt_dir)

    # Load user profiles
    all_profile = anchor_loader.load_user_profiles()

    # Limit number of anchor users
    if args.max_anchor_users is not None:
        anchor_users = anchor_users[:args.max_anchor_users]
        logger.info(f"Limited to first {args.max_anchor_users} anchor users")

    logger.info(f"Total anchor users to train: {len(anchor_users)}")

    # ========== Step 4: Train each anchor user ==========
    logger.info("\n" + "=" * 60)
    logger.info("Starting anchor user training...")
    logger.info("=" * 60)

    # Create anchor trainer
    trainer = AnchorTrainer(
        model_with_task_lora=model_with_task_lora,
        tokenizer=tokenizer,
        args=args,
        prompt_template=prompt_template,
        all_profile=all_profile,
    )

    # Train all users
    stats = trainer.train_all_users(
        anchor_users=anchor_users,
        output_dir=str(ckpt_dir),
        max_users=args.max_anchor_users,
    )

    # ========== Summary ==========
    logger.info("\n" + "=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    logger.info(f"Total anchor users: {stats['total']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Skipped: {stats['skipped']}")
    logger.info(f"Checkpoints saved to: {ckpt_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
