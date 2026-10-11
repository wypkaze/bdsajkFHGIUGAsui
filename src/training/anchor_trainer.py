"""
Anchor user training utilities.

Provides AnchorTrainer class for training personalized UserLoRA
on top of TaskLoRA-merged base model.
"""

import os
import gc
import argparse
from typing import List, Dict, Optional

import torch
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import get_peft_model

from ..data.processor import DataProcessor
from ..data.dataset import create_dataset
from ..models.lora_config import create_lora_config
from ..utils.model_utils import print_trainable_parameters


class AnchorTrainer:
    """
    Trainer for anchor user personalized LoRA.

    Architecture:
    - Merged Model (TaskLoRA merged into base, 4-bit quantized)
    - UserLoRA (created via get_peft_model per user, trainable)

    Each user gets a fresh PeftModel. The base merged model is preserved.
    """

    def __init__(
        self,
        model_with_task_lora: torch.nn.Module,
        tokenizer,
        args: argparse.Namespace,
        prompt_template: Dict,
        all_profile: Dict[str, str],
    ):
        """
        Initialize anchor trainer.

        Args:
            model_with_task_lora: Regular model with TaskLoRA already merged in (4-bit).
            tokenizer: Tokenizer instance
            args: Training arguments namespace
            prompt_template: Prompt template dictionary
            all_profile: Dictionary of user profiles
        """
        self.model_with_task_lora = model_with_task_lora
        self.tokenizer = tokenizer
        self.args = args
        self.prompt_template = prompt_template
        self.all_profile = all_profile

        # Initialize data processor
        self.processor = DataProcessor(
            task_name=args.task_name,
            prompt_template=prompt_template,
            k=args.k,
            add_profile=args.add_profile,
        )

    def process_anchor_train_data(self, anchor_user: Dict) -> List[Dict]:
        """
        Process training data for an anchor user.

        For each anchor user, uses their history records to build training samples.
        Skips the first record (used for validation), uses the rest for training.

        Args:
            anchor_user: Anchor user data with 'user_id', 'profile', 'query'

        Returns:
            List of training data samples
        """
        train_data = []
        user_profile_list = anchor_user.get("profile", [])
        user_id = str(anchor_user.get("user_id", ""))
        user_profile_text = self.all_profile.get(user_id, "") if self.args.add_profile else ""

        # Determine if task needs special formatting
        format_flag = self.args.task_name in [
            "movie_tagging",
            "news_headline",
            "product_rating",
            "scholarly_title",
        ]

        # Iterate through user history records
        for idx, item in enumerate(user_profile_list):
            # Skip first record (used for validation)
            if idx == 0:
                continue

            # Build base prompts
            prompt = self.prompt_template[self.args.task_name]["OPPU_input"].format(**item)
            full_prompt = self.prompt_template[self.args.task_name]["OPPU_full"].format(**item)

            # Use BM25 to retrieve history (if enabled)
            if format_flag and self.args.k > 0:
                visible_history = user_profile_list[:idx]

                # Format history records
                history_list = [
                    self.prompt_template[self.args.task_name]["retrieval_history"].format(**h)
                    for h in visible_history
                ]

                if history_list:
                    # Build BM25 index
                    from rank_bm25 import BM25Okapi

                    tokenized_corpus = [doc.split(" ") for doc in history_list]
                    bm25 = BM25Okapi(tokenized_corpus)

                    # Retrieve relevant history
                    tokenized_query = (
                        self.prompt_template[self.args.task_name]["retrieval_query"]
                        .format(**item)
                        .split(" ")
                    )
                    retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=self.args.k)
                    history_string = "".join(retrieved_history)

                    # Add history to prompts
                    prompt = f"### User History:\n{history_string}\n\n" + prompt
                    full_prompt = f"### User History:\n{history_string}\n\n" + full_prompt

            # Add user profile
            if self.args.add_profile and user_profile_text:
                prompt = f"### User Profile:\n{user_profile_text}\n\n" + prompt
                full_prompt = f"### User Profile:\n{user_profile_text}\n\n" + full_prompt

            train_data.append({"prompt": prompt, "full_prompt": full_prompt})

        return train_data

    def train_single_user(
        self,
        anchor_user: Dict,
        output_dir: str,
    ) -> Optional[str]:
        """
        Train UserLoRA for a single anchor user.

        Architecture:
        - MergedModel (TaskLoRA merged in, 4-bit quantized)
        - UserLoRA (created via get_peft_model() per user, trainable)
        Each user gets a fresh PeftModel. The base merged model is preserved.

        Args:
            anchor_user: Anchor user data dictionary
            output_dir: Output directory path for saving UserLoRA

        Returns:
            Path where UserLoRA was saved, or None if training failed
        """
        user_id = str(anchor_user.get("user_id", "unknown"))

        print(f"\n{'='*50}")
        print(f"  Training UserLoRA for user {user_id}")
        print(f"{'='*50}")

        # Process training data
        train_data = self.process_anchor_train_data(anchor_user)
        if not train_data:
            print(f"Warning: No training data for user {user_id}, skipping...")
            return None

        print(f"  Training samples: {len(train_data)}")

        # Create dataset
        print("  Step 1: Creating dataset...")
        train_dataset = create_dataset(
            train_data,
            self.tokenizer,
            cutoff_len=self.args.cut_off,
            shuffle=True,
        )

        # Create UserLoRA config
        print("  Step 2: Creating UserLoRA config...")
        user_lora_config = create_lora_config(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        )

        # Create UserLoRA via get_peft_model
        print("  Step 3: Setting up UserLoRA...")
        # get_peft_model() freezes the base model and only makes LoRA params trainable.
        model = get_peft_model(self.model_with_task_lora, user_lora_config)
        print(f"  Created UserLoRA via get_peft_model")

        # Verify base model params are frozen
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        base_trainable = [n for n in trainable_names if "lora_" not in n]
        if base_trainable:
            raise ValueError(f"ERROR: Base model params are trainable: {base_trainable}")
        print(f"  Trainable UserLoRA params: {len(trainable_names)}")

        # Print trainable parameters
        print_trainable_parameters(model)

        # Setup output directory
        user_output_dir = os.path.join(output_dir, f"user_{user_id}_anchor_lora")
        os.makedirs(user_output_dir, exist_ok=True)

        # Setup training arguments
        print("  Step 4: Setting up training...")
        training_args = TrainingArguments(
            output_dir=user_output_dir,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=1,
            optim="adamw_torch",
            num_train_epochs=self.args.max_epoch,
            save_steps=1e9,  # Disable auto-save
            logging_steps=10,
            learning_rate=self.args.learning_rate,
            weight_decay=1e-2,
            bf16=True,
            max_grad_norm=0.3,
            warmup_ratio=0.1,
            group_by_length=True,
            lr_scheduler_type="linear",
            report_to="none",
        )

        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,
            data_collator=data_collator,
        )

        # Convert norm layers to float32 for stability
        for name, module in trainer.model.named_modules():
            if "norm" in name:
                module = module.to(torch.float32)

        # Train (only update UserLoRA, TaskLoRA is frozen)
        print("  Step 5: Training UserLoRA...")
        model.config.use_cache = False
        trainer.train()

        # Save trained UserLoRA
        print(f"  Step 6: Saving UserLoRA to {user_output_dir}...")
        model.set_adapter("default")
        model.save_pretrained(user_output_dir)

        print(f"  UserLoRA saved successfully!")

        # Clean up memory: delete UserLoRA PeftModel, keep base model for next user
        print("  Step 7: Cleaning up UserLoRA...")
        del model
        print("  Deleted UserLoRA PeftModel")

        # Cleanup
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        return user_output_dir

    def train_all_users(
        self,
        anchor_users: List[Dict],
        output_dir: str,
        max_users: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Train UserLoRA for all anchor users.

        Args:
            anchor_users: List of anchor user data dictionaries
            output_dir: Output directory path
            max_users: Maximum number of users to train (None=all)

        Returns:
            Dictionary with training statistics
        """
        if max_users is not None:
            anchor_users = anchor_users[:max_users]

        successful = 0
        failed = 0
        skipped = 0

        from tqdm import tqdm

        for idx, anchor_user in enumerate(tqdm(anchor_users, desc="Training anchor users")):
            user_id = str(anchor_user.get("user_id", idx))

            print(f"\nProcessing anchor user {idx + 1}/{len(anchor_users)}: user_id={user_id}")

            try:
                save_path = self.train_single_user(anchor_user, output_dir)
                if save_path:
                    successful += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"Error training user {user_id}: {e}")
                import traceback

                print(traceback.format_exc())
                failed += 1
                continue

        return {
            "total": len(anchor_users),
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
        }
