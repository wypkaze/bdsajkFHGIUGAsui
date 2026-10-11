#!/usr/bin/env python3
"""
PAPPO Task LoRA Training Script

This script trains a personalized LoRA model on LAMP benchmark tasks.
It supports:
- Multiple tasks (citation, movie_tagging, news_headline, etc.)
- BM25-based history retrieval
- Optional user profile integration
- 4-bit quantization for efficient training

Usage:
    python task_LoRA.py --task_name citation --k 1 --max_epoch 3
    python task_LoRA.py --task_name movie_tagging --k 0 --add_profile
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# Add src to path for imports
_ROOT = Path(__file__).resolve().parent.parent  # PAPPO/
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.models import load_model, create_lora_config, apply_lora_to_model
from src.data import (
    load_data_files,
    process_train_data,
    process_test_data,
    create_dataset,
)
from src.utils import (
    print_trainable_parameters,
    split_batch,
    name2taskid,
)


def get_experiment_name(args: argparse.Namespace, k_value: int = None) -> str:
    """Generate experiment name based on arguments.
    
    Args:
        args: Command line arguments
        k_value: Optional k value for multi-k inference (if None, use args.k)
    """
    k = k_value if k_value is not None else args.k
    profile_suffix = "-profile" if args.add_profile else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"k{k}-{args.task_name}{profile_suffix}-{timestamp}"


def get_code_name() -> str:
    """Get the code name for directory structure."""
    return "task_LoRA"


def get_middle_dir(args: argparse.Namespace) -> str:
    """Get middle directory name: {task_name}-{code_name}
    
    Args:
        args: Command line arguments
    
    Returns:
        Middle directory name
    """
    code_name = get_code_name()
    return f"{args.task_name}-{code_name}"


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """
    Setup logging to file and console.
    
    Args:
        log_dir: Directory to save log files
        experiment_name: Name of the experiment
    
    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    # Create logger
    logger = logging.getLogger("pappo_training")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to: {log_file}")
    return logger


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PAPPO Task LoRA Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model arguments
    # Base model. Alternatives:
    #   - "meta-llama/Meta-Llama-3-8B"          (default, gated on HF Hub: pass --access_token)
    #   - "mistralai/Mistral-7B-Instruct-v0.3"  (Mistral 7B instruct, 128k ctx)
    #   - "mistralai/Mistral-7B-Instruct-v0.2"  (Mistral 7B instruct, 32k ctx)
    # Native HF Hub IDs are resolved by `transformers.from_pretrained`; prefix with
    # "huggingface://" to force an explicit snapshot_download first.
    parser.add_argument(
        '--model_name', 
        type=str, 
        default='meta-llama/Meta-Llama-3-8B',
        help='Model name or path (native HF Hub ID or huggingface:// URI). '
             'Alternative: mistralai/Mistral-7B-Instruct-v0.3'
    )
    parser.add_argument(
        '--access_token', 
        type=str, 
        default=None,
        help='HuggingFace access token'
    )
    
    # Device arguments
    parser.add_argument(
        '--gpu_id', 
        type=int, 
        default=0,
        help='GPU device ID to use (e.g., 0, 1, 2)'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=2,
        help='Training batch size per device'
    )
    parser.add_argument(
        '--max_epoch', 
        type=int, 
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--cut_off', 
        type=int, 
        default=1024,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=1e-4,
        help='Learning rate'
    )
    
    # Task arguments
    parser.add_argument(
        '--task_name', 
        type=str, 
        default='citation',
        choices=['citation', 'movie_tagging', 'news_headline', 
                 'product_rating', 'scholarly_title', 'tweet_paraphrase'],
        help='Task name'
    )
    parser.add_argument(
        '--k', 
        type=int, 
        default=1,
        help='Number of retrieved history items for training (0=disabled)'
    )
    parser.add_argument(
        '--k_list',
        type=str,
        default='0,1,2,4',
        help='Comma-separated list of k values for inference (e.g., "0,1,2,4", 0=kNP)'
    )
    parser.add_argument(
        '--add_profile', 
        action='store_true',
        default = True,
        help='Add user profile to prompt'
    )
    
    # Inference arguments
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.1,
        help='Generation temperature'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=str(_ROOT / 'outputs-task_LoRA'),
        help='Output directory for training logs'
    )
    parser.add_argument(
        '--ckpt_dir', 
        type=str, 
        default=str(_ROOT / 'ckpt-task_LoRA'),
        help='Checkpoint save directory'
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default=str(_ROOT / 'data'),
        help='Data directory'
    )
    parser.add_argument(
        '--prompt_dir', 
        type=str, 
        default=str(_ROOT / 'prompt'),
        help='Prompt template directory'
    )
    
    return parser.parse_args()


def setup_training_args(args: argparse.Namespace) -> TrainingArguments:
    """Create TrainingArguments from parsed args."""
    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        optim='adamw_torch',
        num_train_epochs=args.max_epoch,
        save_steps=1e9,  # Disable auto-saving during training
        logging_steps=50,
        learning_rate=args.learning_rate,
        weight_decay=1e-2,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        group_by_length=True,
        lr_scheduler_type='linear',
        report_to='none',
    )


def train_model(
    model,
    tokenizer,
    train_dataset,
    training_args: TrainingArguments,
) -> Trainer:
    """
    Train the model.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        training_args: Training arguments
    
    Returns:
        Trained trainer
    """
    # Prepare data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
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
    
    # Train
    model.config.use_cache = False
    trainer.train()
    
    return trainer


def get_model_save_path(args: argparse.Namespace, experiment_name: str) -> str:
    """
    Get the path where model should be saved.
    Three-layer structure: {ckpt_dir}/{task_name-code_name}/{experiment_name}_ckpt
    
    Args:
        args: Command line arguments
        experiment_name: Name of the experiment
    
    Returns:
        Path where model should be saved
    """
    middle_dir = get_middle_dir(args)
    return f"{args.ckpt_dir}/{middle_dir}/{experiment_name}_ckpt"


def save_model(model, args: argparse.Namespace, experiment_name: str, logger: logging.Logger) -> str:
    """
    Save the trained model.
    
    Args:
        model: Model to save
        args: Command line arguments
        experiment_name: Name of the experiment
        logger: Logger instance
    
    Returns:
        Path where model was saved
    """
    output_name = get_model_save_path(args, experiment_name)
    
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    model.save_pretrained(output_name)
    logger.info(f"Model saved to {output_name}")
    
    return output_name


def run_inference(
    model,
    tokenizer,
    test_prompts: list[tuple[str, str]],
    args: argparse.Namespace,
) -> list[dict]:
    """
    Run inference on test data.
    
    Args:
        model: Model to use
        tokenizer: Tokenizer
        test_prompts: List of (question_id, prompt) tuples
        args: Command line arguments
    
    Returns:
        List of predictions
    """
    model.eval()
    model.config.use_cache = True
    
    pred_all = []
    
    # Process in batches
    batch_size = 1
    test_batches = split_batch(test_prompts, batch_size)
    
    for batch in tqdm(test_batches, desc="Running inference"):
        question_ids, prompts = zip(*batch)
        
        # Tokenize
        inputs = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        )
        inputs = inputs.to(model.device)
        
        # Generate
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    top_k=10,
                    temperature=args.temperature,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=200,
                )
        
        # Decode only the generated tokens (not the input prompt)
        # outputs contains input_ids + generated_ids, we only want generated part
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for i, output in enumerate(decoded):
            pred_all.append({
                "id": question_ids[i],
                "output": output.strip(),
            })
    
    return pred_all


def save_results(
    predictions: list[dict], 
    args: argparse.Namespace, 
    experiment_name: str,
    k_name: str,
    logger: logging.Logger
) -> str:
    """
    Save inference results.
    Three-layer structure: {output_dir}/{task_name-code_name}/output-{k_name}.json
    
    Args:
        predictions: List of predictions
        args: Command line arguments
        experiment_name: Name of the experiment
        k_name: Name for k value (e.g., 'k1', 'k2', 'kNP')
        logger: Logger instance
    
    Returns:
        Path where results were saved
    """
    output_file = {
        'task': name2taskid.get(args.task_name, args.task_name),
        'golds': predictions,
        'model': args.model_name,
        'experiment': experiment_name,
    }
    
    middle_dir = get_middle_dir(args)
    output_path = (
        f'{args.output_dir}/{middle_dir}/'
        f'output-{k_name}.json'
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_file, f, indent=4)
    
    logger.info(f"Results saved to {output_path}")
    return output_path


def main():
    """Main training and inference pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Set GPU device (must be done before any CUDA operations)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"Using GPU: {args.gpu_id}")
    
    # Generate experiment name (use training k value)
    experiment_name = get_experiment_name(args)
    
    # Setup logging - log file will be in the same directory as model
    model_save_dir = get_model_save_path(args, experiment_name)
    log_dir = os.path.dirname(model_save_dir)
    logger = setup_logging(log_dir, experiment_name)
    
    logger.info(f"Using GPU: {args.gpu_id}")
    
    # Parse k_list for inference (after logger is initialized)
    k_list = [int(k.strip()) for k in args.k_list.split(',')]
    logger.info(f"Inference k_list: {k_list}")
    
    logger.info(f"Task: {args.task_name}, k={args.k}, epochs={args.max_epoch}, "
                f"batch_size={args.batch_size}")
    logger.info(f"Experiment: {experiment_name}")
    
    # ========== 1. Load Model ==========
    logger.info("Loading tokenizer and model...")
    tokenizer, base_model = load_model(
        model_name=args.model_name,
        access_token=args.access_token,
        use_4bit=True,
        gpu_id=args.gpu_id,
    )
    
    # Apply LoRA
    lora_config = create_lora_config()
    model = apply_lora_to_model(base_model, lora_config, prepare_for_kbit=True)
    print_trainable_parameters(model)
    
    # ========== 2. Load Data ==========
    logger.info("Loading data...")
    paths, prompt_template, train_data, test_data, all_profile = load_data_files(
        task_name=args.task_name,
        data_dir=args.data_dir,
        prompt_dir=args.prompt_dir,
    )
    
    if args.add_profile:
        logger.info(f"Loaded {len(all_profile)} user profiles")
    
    # ========== 3. Process Training Data ==========
    logger.info("Processing training data...")
    processed_train_data = process_train_data(
        train_data=train_data,
        prompt_template=prompt_template,
        all_profile=all_profile,
        task_name=args.task_name,
        k=args.k,
        add_profile=args.add_profile,
    )
    logger.info(f"Total training samples: {len(processed_train_data)}")
    
    # ========== 4. Create Dataset & Train ==========
    logger.info("Creating dataset and starting training...")
    train_dataset = create_dataset(
        processed_data=processed_train_data,
        tokenizer=tokenizer,
        cutoff_len=args.cut_off,
        shuffle=True,
    )
    
    training_args = setup_training_args(args)
    trainer = train_model(model, tokenizer, train_dataset, training_args)
    
    # ========== 5. Save Model ==========
    save_model(model, args, experiment_name, logger)
    
    # ========== 6. Multi-k Inference ==========
    logger.info(f"Running inference with k values: {k_list}")
    
    for k in k_list:
        logger.info(f"Processing inference with k={k}...")
        
        # Process test data with current k
        test_prompts = process_test_data(
            test_data=test_data,
            prompt_template=prompt_template,
            all_profile=all_profile,
            task_name=args.task_name,
            k=k,
            add_profile=args.add_profile,
        )
        
        # Run inference
        predictions = run_inference(model, tokenizer, test_prompts, args)
        
        # Generate k_name for saving
        if k == 0:
            k_name = "kNP"
        else:
            k_name = f"k{k}"
        
        # Save results for this k
        save_results(predictions, args, experiment_name, k_name, logger)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
