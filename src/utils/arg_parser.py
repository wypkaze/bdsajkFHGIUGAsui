"""
Argument parsing utilities for PAPPO training scripts.

Provides standardized argument parsing with automatic path building.
"""

import argparse
from pathlib import Path


def create_anchor_train_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for anchor user training.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="PAPPO Anchor User Personalized LoRA Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model parameters
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
        help='Base model name or path (native HF Hub ID or huggingface:// URI). '
             'Alternative: mistralai/Mistral-7B-Instruct-v0.3'
    )
    parser.add_argument(
        '--task_lora_path',
        type=str,
        required=True,
        help='TaskLoRA checkpoint directory path'
    )
    parser.add_argument(
        '--access_token',
        type=str,
        default=None,
        help='HuggingFace access token'
    )

    # Device parameters
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='GPU device ID to use'
    )

    # Training parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Per-device training batch size'
    )
    parser.add_argument(
        '--max_epoch',
        type=int,
        default=3,
        help='Number of training epochs per anchor user'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--cut_off',
        type=int,
        default=1024,
        help='Maximum sequence length'
    )

    # LoRA parameters
    parser.add_argument(
        '--lora_r',
        type=int,
        default=8,
        help='LoRA rank (r) for UserLoRA'
    )
    parser.add_argument(
        '--lora_alpha',
        type=int,
        default=8,
        help='LoRA alpha parameter for UserLoRA'
    )

    # Task parameters
    parser.add_argument(
        '--task_name',
        type=str,
        required=True,
        choices=['citation', 'movie_tagging', 'news_headline',
                 'product_rating', 'scholarly_title', 'tweet_paraphrase'],
        help='Task name'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=1,
        help='Number of history items to retrieve using BM25'
    )
    parser.add_argument(
        '--add_profile',
        action='store_true',
        default=False,
        help='Add user profile to prompts'
    )

    # Data parameters
    parser.add_argument(
        '--anchor_idx_path',
        type=str,
        default=None,
        help='Path to anchor user index JSON (auto-built if not provided)'
    )
    parser.add_argument(
        '--candidate_path',
        type=str,
        default=None,
        help='Path to candidate user data JSON (auto-built if not provided)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Data directory root path'
    )
    parser.add_argument(
        '--prompt_dir',
        type=str,
        default='./prompt',
        help='Prompt template directory'
    )
    parser.add_argument(
        '--anchor_dir',
        type=str,
        default='./anchors',
        help='Anchor directory root path'
    )

    # Output parameters
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default='./ckpts-anchor-train',
        help='Checkpoint save directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Training output directory (auto-built if not provided)'
    )
    parser.add_argument(
        '--max_anchor_users',
        type=int,
        default=None,
        help='Maximum number of anchor users to train (None=all)'
    )

    parser.add_argument(
        '--merged_model_save_path',
        type=str,
        default=None,
        help='Path to save/load the merged model (TaskLoRA merged into base model). '
             'If not provided, defaults to <output_dir>/merged_model. '
             'If the merged model already exists, skip the merge step.'
    )

    return parser


def parse_anchor_train_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for anchor user training.

    Automatically builds default paths for anchor index and candidate files
    if not provided.

    Returns:
        Parsed arguments namespace
    """
    parser = create_anchor_train_parser()
    args = parser.parse_args()

    # Auto-build default paths if not provided
    if args.anchor_idx_path is None:
        args.anchor_idx_path = str(Path(args.anchor_dir) / 'anchors_by_his' / args.task_name / 'k16' / 'anchor_user_idx.json')

    if args.candidate_path is None:
        args.candidate_path = str(Path(args.data_dir) / args.task_name / 'user_anchor_candidate.json')

    if args.output_dir is None:
        args.output_dir = str(Path(args.ckpt_dir) / f"{args.task_name}-anchor_train")

    # Auto-build merged model save path if not provided
    if args.merged_model_save_path is None:
        args.merged_model_save_path = str(Path(args.output_dir) / "merged_model")

    return args


def create_pair_gen_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for DPO pair generation.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="PAPPO DPO Pair Generation for Anchor Users",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Step selection
    parser.add_argument(
        '--step',
        type=int,
        default=0,
        choices=[0, 1, 2],
        help='Step to run: 0=both, 1=cluster assignment only, 2=pair generation only'
    )

    # Model parameters
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
        help='Base model name or path (native HF Hub ID or huggingface:// URI). '
             'Alternative: mistralai/Mistral-7B-Instruct-v0.3'
    )
    parser.add_argument(
        '--task_lora_path',
        type=str,
        default=None,
        help='TaskLoRA checkpoint directory path (required for step 2)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='./deberta',
        help='DeBERTa encoder model path (for step 1)'
    )
    parser.add_argument(
        '--access_token',
        type=str,
        default=None,
        help='HuggingFace access token'
    )

    # Device parameters
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='GPU device ID to use'
    )

    # Task parameters
    parser.add_argument(
        '--task_name',
        type=str,
        required=True,
        choices=['citation', 'movie_tagging', 'news_headline',
                 'product_rating', 'scholarly_title', 'tweet_paraphrase'],
        help='Task name'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=1,
        help='Number of history items to retrieve using BM25'
    )
    parser.add_argument(
        '--add_profile',
        action='store_true',
        default=False,
        help='Add user profile to prompts'
    )
    parser.add_argument(
        '--entries_per_cluster',
        type=int,
        default=1,
        help='Number of profile entries to keep per (test_user, cluster) pair (closest to cluster center)'
    )

    # LoRA parameters
    parser.add_argument(
        '--lora_r',
        type=int,
        default=8,
        help='LoRA rank (r) for UserLoRA'
    )
    parser.add_argument(
        '--lora_alpha',
        type=int,
        default=8,
        help='LoRA alpha parameter for UserLoRA'
    )

    # Data parameters
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Data directory root path'
    )
    parser.add_argument(
        '--anchor_dir',
        type=str,
        default='./anchors',
        help='Anchor directory root path'
    )
    parser.add_argument(
        '--prompt_dir',
        type=str,
        default='./prompt',
        help='Prompt template directory'
    )
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default='./ckpts-anchor-train',
        help='UserLoRA checkpoint directory'
    )

    # Output parameters
    parser.add_argument(
        '--max_users',
        type=int,
        default=None,
        help='Maximum number of test users to process (None=all 100)'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=256,
        help='Maximum new tokens to generate'
    )
    parser.add_argument(
        '--cluster_k',
        type=int,
        default=16,
        help='Number of clusters (k value used in anchor selection)'
    )

    return parser


def parse_pair_gen_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for DPO pair generation.

    Validates required arguments based on selected step.

    Returns:
        Parsed arguments namespace
    """
    parser = create_pair_gen_parser()
    args = parser.parse_args()

    return args
