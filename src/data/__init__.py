"""
Data loading and processing for PAPPO task LoRA training.

This module provides:
- Data loading from JSON files
- BM25-based history retrieval
- Prompt construction for different tasks
- Dataset creation and tokenization
"""

from .loader import load_data_files, load_json_file, DataPaths
from .processor import DataProcessor, process_train_data, process_test_data
from .dataset import create_dataset, tokenize_prompt, generate_and_tokenize_prompt
from .anchor_loader import AnchorUserLoader, load_anchor_users_simple
from .precompute_loader import (
    dpo_pair_path,
    precompute_user_path,
    precompute_strategy_dir,
    load_spval_index,
    load_entry_idx_index,
    TargetUserPrompter,
    build_user_items,
    load_user_profiles,
)

__all__ = [
    'load_data_files',
    'load_json_file',
    'DataPaths',
    'DataProcessor',
    'process_train_data',
    'process_test_data',
    'create_dataset',
    'tokenize_prompt',
    'generate_and_tokenize_prompt',
    'AnchorUserLoader',
    'load_anchor_users_simple',
    'dpo_pair_path',
    'precompute_user_path',
    'precompute_strategy_dir',
    'load_spval_index',
    'load_entry_idx_index',
    'TargetUserPrompter',
    'build_user_items',
    'load_user_profiles',
]
