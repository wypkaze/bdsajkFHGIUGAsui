"""
Utility functions for PAPPO task LoRA training.

This module provides various utility functions including:
- Text extraction functions for different tasks
- Token manipulation utilities
- Batch processing helpers
- Model inspection tools
"""

from .extractors import (
    extract_option,
    extract_citation_title,
    extract_movie,
    extract_news_headline,
    extract_product_review,
    extract_scholarly_title,
    extract_tweet_paraphrasing,
)

from .text_utils import (
    get_first_k_tokens,
    split_batch,
)

from .model_utils import (
    print_trainable_parameters,
)

from .constants import (
    name2taskid,
    TASK_EXTRACTORS,
)

from .logging_utils import (
    setup_logging,
    get_logger,
)

from .arg_parser import (
    create_anchor_train_parser,
    parse_anchor_train_arguments,
    create_pair_gen_parser,
    parse_pair_gen_arguments,
)

__all__ = [
    # Extractors
    'extract_option',
    'extract_citation_title',
    'extract_movie',
    'extract_news_headline',
    'extract_product_review',
    'extract_scholarly_title',
    'extract_tweet_paraphrasing',
    # Text utils
    'get_first_k_tokens',
    'split_batch',
    # Model utils
    'print_trainable_parameters',
    # Constants
    'name2taskid',
    'TASK_EXTRACTORS',
    # Logging utils
    'setup_logging',
    'get_logger',
    # Argument parser
    'create_anchor_train_parser',
    'parse_anchor_train_arguments',
    'create_pair_gen_parser',
    'parse_pair_gen_arguments',
]
