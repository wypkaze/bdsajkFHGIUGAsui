"""
Utility functions for anchor user selection.

This module provides general utility functions used across the anchor selection pipeline.
"""

from typing import List, Any


def get_first_k_tokens(text: str, k: int) -> str:
    """
    Extracts the first k tokens from a text string.
    
    Args:
        text: Input text string
        k: Number of tokens to extract
    
    Returns:
        String containing the first k tokens
    """
    if not isinstance(text, str):
        text = str(text)
    tokens = text.split()
    output = " ".join(tokens[:k])
    return output


def split_batch(init_list: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split a list into batches of specified size.
    
    Args:
        init_list: List to be split
        batch_size: Size of each batch
    
    Returns:
        List of batches
    """
    if not init_list:
        return []
    
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    if count != 0:
        end_list.append(init_list[-count:])
    return end_list
