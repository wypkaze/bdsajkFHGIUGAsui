"""
Dataset creation and tokenization utilities.
"""

from datasets import Dataset
from transformers import PreTrainedTokenizer


def tokenize_prompt(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    cutoff_len: int = 2048,
    add_eos_token: bool = True,
) -> dict:
    """
    Tokenize a prompt string.
    
    Args:
        tokenizer: Tokenizer to use
        prompt: Text to tokenize
        cutoff_len: Maximum sequence length
        add_eos_token: Whether to add EOS token
    
    Returns:
        Dictionary with input_ids, attention_mask, labels
    """
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    
    if add_eos_token and result["input_ids"][-1] != tokenizer.eos_token_id:
        result = tokenizer(
            prompt + tokenizer.eos_token,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
    
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(
    data_point: dict,
    tokenizer: PreTrainedTokenizer,
    cutoff_len: int = 2048,
) -> dict:
    """
    Generate and tokenize a training sample with label masking.
    
    Masks the prompt part of the input so only the answer contributes to loss.
    
    Args:
        data_point: Dictionary with 'prompt' and 'full_prompt'
        tokenizer: Tokenizer to use
        cutoff_len: Maximum sequence length
    
    Returns:
        Tokenized data with masked labels
    """
    full_prompt = data_point['full_prompt']
    tokenized_full_prompt = tokenize_prompt(tokenizer, full_prompt, cutoff_len)
    
    user_prompt = data_point['prompt']
    tokenized_user_prompt = tokenize_prompt(
        tokenizer, user_prompt, cutoff_len, add_eos_token=False
    )
    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    
    # Only compute loss on answer part
    labels = tokenized_full_prompt["input_ids"].copy()
    for i in range(min(user_prompt_len, len(labels))):
        labels[i] = -100
    
    tokenized_full_prompt["labels"] = labels
    return tokenized_full_prompt


def create_dataset(
    processed_data: list[dict],
    tokenizer: PreTrainedTokenizer,
    cutoff_len: int = 2048,
    shuffle: bool = True,
) -> Dataset:
    """
    Create a HuggingFace Dataset from processed data.
    
    Args:
        processed_data: List of processed data points
        tokenizer: Tokenizer to use
        cutoff_len: Maximum sequence length
        shuffle: Whether to shuffle the dataset
    
    Returns:
        HuggingFace Dataset
    """
    dataset = Dataset.from_list(processed_data)
    
    def tokenize_fn(data_point):
        return generate_and_tokenize_prompt(data_point, tokenizer, cutoff_len)
    
    dataset = dataset.map(tokenize_fn)
    
    if shuffle:
        dataset = dataset.shuffle()
    
    return dataset
