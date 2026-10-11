"""
Text manipulation utilities.
"""


def get_first_k_tokens(text: str, k: int) -> str:
    """
    Get the first k tokens from a text string.
    
    Tokens are split by whitespace.
    
    Args:
        text: Input text string
        k: Number of tokens to extract
    
    Returns:
        First k tokens joined by space
    """
    text = str(text)
    tokens = text.split()
    return " ".join(tokens[:k])


def split_batch(init_list: list, batch_size: int) -> list:
    """
    Split a list into batches of specified size.
    
    Args:
        init_list: List to split
        batch_size: Size of each batch
    
    Returns:
        List of batches
    """
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    if count != 0:
        end_list.append(init_list[-count:])
    return end_list
