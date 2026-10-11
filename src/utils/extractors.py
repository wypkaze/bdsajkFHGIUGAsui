"""
Text extraction functions for different LAMP tasks.

Each function extracts specific content from the task input text.
"""

import re


def extract_option(s: str, num: int) -> str | None:
    """
    Extract option content from citation task input.
    
    Args:
        s: Input text containing options like [1]: "content"
        num: Option number to extract (1 or 2)
    
    Returns:
        Extracted option content or None if not found
    """
    match = re.search(r'\[' + str(num) + '\]: "([^"]*)"', s)
    return match.group(1) if match else None


def extract_citation_title(text: str) -> str | None:
    """
    Extract paper title from citation task input.
    
    Args:
        text: Input text containing 'written the paper with the title "..."'
    
    Returns:
        Extracted paper title or None if not found
    """
    pattern = r'written the paper with the title "([^"]*)"'
    match = re.search(pattern, text)
    return match.group(1) if match else None


def extract_movie(text: str) -> str:
    """
    Extract movie description from movie_tagging task input.
    
    Args:
        text: Input text containing '] description: ...'
    
    Returns:
        Extracted movie description
    
    Raises:
        ValueError: If marker not found in text
    """
    marker = "] description: "
    marker_pos = text.find(marker)
    if marker_pos == -1:
        raise ValueError("Marker not found in movie text")
    return text[marker_pos + len(marker):]


def extract_news_headline(text: str) -> str:
    """
    Extract article content from news_headline task input.
    
    Args:
        text: Input text containing 'Generate a headline for the following article: ...'
    
    Returns:
        Extracted article content
    
    Raises:
        ValueError: If marker not found in text
    """
    marker = "Generate a headline for the following article: "
    marker_pos = text.find(marker)
    if marker_pos == -1:
        raise ValueError("Marker not found in news text")
    return text[marker_pos + len(marker):]


def extract_product_review(text: str) -> str:
    """
    Extract review content from product_rating task input.
    
    Args:
        text: Input text containing 'without further explanation. review: ...'
    
    Returns:
        Extracted review content
    
    Raises:
        ValueError: If marker not found in text
    """
    marker = "without further explanation. review: "
    marker_pos = text.find(marker)
    if marker_pos == -1:
        raise ValueError("Marker not found in product text")
    return text[marker_pos + len(marker):]


def extract_scholarly_title(text: str) -> str:
    """
    Extract abstract from scholarly_title task input.
    
    Args:
        text: Input text containing 'Generate a title for the following abstract of a paper: ...'
    
    Returns:
        Extracted abstract content
    
    Raises:
        ValueError: If marker not found in text
    """
    marker = "Generate a title for the following abstract of a paper: "
    marker_pos = text.find(marker)
    if marker_pos == -1:
        raise ValueError("Marker not found in scholarly text")
    return text[marker_pos + len(marker):]


def extract_tweet_paraphrasing(text: str) -> str:
    """
    Extract original text from tweet_paraphrase task input.
    
    Args:
        text: Input text containing 'Paraphrase the following text into tweet without any explanation before or after it: ...'
    
    Returns:
        Extracted original text
    
    Raises:
        ValueError: If marker not found in text
    """
    marker = "Paraphrase the following tweet without any explanation before or after it: "
    marker_pos = text.find(marker)
    if marker_pos == -1:
        raise ValueError("Marker not found in tweet text")
    return text[marker_pos + len(marker):]
