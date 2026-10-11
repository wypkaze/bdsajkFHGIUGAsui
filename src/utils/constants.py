"""
Constants used across the PAPPO project.
"""

# Task name to ID mapping
name2taskid = {
    "citation": "LaMP_1",
    "movie_tagging": "LaMP_2M",
    "news_categorize": "LaMP_2N",
    "news_headline": "LaMP_4",
    "product_rating": "LaMP_3",
    "scholarly_title": "LaMP_5",
    "tweet_paraphrase": "LaMP_7"
}

# Import extractors to create function mapping
from .extractors import (
    extract_citation_title,
    extract_movie,
    extract_news_headline,
    extract_product_review,
    extract_scholarly_title,
    extract_tweet_paraphrasing,
)

# Task to extractor function mapping (actual function references)
TASK_EXTRACTORS = {
    "citation": extract_citation_title,
    "movie_tagging": extract_movie,
    "news_headline": extract_news_headline,
    "product_rating": extract_product_review,
    "scholarly_title": extract_scholarly_title,
    "tweet_paraphrase": extract_tweet_paraphrasing,
}
