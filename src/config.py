"""
Task configuration for the anchor log-prob scoring + Top/Bottom generation pipeline.

This is a *thin* new layer on top of the reused src components.
It only declares, per task:
  - which profile-entry field is the input (x) and which is the gold answer (y_w)
  - how Pass B produces y_l (free generation params, or the movie_tagging 15-tag branch)
  - how to extract/clean the raw generation (marker aligned with prompt_pair.json)

Generation parameters mirror the reference implementation
`src/training/pair_generator.py::_do_generate` so behavior stays consistent.
"""

# Tasks handled by this pipeline (product_rating intentionally removed).
SUPPORTED_TASKS = [
    "citation",
    "movie_tagging",
    "news_headline",
    "scholarly_title",
    "tweet_paraphrase",
]

# 15 candidate tags for movie_tagging (taken verbatim from prompt_pair.json).
MOVIE_TAGS = [
    "sci-fi", "based on a book", "comedy", "action", "twist ending",
    "dystopia", "dark comedy", "classic", "psychology", "fantasy",
    "romance", "thought-provoking", "social commentary", "violence", "true story",
]

# Sampling generation params shared by the short-text tasks (news/scholarly/tweet).
_SAMPLE_GEN = {
    "max_new_tokens": 64,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": None,
    "repetition_penalty": 1.1,
}

TASK_CONFIG = {
    "citation": {
        "x_field": "title",
        "y_w_field": "citation",
        # x is taken directly from the profile-entry field.
        "x_from_formalize": False,
        # greedy, long output, mild repetition penalty (matches _do_generate).
        "gen": {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "repetition_penalty": 1.15,
        },
        # citation is cleaned specially: split on "Answer:" then strip "Paper Title:".
        "extract": "citation",
        "special": None,
    },
    "movie_tagging": {
        "x_field": "description",
        "y_w_field": "tag",
        "x_from_formalize": False,
        "gen": None,  # no free generation
        "extract": "movie_tagging",
        # Pass B ranks the 15 tags by log p_anchor(tag | prompt) and picks the
        # first tag != y_w, which structurally guarantees y_l != y_w.
        "special": "tag_rank",
        "tags": MOVIE_TAGS,
    },
    "news_headline": {
        "x_field": "text",
        "y_w_field": "title",
        "x_from_formalize": False,
        "gen": dict(_SAMPLE_GEN),
        "extract": "news_headline",
        "special": None,
    },
    "scholarly_title": {
        "x_field": "abstract",
        "y_w_field": "title",
        "x_from_formalize": False,
        "gen": dict(_SAMPLE_GEN),
        "extract": "scholarly_title",
        "special": None,
    },
    "tweet_paraphrase": {
        # x is the formalized tweet produced by Pass 0 (formalize_tweets.py),
        # y_w is the raw tweet (the entry's "text" field).
        "x_field": "text",
        "y_w_field": "text",
        "x_from_formalize": True,
        "gen": dict(_SAMPLE_GEN),
        "extract": "tweet_paraphrase",
        "special": None,
    },
}


def get_task_config(task_name: str) -> dict:
    """Return the config dict for a task, raising on unknown/unsupported tasks."""
    if task_name not in TASK_CONFIG:
        raise ValueError(
            f"Unknown or unsupported task_name: {task_name}. "
            f"Supported: {SUPPORTED_TASKS}"
        )
    return TASK_CONFIG[task_name]
