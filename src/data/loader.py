"""
Data loading utilities.
"""

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class DataPaths:
    """Container for data file paths."""
    train_data: str
    test_data: str
    test_labels: str
    profile_id2text: str
    prompt_template: str


def load_json_file(path: str) -> Any:
    """Load a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_data_files(
    task_name: str,
    data_dir: str = "./data",
    prompt_dir: str = "./prompt",
) -> tuple[DataPaths, dict, list, list, dict]:
    """
    Load all necessary data files for training.
    
    Args:
        task_name: Name of the task
        data_dir: Directory containing task data
        prompt_dir: Directory containing prompt templates
    
    Returns:
        Tuple of (DataPaths, prompt_template, train_data, test_data, all_profile)
    """
    # Create paths
    paths = DataPaths(
        train_data=f"{data_dir}/{task_name}/user_base_LLM.json",
        test_data=f"{data_dir}/{task_name}/test_100/user_test_100.json",
        test_labels=f"{data_dir}/{task_name}/test_100/user_test_100_label.json",
        profile_id2text=f"{data_dir}/{task_name}/profile-id2text.json",
        prompt_template=f"{prompt_dir}/prompt.json",
    )
    
    # Load data
    train_data = load_json_file(paths.train_data)
    test_data = load_json_file(paths.test_data)
    prompt_template = load_json_file(paths.prompt_template)
    
    # Load profile-id2text (may not exist)
    all_profile = {}
    try:
        all_profile = load_json_file(paths.profile_id2text)
    except FileNotFoundError:
        print(f"Warning: profile-id2text.json not found for {task_name}")
    
    return paths, prompt_template, train_data, test_data, all_profile
