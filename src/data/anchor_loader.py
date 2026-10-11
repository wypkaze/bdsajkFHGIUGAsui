"""
Anchor user data loading utilities.

Provides specialized data loading for anchor user training,
including anchor index and candidate user data loading.
"""

from typing import Tuple, List, Dict, Optional
from pathlib import Path

from .loader import load_json_file


class AnchorUserLoader:
    """
    Loader for anchor user data.

    Handles loading of anchor user indices and candidate user data,
    with support for user profile loading.
    """

    def __init__(
        self,
        task_name: str,
        data_dir: str = "./data",
        anchor_dir: str = "./anchors",
        k: int = 16,
        anchor_idx_path: Optional[str] = None,
        candidate_path: Optional[str] = None,
    ):
        """
        Initialize anchor user loader.

        Args:
            task_name: Name of the task
            data_dir: Data directory root path
            anchor_dir: Anchor directory root path
            k: Number of anchors parameter (used for path building)
            anchor_idx_path: Override path for anchor index JSON
            candidate_path: Override path for candidate user data JSON
        """
        self.task_name = task_name
        self.data_dir = Path(data_dir)
        self.anchor_dir = Path(anchor_dir)
        self.k = k

        # Set paths (use overrides if provided, otherwise build defaults)
        if anchor_idx_path:
            self.anchor_idx_path = Path(anchor_idx_path)
        else:
            self.anchor_idx_path = self.anchor_dir / 'anchors_by_his' / task_name / f'k{k}' / 'anchor_user_idx.json'

        if candidate_path:
            self.candidate_path = Path(candidate_path)
        else:
            self.candidate_path = self.data_dir / task_name / 'user_anchor_candidate.json'

        self.profile_path = self.data_dir / task_name / 'profile-id2text.json'

    def load_anchor_index(self) -> Dict:
        """
        Load anchor user index data.

        Returns:
            Dictionary containing anchor index information
        """
        print(f"Loading anchor index: {self.anchor_idx_path}")
        anchor_data = load_json_file(str(self.anchor_idx_path))
        return anchor_data

    def load_anchor_users(self) -> Tuple[List[int], List[Dict]]:
        """
        Load anchor user IDs and detailed user data.

        Loads anchor user index and matches with candidate user data
        to get full user information.

        Returns:
            Tuple of (anchor_ids, anchor_users_data)
        """
        # Load anchor index
        anchor_data = self.load_anchor_index()
        anchor_ids = anchor_data.get('all_anchor_ids', [])
        print(f"Found {len(anchor_ids)} anchor users in index")

        # Load candidate user data
        print(f"Loading candidate user data: {self.candidate_path}")
        all_candidates = load_json_file(str(self.candidate_path))

        # Build user_id to data mapping
        candidate_dict = {user['user_id']: user for user in all_candidates}

        # Get detailed data for anchor users
        anchor_users = []
        for uid in anchor_ids:
            if uid in candidate_dict:
                anchor_users.append(candidate_dict[uid])
            else:
                print(f"Warning: Anchor user {uid} not found in candidate data")

        print(f"Successfully loaded {len(anchor_users)} anchor user data")
        return anchor_ids, anchor_users

    def load_user_profiles(self) -> Dict[str, str]:
        """
        Load user profile texts.

        Returns:
            Dictionary mapping user_id to profile text
        """
        if self.profile_path.exists():
            print(f"Loading user profiles: {self.profile_path}")
            return load_json_file(str(self.profile_path))
        else:
            print(f"Warning: User profile file not found: {self.profile_path}")
            return {}

    def load_all(self) -> Tuple[List[int], List[Dict], Dict[str, str]]:
        """
        Load all anchor user related data at once.

        Returns:
            Tuple of (anchor_ids, anchor_users, user_profiles)
        """
        anchor_ids, anchor_users = self.load_anchor_users()
        profiles = self.load_user_profiles()
        return anchor_ids, anchor_users, profiles


def load_anchor_users_simple(
    task_name: str,
    data_dir: str = "./data",
    anchor_dir: str = "./anchors",
    anchor_idx_path: Optional[str] = None,
    candidate_path: Optional[str] = None,
) -> Tuple[List[int], List[Dict]]:
    """
    Simple function to load anchor users without creating a loader instance.

    Args:
        task_name: Name of the task
        data_dir: Data directory root path
        anchor_dir: Anchor directory root path
        anchor_idx_path: Override path for anchor index JSON
        candidate_path: Override path for candidate user data JSON

    Returns:
        Tuple of (anchor_ids, anchor_users_data)
    """
    loader = AnchorUserLoader(
        task_name=task_name,
        data_dir=data_dir,
        anchor_dir=anchor_dir,
        anchor_idx_path=anchor_idx_path,
        candidate_path=candidate_path,
    )
    return loader.load_anchor_users()
