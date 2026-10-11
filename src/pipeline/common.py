"""
Shared helpers for the anchor log-prob pipeline.

Path convention (per the approved plan): all defaults are derived from the
repository root at runtime (NO hardcoded absolute paths), and every
path can be overridden via CLI args in the three scripts.

Layout assumed (inputs live under PAPPO_record, code under PAPPO):
    PAPPO_record/data/<task>/test_100/user_test_100.json
    PAPPO_record/data/<task>/user_anchor_candidate.json
    PAPPO_record/data/<task>/profile-id2text.json
    PAPPO_record/anchors/anchors_by_his/<task>/k16/anchor_user_idx.json
    PAPPO_record/ckpts/anchor-train/<task>-anchor_train/merged_model
    PAPPO_record/ckpts/anchor-train/<task>-anchor_train/user_<id>_anchor_lora
    PAPPO/prompt/prompt_pair.json
Products are written to:
    PAPPO_record/data/<task>/<out_subdir>/   (default out_subdir = "pappo_anchor")
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import get_task_config
from ..data.anchor_loader import AnchorUserLoader
from ..data.loader import load_json_file

# --- Repo-relative path roots (computed, never hardcoded) -------------------
_SRC_DIR = Path(__file__).resolve().parent.parent      # .../PAPPO/src
IMPLE_DIR = _SRC_DIR.parent                            # .../PAPPO
REPO_ROOT = IMPLE_DIR.parent                           # .../<repo_root>

DEFAULT_DATA_DIR = REPO_ROOT / "PAPPO_record" / "data"
DEFAULT_CKPT_DIR = REPO_ROOT / "PAPPO_record" / "ckpts" / "anchor-train"
DEFAULT_ANCHOR_IDX_DIR = REPO_ROOT / "PAPPO_record" / "anchors" / "anchors_by_his"
DEFAULT_PROMPT_DIR = IMPLE_DIR / "prompt"


# --- Small IO helpers -------------------------------------------------------
def load_json(path) -> object:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# --- Path builders ----------------------------------------------------------
def merged_model_path(ckpt_dir, task_name: str) -> str:
    return str(Path(ckpt_dir) / f"{task_name}-anchor_train" / "merged_model")


def anchor_lora_path(ckpt_dir, task_name: str, anchor_id) -> str:
    return str(Path(ckpt_dir) / f"{task_name}-anchor_train" / f"user_{anchor_id}_anchor_lora")


def out_dir_for(data_dir, task_name: str, out_subdir: str = "pappo_anchor") -> Path:
    return Path(data_dir) / task_name / out_subdir


def load_prompt_pair(prompt_dir) -> Dict:
    return load_json_file(str(Path(prompt_dir) / "prompt_pair.json"))


# --- Anchor loading ---------------------------------------------------------
def load_anchor_map(
    task_name: str,
    data_dir,
    anchor_idx_dir,
    k: int = 16,
) -> Tuple[List[int], Dict[int, Dict]]:
    """
    Load anchor ids (dynamic count; e.g. 64 for most tasks, 56 for movie_tagging)
    and an id -> anchor-user-dict map (with 'profile' history).

    Uses the reused AnchorUserLoader with explicit path overrides so it reads
    from PAPPO_record regardless of the current working directory.
    """
    anchor_idx_path = Path(anchor_idx_dir) / task_name / f"k{k}" / "anchor_user_idx.json"
    candidate_path = Path(data_dir) / task_name / "user_anchor_candidate.json"
    loader = AnchorUserLoader(
        task_name=task_name,
        data_dir=str(data_dir),
        anchor_idx_path=str(anchor_idx_path),
        candidate_path=str(candidate_path),
        k=k,
    )
    anchor_ids, anchor_users = loader.load_anchor_users()
    anchor_map = {u["user_id"]: u for u in anchor_users}
    return anchor_ids, anchor_map


# --- Query building ---------------------------------------------------------
def build_queries(
    task_name: str,
    test_users: List[Dict],
    formalized_x: Optional[Dict] = None,
    max_users: Optional[int] = None,
    x_source_field: str = "formal_x",
) -> List[Dict]:
    """
    Flatten every profile entry of every test user into one query (NO filtering).

    Each query: {qid, user_id, entry_idx, x, y_w}. qid = "<user_id>::<entry_idx>".
    x/y_w come from the task's profile-entry fields, except tweet_paraphrase
    where x is taken from the `formalized_x` dict's `x_source_field` (y_w stays
    the raw tweet). `x_source_field` selects which rewrite product feeds x:
        "formal_x"    -> Pass 0  formalized_x.json
        "detoned_x"   -> Pass 0.5 detoned_x.json   (de-emotionalized)
        "corrected_x" -> Pass 0.6 corrected_x.json (meaning-corrected)
    The supplied dict must contain that field per qid.

    Entries whose y_w is empty are skipped (they cannot form a preference pair);
    for tweet, entries missing the chosen x_source_field are skipped.
    """
    cfg = get_task_config(task_name)
    x_field = cfg["x_field"]
    y_w_field = cfg["y_w_field"]
    x_from_formalize = cfg.get("x_from_formalize", False)

    users = test_users if max_users is None else test_users[:max_users]

    queries: List[Dict] = []
    for user in users:
        user_id = str(user.get("user_id", "unknown"))
        profile = user.get("profile", []) or []
        for entry_idx, entry in enumerate(profile):
            qid = f"{user_id}::{entry_idx}"
            y_w = str(entry.get(y_w_field, "") or "")

            if x_from_formalize:
                fx = (formalized_x or {}).get(qid)
                if fx is None or not str(fx.get(x_source_field, "") or "").strip():
                    # The chosen x source has not produced this qid yet -> skip.
                    continue
                x = str(fx[x_source_field])
            else:
                x = str(entry.get(x_field, "") or "")

            if not y_w.strip() or not x.strip():
                continue

            queries.append({
                "qid": qid,
                "user_id": user_id,
                "entry_idx": entry_idx,
                "x": x,
                "y_w": y_w,
            })
    return queries
