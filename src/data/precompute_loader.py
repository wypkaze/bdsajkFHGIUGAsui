"""
Load per-user PAPPO training items.

Inputs (both keyed by the test user):
    data/<task>/<out_subdir>/dpo_pair.json                 {uid: [{x, y_w, y_l}, ...]}
    records/precompute/<task>/<strategy>/user_<uid>.json   [{qid, entry_idx, x, y_w, y_l,
                                                            len_w, len_l, sp_val_pappo_*, ...}]

PAPPO consumes the record-level *aggregated* reference margin Δ̂_ref that
precompute.py already wrote per triple:

    sp_val_pappo_norm   (--norm, per-token)  |  sp_val_pappo_unorm (--no_norm)

so no anchor model is ever forwarded at training time.

Prompts are rebuilt with src.pipeline.prompts (identical to precompute), keyed by
the record's own `entry_idx`, so no queries.json join is needed in the normal
(--require_ref) path.

Trimmed: the most-relevant-anchor reference index (DPO / HyPO), the meta-only
index (SimPO / IPO) and the ablation re-aggregation hooks all belonged to the
removed methods and are gone.
"""

from pathlib import Path

from ..pipeline.prompts import build_target_prompt, format_history_lines
from .loader import load_json_file


# ------------------------- path helpers -------------------------
def dpo_pair_path(data_dir, task_name, out_subdir="pappo_anchor"):
    return Path(data_dir) / task_name / out_subdir / "dpo_pair.json"


def precompute_user_path(records_dir, task_name, strategy, uid):
    return Path(records_dir) / "precompute" / task_name / strategy / f"user_{uid}.json"


def precompute_strategy_dir(records_dir, task_name, strategy):
    return Path(records_dir) / "precompute" / task_name / strategy


# ------------------------- sp_val index (PAPPO) -------------------------
def load_spval_index(task_name, uid, records_dir, strategy="similar", logger=None):
    """(x, y_w, y_l) -> {qid, entry_idx, len_w, len_l, sp_val_pappo_*} for PAPPO.

    PAPPO consumes the record-level *aggregated* reference margin Δ̂_ref that
    precompute.py already wrote per triple:

        sp_val_pappo_norm / sp_val_pappo_unorm   (ReLU-gated weighted mean)

    The training loss reads the norm/unorm variant matching --norm/--no_norm, so no
    anchor model is ever forwarded here. Silently returns {} when the user's
    precompute file does not exist; callers decide whether that is fatal.
    """
    path = precompute_user_path(records_dir, task_name, strategy, uid)
    if not path.exists():
        if logger:
            logger.warning(f"[spval] no precompute file for user {uid}: {path}")
        return {}
    index = {}
    for rec in load_json_file(str(path)):
        key = (str(rec["x"]), str(rec["y_w"]), str(rec["y_l"]))
        if key in index:
            continue  # first match wins (duplicate triples are possible)
        entry = _meta_of(rec)
        entry.update({
            "sp_val_pappo_unorm": float(rec.get("sp_val_pappo_unorm", 0.0) or 0.0),
            "sp_val_pappo_norm": float(rec.get("sp_val_pappo_norm", 0.0) or 0.0),
        })
        index[key] = entry
    if logger:
        logger.info(f"[spval] user {uid}: {len(index)} triples from {path.name}")
    return index


def _meta_of(rec):
    try:
        entry_idx = int(rec["entry_idx"])
    except (TypeError, ValueError):
        entry_idx = 0
    return {
        "qid": rec.get("qid"),
        "entry_idx": entry_idx,
        "len_w": max(int(rec.get("len_w", 1) or 1), 1),
        "len_l": max(int(rec.get("len_l", 1) or 1), 1),
    }


# ------------------------- prompt builder -------------------------
class TargetUserPrompter:
    """Rebuilds the precompute-identical target-user prompt, caching per (entry_idx, x)."""

    def __init__(self, task_name, profile_list, prompt_tmpl, k, x_field):
        self.task_name = task_name
        self.prompt_tmpl = prompt_tmpl
        self.k = k
        self.x_field = x_field
        self.hist_lines = format_history_lines(profile_list, prompt_tmpl["retrieval_history"])
        self._visible_cache = {}
        self._prompt_cache = {}

    def _visible(self, entry_idx):
        if entry_idx not in self._visible_cache:
            self._visible_cache[entry_idx] = [l for l in self.hist_lines[:entry_idx] if l is not None]
        return self._visible_cache[entry_idx]

    def prompt_for(self, x, entry_idx):
        key = (entry_idx, x)
        if key not in self._prompt_cache:
            self._prompt_cache[key] = build_target_prompt(
                self.task_name, x, self._visible(entry_idx),
                self.prompt_tmpl, self.k, self.x_field,
            )
        return self._prompt_cache[key]


# ------------------------- item assembly -------------------------
def load_entry_idx_index(pa_dir):
    """(x, y_w) -> entry_idx, recovered from queries.json.

    Only needed by the `zero_ref` fallback (--no_require_ref): dpo_pair.json carries
    no entry_idx, so the causal history slice has to come from queries.json.
    `entry_idx` is stored as a string there.
    """
    path = Path(pa_dir) / "queries.json"
    if not path.exists():
        return {}
    out = {}
    for q in load_json_file(str(path)):
        key = (str(q.get("x", "")), str(q.get("y_w", "")))
        if key in out:
            continue  # first match wins (<1% inherent collisions)
        try:
            out[key] = int(q["entry_idx"])
        except (TypeError, ValueError, KeyError):
            out[key] = 0
    return out


_ZERO_REF = {
    "qid": None,
    "entry_idx": 0,
    "len_w": 1,
    "len_l": 1,
    # Aggregated reference margin (PAPPO); zero when recovered from queries.json
    # so the loss degenerates to the policy-only term.
    "sp_val_pappo_unorm": 0.0,
    "sp_val_pappo_norm": 0.0,
}


def build_user_items(uid, dpo_triples, ref_index, prompter, require_ref=True,
                     entry_idx_index=None, tokenizer=None, logger=None):
    """Turn a user's dpo_pair triples into training items carrying Δ̂_ref.

    Each item: {prompt, y_w, y_l, len_w, len_l, entry_idx, qid,
                sp_val_pappo_norm, sp_val_pappo_unorm}

    `ref_index` comes from load_spval_index.

    require_ref=True (default): triples with no precompute record are dropped (counted).
    require_ref=False: such triples are kept with the reference margin forced to 0 (the
    loss then degenerates to the policy-only term). Their entry_idx is recovered from
    queries.json and token lengths from the tokenizer; triples whose entry_idx or
    tokenizer are unavailable are dropped.
    """
    items = []
    n_missing = 0
    n_unmatched_idx = 0
    n_no_tokenizer = 0
    for triple in dpo_triples:
        x_raw = str(triple.get("x", ""))
        y_w = str(triple.get("y_w", "")).strip()
        y_l = str(triple.get("y_l", "")).strip()
        if not x_raw.strip() or not y_w or not y_l:
            continue
        key = (x_raw, str(triple.get("y_w", "")), str(triple.get("y_l", "")))
        ref = ref_index.get(key)
        if ref is not None:
            item = dict(_ZERO_REF)
            item.update(ref)
            item["prompt"] = prompter.prompt_for(x_raw, ref["entry_idx"])
            item["y_w"] = str(triple["y_w"])
            item["y_l"] = str(triple["y_l"])
            items.append(item)
            continue

        n_missing += 1
        if require_ref:
            continue

        entry_idx = (entry_idx_index or {}).get((x_raw, str(triple.get("y_w", ""))))
        if entry_idx is None:
            n_unmatched_idx += 1
            continue
        if tokenizer is None:
            n_no_tokenizer += 1
            continue
        item = dict(_ZERO_REF)
        item.update({
            "entry_idx": entry_idx,
            "len_w": _tok_len(tokenizer, str(triple["y_w"])),
            "len_l": _tok_len(tokenizer, y_l),
            "prompt": prompter.prompt_for(x_raw, entry_idx),
            "y_w": str(triple["y_w"]),
            "y_l": y_l,
        })
        items.append(item)

    if logger and n_missing:
        msg = (f"[items] user {uid}: {n_missing} triples without precompute ref "
               f"-> dropped ({len(items)} kept)")
        if not require_ref:
            msg = (f"[items] user {uid}: {n_missing} triples missing from the precompute "
                   f"index -> recovered from queries.json ({len(items)} kept, "
                   f"{n_unmatched_idx} dropped for unknown entry_idx, "
                   f"{n_no_tokenizer} dropped for no tokenizer)")
        logger.warning(msg)
    return items


def _tok_len(tokenizer, text):
    if tokenizer is None:
        return 1
    return max(len(tokenizer.encode(str(text), add_special_tokens=False)), 1)


def load_user_profiles(data_dir, task_name):
    """uid -> profile list, from test_100/user_test_100.json."""
    path = Path(data_dir) / task_name / "test_100" / "user_test_100.json"
    if not path.exists():
        return {}
    return {str(u["user_id"]): (u.get("profile", []) or []) for u in load_json_file(str(path))}
