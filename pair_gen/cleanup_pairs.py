#!/usr/bin/env python3
"""
Post-process Pass B output: drop duplicate y_l within the same query.

Every query (x, y_w) gets TWO y_l candidates in Pass B -- one from the Top-1
"similar" anchor and one from the Bottom-1 "dissimilar" anchor. Occasionally the
two anchors generate the *same* y_l, which leaves a redundant (x, y_w, y_l)
triple. This script removes the duplicate so each query keeps a single copy of
an identical y_l (the first / "similar" one wins).

Dedup granularity is PER ENTRY (one query), never across entries: two different
entries of the same user may legitimately share the same (x, y_w) text, and
those must NOT be merged. The verbose dpo_pair_log.json carries explicit
entry -> pairs boundaries, so it is the source of truth; dpo_pair.json (the
clean training file) is rebuilt from the deduped log to keep them consistent.
If no log file is present, it falls back to grouping the clean file by adjacent
(x, y_w) runs (which is entry-aligned, see Pass B's assemble order).

Files (under PAPPO_record/data/<task>/<out_subdir>/):
    dpo_pair_log.json  {user_id: [{entry_idx, x, y_w, pairs:[{..., y_l}, ...]}]}
    dpo_pair.json      {user_id: [{x, y_w, y_l}, ...]}

Output (default is NON-destructive, writes *_dedup.json next to the originals):
    --out_suffix _dedup   ->  dpo_pair_dedup.json / dpo_pair_log_dedup.json
    --inplace             ->  overwrite originals (auto .bak unless --no_backup)
    --dry_run             ->  report counts only, write nothing

Usage:
    python dedup_pairs.py --task_name citation --dry_run
    python dedup_pairs.py --task_name citation                 # -> *_dedup.json
    python dedup_pairs.py --task_name all --inplace            # clean every task
"""

import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # PAPPO/
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.config import SUPPORTED_TASKS
from src.pipeline.common import (
    DEFAULT_DATA_DIR,
    out_dir_for,
    load_json,
    save_json,
)


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Drop duplicate y_l within the same query (Pass B post-process)")
    p.add_argument('--task_name', type=str, required=True,
                   help=f"One of {SUPPORTED_TASKS} or 'all'")
    p.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument('--out_subdir', type=str, default='pappo_anchor',
                   help='Pass A/B artifact subdir under data/<task>/')
    p.add_argument('--inplace', action='store_true',
                   help='Overwrite dpo_pair.json / dpo_pair_log.json in place')
    p.add_argument('--no_backup', action='store_true',
                   help='With --inplace, do NOT write .bak backups')
    p.add_argument('--out_suffix', type=str, default='_dedup',
                   help='Suffix for output files when not --inplace')
    p.add_argument('--dry_run', action='store_true',
                   help='Report what would change; write nothing')
    p.add_argument('--exact', action='store_true',
                   help='Compare y_l exactly (default: compare str(y_l).strip())')
    return p.parse_args()


def _key(y_l, exact):
    """Normalisation key for comparing two y_l strings."""
    s = y_l if isinstance(y_l, str) else str(y_l)
    return s if exact else s.strip()


def dedup_log(log, exact):
    """Dedup each entry's pairs by y_l (keep first). Returns (new_log, stats)."""
    new_log = {}
    stats = {"entries": 0, "entries_hit": 0, "pairs_before": 0,
             "pairs_after": 0, "removed": 0}
    for uid, entries in log.items():
        new_entries = []
        for e in entries:
            pairs = e.get("pairs", []) or []
            stats["entries"] += 1
            stats["pairs_before"] += len(pairs)
            seen = set()
            kept = []
            for pr in pairs:
                k = _key(pr.get("y_l"), exact)
                if k in seen:
                    stats["removed"] += 1
                    continue
                seen.add(k)
                kept.append(pr)
            if len(kept) != len(pairs):
                stats["entries_hit"] += 1
            stats["pairs_after"] += len(kept)
            ne = dict(e)
            ne["pairs"] = kept
            new_entries.append(ne)
        new_log[uid] = new_entries
    return new_log, stats


def clean_from_log(log):
    """Rebuild the clean {user_id:[{x,y_w,y_l}]} view from a (deduped) log."""
    clean = {}
    for uid, entries in log.items():
        triples = []
        for e in entries:
            for pr in e.get("pairs", []) or []:
                y_l = pr.get("y_l")
                if y_l is None or not str(y_l).strip():
                    continue
                triples.append({"x": e["x"], "y_w": e["y_w"], "y_l": y_l})
        clean[uid] = triples
    return clean


def dedup_clean_adjacent(clean, exact):
    """Fallback: dedup the clean file by adjacent (x, y_w) runs (entry-aligned).

    Pass B appends each entry's <=2 triples consecutively, so a maximal run of
    identical (x, y_w) equals one entry unless two entries share that text AND
    are adjacent -- rare, and dedup only ever collapses an exact y_l repeat, so
    this stays safe. Returns (new_clean, stats).
    """
    new_clean = {}
    stats = {"entries": 0, "entries_hit": 0, "pairs_before": 0,
             "pairs_after": 0, "removed": 0}
    for uid, triples in clean.items():
        out = []
        i = 0
        n = len(triples)
        while i < n:
            j = i
            while j < n and (triples[j]["x"], triples[j]["y_w"]) == \
                    (triples[i]["x"], triples[i]["y_w"]):
                j += 1
            group = triples[i:j]
            stats["entries"] += 1
            stats["pairs_before"] += len(group)
            seen = set()
            kept = []
            for t in group:
                k = _key(t.get("y_l"), exact)
                if k in seen:
                    stats["removed"] += 1
                    continue
                seen.add(k)
                kept.append(t)
            if len(kept) != len(group):
                stats["entries_hit"] += 1
            stats["pairs_after"] += len(kept)
            out.extend(kept)
            i = j
        new_clean[uid] = out
    return new_clean, stats


def _write(obj, path, args, dry):
    if dry:
        return
    save_json(obj, path)


def _backup(path):
    bak = Path(str(path) + ".bak")
    if path.exists():
        bak.write_bytes(path.read_bytes())
    return bak


def process_task(task, args):
    out_dir = out_dir_for(args.data_dir, task, args.out_subdir)
    log_path = out_dir / "dpo_pair_log.json"
    clean_path = out_dir / "dpo_pair.json"

    if not clean_path.exists() and not log_path.exists():
        print(f"[{task}] skip: no dpo_pair.json / dpo_pair_log.json under {out_dir}")
        return

    print(f"\n=== [{task}] {out_dir} ===")

    # Prefer the verbose log (explicit entry boundaries) as source of truth.
    if log_path.exists():
        log = load_json(str(log_path))
        new_log, stats = dedup_log(log, args.exact)
        new_clean = clean_from_log(new_log)
        src = "dpo_pair_log.json (rebuilt dpo_pair.json from it)"
        write_log = True
    else:
        clean = load_json(str(clean_path))
        new_clean, stats = dedup_clean_adjacent(clean, args.exact)
        new_log = None
        src = "dpo_pair.json (adjacent-group fallback; no log present)"
        write_log = False

    print(f"  source            : {src}")
    print(f"  queries (entries) : {stats['entries']}  "
          f"with duplicate y_l: {stats['entries_hit']}")
    print(f"  y_l pairs         : {stats['pairs_before']} -> {stats['pairs_after']}  "
          f"(removed {stats['removed']})")

    if stats["removed"] == 0:
        print("  nothing to do (no duplicate y_l found).")
        return

    if args.dry_run:
        print("  [dry_run] no files written.")
        return

    # Resolve destination paths.
    if args.inplace:
        clean_dst, log_dst = clean_path, log_path
        if not args.no_backup:
            _backup(clean_path)
            if write_log:
                _backup(log_path)
            print(f"  backups written (.bak)")
    else:
        clean_dst = out_dir / f"dpo_pair{args.out_suffix}.json"
        log_dst = out_dir / f"dpo_pair_log{args.out_suffix}.json"

    _write(new_clean, clean_dst, args, args.dry_run)
    print(f"  wrote clean  -> {clean_dst}")
    if write_log:
        _write(new_log, log_dst, args, args.dry_run)
        print(f"  wrote log    -> {log_dst}")


def main():
    args = parse_arguments()
    if args.task_name == "all":
        tasks = list(SUPPORTED_TASKS)
    elif args.task_name in SUPPORTED_TASKS:
        tasks = [args.task_name]
    else:
        raise SystemExit(
            f"--task_name must be one of {SUPPORTED_TASKS} or 'all'; "
            f"got '{args.task_name}'")

    mode = "DRY_RUN" if args.dry_run else ("INPLACE" if args.inplace
                                           else f"suffix='{args.out_suffix}'")
    print(f"dedup_pairs: tasks={tasks} mode={mode} exact={args.exact}")
    for t in tasks:
        process_task(t, args)
    print("\ndone.")


if __name__ == "__main__":
    main()
