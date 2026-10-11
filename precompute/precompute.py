#!/usr/bin/env python3
"""
Precompute reference margins for PAPPO training.

This is the "precompute" stage extracted from the original precompute routine,
rebuilt on top of the PAPPO anchor pipeline.
For every DPO triple (x, y_w, y_l) it loads a small set of *reference anchors*
and records the anchor log-probs of y_w / y_l (raw + length-normalised), then
aggregates them into the PAPPO "sp_val" margin used later as
delta_ref_hat during PAPPO training.

Anchor selection (--anchor_selection), 4 anchors per query, taken from Pass A's
anchor_ranking.json:
    similar : top-4 most similar anchors   (ranking["top4_similar"])
    noisy   : bottom-4 anti-anchors        (ranking["bottom4_antianchor"])
    random  : 4 anchors sampled at random  (from the full 64/56 ranking)

Prompt (reference log-probs are computed on the TARGET-user prompt, exactly like
pappo.py): OPPU_input template + the target user's BM25-retrieved history built
from profile[:entry_idx] (causal). Only the anchor LoRA is swapped in.

Inputs (under PAPPO_record, produced by Pass A / Pass B):
    data/<task>/pappo_anchor/queries.json         qid -> {user_id,entry_idx,x,y_w}
    data/<task>/pappo_anchor/dpo_pair.json        {user_id: [{x,y_w,y_l}, ...]}
    data/<task>/pappo_anchor/anchor_ranking.json  qid -> {ranking,top4,bottom4}
    data/<task>/test_100/user_test_100.json       target-user profiles
    prompt/prompt.json                            OPPU_input/retrieval_* per task
    ckpts/anchor-train/<task>-anchor_train/{merged_model,user_<id>_anchor_lora}

Working cache (intermediate, resumable):
    data/<task>/pappo_anchor/precompute_<strategy>/prep.json
    data/<task>/pappo_anchor/precompute_<strategy>/ref_scores/anchor_<id>.json

Products (final, one file per test user):
    records/precompute/<task>/<strategy>/user_<uid>.json

Per-record fields:
    qid, user_id, entry_idx, anchor_selection, x, y_w, y_l, len_w, len_l
    anchors[]: anchor_id, ref_w, ref_l (raw), ref_w_norm, ref_l_norm,
               delta_ref_k[_norm] = ref_w - ref_l,
               lambda_pappo[_norm] = ReLU(delta)
    sp_val_pappo_unorm, sp_val_pappo_norm

Usage:
    python precompute.py --task_name citation --anchor_selection noisy --gpu_id 0
    # multi-GPU (shard by anchor), then assemble once:
    python precompute.py --task_name citation --anchor_selection similar --phase prep
    python precompute.py --task_name citation --anchor_selection similar --phase score --gpu_id 0 --shard_id 0 --num_shards 4
    python precompute.py --task_name citation --anchor_selection similar --phase assemble
"""

import os
import sys

# --- env MUST be set before importing torch (mirror score_anchors.py) ---
_gpu_id = 0
for i, arg in enumerate(sys.argv):
    if arg == '--gpu_id' and i + 1 < len(sys.argv):
        _gpu_id = sys.argv[i + 1]
        break
os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_id)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import random
import argparse
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

_ROOT = Path(__file__).resolve().parent.parent  # PAPPO/
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.config import SUPPORTED_TASKS, get_task_config
from src.pipeline.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_CKPT_DIR,
    DEFAULT_PROMPT_DIR,
    merged_model_path,
    anchor_lora_path,
    out_dir_for,
    load_json,
    save_json,
)
from src.pipeline.prompts import format_history_lines, build_target_prompt
from src.scoring.logprob import compute_log_prob_batch

# records live next to data/ under PAPPO_record (never hardcoded).
DEFAULT_RECORDS_DIR = Path(DEFAULT_DATA_DIR).parent / "records"

# log-prob sentinel: anything <= this is treated as an invalid/empty completion.
_NEG_INF = -1e8


def parse_arguments():
    p = argparse.ArgumentParser(description="Precompute PAPPO reference margins")
    p.add_argument('--task_name', type=str, required=True, choices=SUPPORTED_TASKS)
    p.add_argument('--anchor_selection', type=str, required=True,
                   choices=['random', 'noisy', 'similar'],
                   help='How to pick the 4 reference anchors per query')
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--n_anchors', type=int, default=4, help='Reference anchors per query')
    p.add_argument('--k', type=int, default=1,
                   help='BM25 retrieval count for the TARGET-user history')
    p.add_argument('--cutoff_len', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42, help='Seed for --anchor_selection random')
    p.add_argument('--shard_id', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--phase', type=str, default='all',
                   choices=['prep', 'score', 'assemble', 'all'])
    p.add_argument('--max_users', type=int, default=None, help='Only first N test users')
    p.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument('--ckpt_dir', type=str, default=str(DEFAULT_CKPT_DIR))
    p.add_argument('--prompt_dir', type=str, default=str(DEFAULT_PROMPT_DIR))
    p.add_argument('--records_dir', type=str, default=str(DEFAULT_RECORDS_DIR))
    p.add_argument('--out_subdir', type=str, default='pappo_anchor',
                   help='Pass A/B artifact subdir under data/<task>/')
    return p.parse_args()


# ===================== target-user prompt (see src/pipeline/prompts.py) =====================


def select_anchor_ids(strategy, ranking_entry, seed, qid, n):
    """Pick <=n reference anchor ids for one query under the given strategy."""
    if strategy == "similar":
        ids = [a[0] for a in ranking_entry.get("top4_similar", [])]
    elif strategy == "noisy":
        ids = [a[0] for a in ranking_entry.get("bottom4_antianchor", [])]
    else:  # random
        all_ids = [a[0] for a in ranking_entry.get("ranking", [])]
        if len(all_ids) <= n:
            ids = list(all_ids)
        else:
            rng = random.Random(f"{seed}::{qid}")  # deterministic per qid
            ids = rng.sample(all_ids, n)
    return ids[:n]


# ===================== aggregation formulas =====================

def sp_val_pappo(deltas):
    """PAPPO delta_ref_hat: ReLU-gated weighted sum of the anchor margins.

    lambda_k = ReLU(delta_k); sp = sum_k (lambda_k / sum_i lambda_i) * delta_k
    (0 when every anchor is ReLU-truncated).
    """
    lambdas = [max(0.0, d) for d in deltas]
    total = sum(lambdas)
    if total <= 0:
        return 0.0
    return sum((lk / total) * dk for lk, dk in zip(lambdas, deltas))


# ===================== Phase: prep (CPU) =====================

def run_prep(args, work_dir):
    """Build per-qid state: target-user prompt, y_l list, selected anchors, lengths."""
    work_dir.mkdir(parents=True, exist_ok=True)
    pa_dir = out_dir_for(args.data_dir, args.task_name, args.out_subdir)

    queries = load_json(str(pa_dir / "queries.json"))
    dpo_pairs = load_json(str(pa_dir / "dpo_pair.json"))
    ranking = load_json(str(pa_dir / "anchor_ranking.json"))
    test_users = load_json(str(Path(args.data_dir) / args.task_name / "test_100" / "user_test_100.json"))
    prompt_json = load_json(str(Path(args.prompt_dir) / "prompt.json"))
    prompt_tmpl = prompt_json[args.task_name]
    x_field = get_task_config(args.task_name)["x_field"]

    # Tokenizer (CPU only) for answer token lengths, matching pappo.py's
    # len_w = len(tokenizer.encode(y_w, add_special_tokens=False)).
    mm_path = merged_model_path(args.ckpt_dir, args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(mm_path, trust_remote_code=True, use_fast=True)

    def tok_len(text):
        return max(len(tokenizer.encode(str(text), add_special_tokens=False)), 1)

    # (user_id, x, y_w) -> (qid, entry_idx); first match wins (dpo_pair.json
    # carries no entry_idx, so this join recovers it for the causal history).
    key2qid = {}
    for q in queries:
        key = (q["user_id"], q["x"], q["y_w"])
        if key not in key2qid:
            # queries.json stores entry_idx as a string; coerce to int so it can
            # slice the causal history (profile[:entry_idx]) and sort records.
            try:
                eidx = int(q["entry_idx"])
            except (TypeError, ValueError):
                eidx = 0
            key2qid[key] = (q["qid"], eidx)

    profile_by_user = {str(u["user_id"]): (u.get("profile", []) or []) for u in test_users}

    user_ids = sorted(dpo_pairs.keys())
    if args.max_users is not None:
        user_ids = user_ids[:args.max_users]

    data = {}
    n_noqid = n_norank = 0
    for uid in user_ids:
        profile_list = profile_by_user.get(str(uid), [])
        hist_lines = format_history_lines(profile_list, prompt_tmpl["retrieval_history"])

        # Group this user's triples by qid (similar + dissimilar y_l share a qid).
        per_qid = {}
        for t in dpo_pairs[uid]:
            got = key2qid.get((uid, t["x"], t["y_w"]))
            if got is None:
                n_noqid += 1
                continue
            qid, entry_idx = got
            slot = per_qid.get(qid)
            if slot is None:
                slot = per_qid[qid] = {"entry_idx": entry_idx, "x": t["x"],
                                       "y_w": t["y_w"], "y_l_list": []}
            y_l = str(t.get("y_l", "")).strip()
            if y_l:
                slot["y_l_list"].append({"y_l": y_l, "len_l": tok_len(y_l)})

        for qid, slot in per_qid.items():
            if not slot["y_l_list"]:
                continue
            rank_entry = ranking.get(qid)
            if rank_entry is None:
                n_norank += 1
                continue
            entry_idx = slot["entry_idx"]
            visible = [l for l in hist_lines[:entry_idx] if l is not None]
            prompt = build_target_prompt(args.task_name, slot["x"], visible,
                                         prompt_tmpl, args.k, x_field)
            data[qid] = {
                "user_id": str(uid),
                "entry_idx": entry_idx,
                "x": slot["x"],
                "y_w": slot["y_w"],
                "len_w": tok_len(slot["y_w"]),
                "prompt": prompt,
                "y_l_list": slot["y_l_list"],
                "anchor_ids": select_anchor_ids(args.anchor_selection, rank_entry,
                                                args.seed, qid, args.n_anchors),
            }

    prep = {
        "task_name": args.task_name,
        "anchor_selection": args.anchor_selection,
        "n_anchors": args.n_anchors,
        "k": args.k,
        "seed": args.seed,
        "cutoff_len": args.cutoff_len,
        "data": data,
    }
    save_json(prep, work_dir / "prep.json")
    n_sel = len({a for rec in data.values() for a in rec["anchor_ids"]})
    print(f"[prep] qids={len(data)} selected_anchors={n_sel} "
          f"skipped(no_qid={n_noqid}, no_rank={n_norank}) -> {work_dir / 'prep.json'}")


# ===================== Phase: score (GPU, anchor-major, shardable) =====================

def build_anchor_index(data):
    """anchor_id -> list of qids it must score (inverted from per-qid selection)."""
    anchor2qids = defaultdict(list)
    for qid, rec in data.items():
        for aid in rec["anchor_ids"]:
            anchor2qids[aid].append(qid)
    return anchor2qids


def run_score(args, work_dir):
    prep_path = work_dir / "prep.json"
    if not prep_path.exists():
        print("[score] prep.json missing -> running prep first")
        run_prep(args, work_dir)
    data = load_json(str(prep_path))["data"]

    anchor2qids = build_anchor_index(data)
    union = sorted(anchor2qids.keys())
    shard = union[args.shard_id::args.num_shards]
    print(f"[score] anchors total={len(union)} shard {args.shard_id}/{args.num_shards}={len(shard)}")
    if not shard:
        return

    ref_dir = work_dir / "ref_scores"
    ref_dir.mkdir(parents=True, exist_ok=True)

    mm_path = merged_model_path(args.ckpt_dir, args.task_name)
    print(f"[score] loading merged model: {mm_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        mm_path, trust_remote_code=True, padding_side="left", use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        mm_path, quantization_config=quantization_config,
        device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    base_model.config.use_cache = True
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.eval()

    from tqdm import tqdm
    for aid in tqdm(shard, desc=f"ref(shard {args.shard_id})"):
        cache = ref_dir / f"anchor_{aid}.json"
        if cache.exists():
            continue  # resume
        lora_path = anchor_lora_path(args.ckpt_dir, args.task_name, aid)
        if not os.path.exists(lora_path):
            print(f"  [skip] anchor {aid}: LoRA not found at {lora_path}")
            continue

        qids = anchor2qids[aid]
        peft_model = PeftModel.from_pretrained(base_model, lora_path)
        peft_model.eval()
        try:
            prompts_w = [data[q]["prompt"] for q in qids]
            comps_w = [data[q]["y_w"] for q in qids]
            prompts_l, comps_l = [], []
            for q in qids:
                for yl in data[q]["y_l_list"]:
                    prompts_l.append(data[q]["prompt"])
                    comps_l.append(yl["y_l"])

            scores = compute_log_prob_batch(
                peft_model, tokenizer, prompts_w + prompts_l, comps_w + comps_l,
                args.cutoff_len, args.batch_size, length_normalized=False,
            )
            rw = scores[:len(qids)]
            rl = scores[len(qids):]

            out = {q: {"ref_w": rw[i], "yls": []} for i, q in enumerate(qids)}
            idx = 0
            for q in qids:
                for yl in data[q]["y_l_list"]:
                    out[q]["yls"].append({"y_l": yl["y_l"], "ref_l": rl[idx]})
                    idx += 1
            save_json(out, cache)
        except Exception as e:
            print(f"  [error] anchor {aid}: {e}")
        finally:
            del peft_model
            gc.collect()
            torch.cuda.empty_cache()

    print("[score] done for this shard.")


# ===================== Phase: assemble (CPU) =====================

def run_assemble(args, work_dir):
    prep_path = work_dir / "prep.json"
    if not prep_path.exists():
        raise SystemExit(f"[assemble] missing {prep_path}; run --phase prep/score first.")
    prep = load_json(str(prep_path))
    data = prep["data"]

    ref_dir = work_dir / "ref_scores"
    caches = {}
    for f in sorted(ref_dir.glob("anchor_*.json")):
        aid = int(f.stem.split("_", 1)[1])
        caches[aid] = load_json(str(f))
    if not caches:
        raise SystemExit(f"[assemble] no anchor caches under {ref_dir}; run --phase score first.")
    print(f"[assemble] loaded {len(caches)} anchor caches")

    records_out = Path(args.records_dir) / "precompute" / args.task_name / args.anchor_selection
    records_out.mkdir(parents=True, exist_ok=True)

    by_user = defaultdict(list)
    n_rec = n_skip = 0
    for qid, rec in data.items():
        len_w = rec["len_w"]
        anchor_ids = rec["anchor_ids"]
        for lj, yl in enumerate(rec["y_l_list"]):
            y_l, len_l = yl["y_l"], yl["len_l"]
            anchors_detail = []
            rw_u, rl_u, rw_n, rl_n = [], [], [], []
            for aid in anchor_ids:
                c = caches.get(aid, {}).get(qid)
                if c is None:
                    continue
                yls = c.get("yls", [])
                if lj >= len(yls):
                    continue
                ref_w = c.get("ref_w")
                ref_l = yls[lj].get("ref_l")
                if ref_w is None or ref_l is None:
                    continue
                if ref_w <= _NEG_INF or ref_l <= _NEG_INF:
                    continue
                ref_w_norm = ref_w / len_w
                ref_l_norm = ref_l / len_l
                d_u = ref_w - ref_l
                d_n = ref_w_norm - ref_l_norm
                anchors_detail.append({
                    "anchor_id": aid,
                    "ref_w": round(ref_w, 6), "ref_l": round(ref_l, 6),
                    "ref_w_norm": round(ref_w_norm, 6), "ref_l_norm": round(ref_l_norm, 6),
                    "delta_ref_k": round(d_u, 6),
                    "delta_ref_k_norm": round(d_n, 6),
                    "lambda_pappo": round(max(0.0, d_u), 6),
                    "lambda_pappo_norm": round(max(0.0, d_n), 6),
                })
                rw_u.append(ref_w); rl_u.append(ref_l)
                rw_n.append(ref_w_norm); rl_n.append(ref_l_norm)

            if not anchors_detail:
                n_skip += 1
                continue

            by_user[rec["user_id"]].append({
                "qid": qid,
                "user_id": rec["user_id"],
                "entry_idx": rec["entry_idx"],
                "anchor_selection": args.anchor_selection,
                "x": rec["x"], "y_w": rec["y_w"], "y_l": y_l,
                "len_w": len_w, "len_l": len_l,
                "anchors": anchors_detail,
                "sp_val_pappo_unorm": round(sp_val_pappo([a - b for a, b in zip(rw_u, rl_u)]), 6),
                "sp_val_pappo_norm": round(sp_val_pappo([a - b for a, b in zip(rw_n, rl_n)]), 6),
            })
            n_rec += 1

    for uid, recs in by_user.items():
        recs.sort(key=lambda r: (r["entry_idx"], r["y_l"]))
        save_json(recs, records_out / f"user_{uid}.json")
    print(f"[assemble] users={len(by_user)} records={n_rec} skipped={n_skip} -> {records_out}")


# ===================== Main =====================

def main():
    args = parse_arguments()
    print(f"Using GPU: {args.gpu_id} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')})")
    print(f"task={args.task_name} anchor_selection={args.anchor_selection} phase={args.phase}")

    work_dir = out_dir_for(args.data_dir, args.task_name, args.out_subdir) / \
        f"precompute_{args.anchor_selection}"
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in ("prep", "all"):
        run_prep(args, work_dir)
    if args.phase in ("score", "all"):
        run_score(args, work_dir)
    if args.phase in ("assemble", "all"):
        if args.num_shards > 1 and args.phase == "all":
            print("[warn] num_shards>1: run '--phase assemble' separately AFTER all shards finish.")
        else:
            run_assemble(args, work_dir)


if __name__ == "__main__":
    main()
