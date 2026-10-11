#!/usr/bin/env python3
"""
Pass B: generate y_l for the Top-1 (most similar) and Bottom-1 (least similar)
anchor of every query, using each anchor's OWN history.

- Reads queries.json + anchor_ranking.json (from Pass A).
- For each query: top1 = ranking[0] (similar), bottom1 = ranking[-1] (dissimilar).
- anchor-major: each selected anchor's LoRA is loaded ONCE; per-anchor results
  are cached under gen/anchor_<id>.json for resume / multi-GPU sharding.
- movie_tagging uses the special 15-tag branch: rank tags by log p_anchor(tag|prompt)
  and take the FIRST tag != y_w as y_l (structurally guarantees y_l != y_w).
- Other tasks: batched free generation, cleaned by marker (clean_generation).

Products (under PAPPO_record/data/<task>/<out_subdir>/):
    gen/anchor_<id>.json   {qid: {"y_l":..., "tag_ranking":[...] (movie only)}}
    dpo_pair.json          CLEAN training record: {user_id: [{x, y_w, y_l}, ...]}
    dpo_pair_log.json      VERBOSE log: {user_id: [{entry_idx, x, y_w, pairs:[
                            {anchor_id, anchor_type, rank, logp, y_l,
                             tag_ranking (movie, BOTH similar & dissimilar)}]}]}

Usage:
    python gen_pairs.py --task_name movie_tagging --gpu_id 0 --max_users 2
    python gen_pairs.py --task_name citation --gpu_id 0 --phase gen --shard_id 0 --num_shards 4
    python gen_pairs.py --task_name citation --phase assemble
"""

import os
import sys

# --- env MUST be set before importing torch (mirror sft_target_user.py L15-22) ---
_gpu_id = 0
for i, arg in enumerate(sys.argv):
    if arg == '--gpu_id' and i + 1 < len(sys.argv):
        _gpu_id = sys.argv[i + 1]
        break
os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_id)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import argparse
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

_ROOT = Path(__file__).resolve().parent.parent  # PAPPO/
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.config import SUPPORTED_TASKS, get_task_config, MOVIE_TAGS
from src.pipeline.common import (
    DEFAULT_DATA_DIR,
    DEFAULT_CKPT_DIR,
    DEFAULT_ANCHOR_IDX_DIR,
    DEFAULT_PROMPT_DIR,
    merged_model_path,
    anchor_lora_path,
    out_dir_for,
    load_json,
    save_json,
    load_anchor_map,
    load_prompt_pair,
)
from src.generation.generator import AnchorPromptBuilder, generate_batch
from src.scoring.logprob import score_candidates


def parse_arguments():
    p = argparse.ArgumentParser(description="Pass B: generate y_l for Top-1/Bottom-1 anchors")
    p.add_argument('--task_name', type=str, required=True, choices=SUPPORTED_TASKS)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--k', type=int, default=1, help='BM25 retrieval count for anchor history')
    p.add_argument('--anchor_k', type=int, default=16)
    p.add_argument('--cutoff_len', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=8, help='Generation batch size')
    p.add_argument('--max_new_tokens', type=int, default=None,
                   help='Override TASK_CONFIG max_new_tokens (optional)')
    p.add_argument('--shard_id', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--phase', type=str, default='all', choices=['gen', 'assemble', 'all'])
    p.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument('--ckpt_dir', type=str, default=str(DEFAULT_CKPT_DIR))
    p.add_argument('--anchor_idx_dir', type=str, default=str(DEFAULT_ANCHOR_IDX_DIR))
    p.add_argument('--prompt_dir', type=str, default=str(DEFAULT_PROMPT_DIR))
    p.add_argument('--out_subdir', type=str, default='pappo_anchor')
    p.add_argument('--max_users', type=int, default=None,
                   help='Only assemble queries of the first N users (debug)')
    return p.parse_args()


def load_state(out_dir):
    """Load queries (qid->query) and ranking (qid->ranking list)."""
    queries = load_json(str(out_dir / "queries.json"))
    qmap = {q["qid"]: q for q in queries}
    ranking_all = load_json(str(out_dir / "anchor_ranking.json"))
    # ranking_all[qid] = {"ranking": [[aid, logp], ...], ...}
    rmap = {qid: v["ranking"] for qid, v in ranking_all.items()}
    return qmap, rmap


def selected_anchor_qids(rmap):
    """
    Build anchor_id -> set(qids) for anchors chosen as top1 or bottom1.
    Returns (anchor2qids, per_query_sel) where per_query_sel[qid] =
        {"similar": (aid, rank, logp), "dissimilar": (aid, rank, logp)}.
    """
    anchor2qids = defaultdict(set)
    per_query_sel = {}
    for qid, ranking in rmap.items():
        if not ranking:
            continue
        top_aid, top_lp = ranking[0][0], ranking[0][1]
        bot_aid, bot_lp = ranking[-1][0], ranking[-1][1]
        sel = {
            "similar": (top_aid, 0, top_lp),
            "dissimilar": (bot_aid, len(ranking) - 1, bot_lp),
        }
        per_query_sel[qid] = sel
        anchor2qids[top_aid].add(qid)
        anchor2qids[bot_aid].add(qid)
    return anchor2qids, per_query_sel


def run_gen(args, out_dir):
    qmap, rmap = load_state(out_dir)
    anchor2qids, _ = selected_anchor_qids(rmap)
    cfg = get_task_config(args.task_name)

    gen_dir = out_dir / "gen"
    gen_dir.mkdir(parents=True, exist_ok=True)

    all_anchor_ids = sorted(anchor2qids.keys())
    shard_anchors = all_anchor_ids[args.shard_id::args.num_shards]
    print(f"[gen] selected anchors: {len(all_anchor_ids)}; this shard: {len(shard_anchors)}")
    if not shard_anchors:
        return

    _, anchor_map = load_anchor_map(
        args.task_name, args.data_dir, args.anchor_idx_dir, k=args.anchor_k,
    )
    pair_tmpl = load_prompt_pair(args.prompt_dir)

    # --- Load merged model (4bit), inference only ---
    mm_path = merged_model_path(args.ckpt_dir, args.task_name)
    print(f"Loading merged model: {mm_path}")
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

    gen_params = dict(cfg["gen"]) if cfg["gen"] else {}
    if args.max_new_tokens is not None:
        gen_params["max_new_tokens"] = args.max_new_tokens

    from tqdm import tqdm
    for anchor_id in tqdm(shard_anchors, desc=f"gen(shard {args.shard_id})"):
        cache_path = gen_dir / f"anchor_{anchor_id}.json"
        if cache_path.exists():
            continue  # resume

        anchor_user = anchor_map.get(anchor_id)
        if anchor_user is None:
            print(f"  [skip] anchor {anchor_id}: no profile")
            continue
        lora_path = anchor_lora_path(args.ckpt_dir, args.task_name, anchor_id)
        if not os.path.exists(lora_path):
            print(f"  [skip] anchor {anchor_id}: LoRA not found")
            continue

        qids = sorted(anchor2qids[anchor_id])
        peft_model = PeftModel.from_pretrained(base_model, lora_path)
        peft_model.eval()
        result = {}
        try:
            builder = AnchorPromptBuilder(anchor_user, args.task_name, pair_tmpl, k=args.k)

            if cfg.get("special") == "tag_rank":
                # movie_tagging: rank the 15 tags by length-normalised
                # log p_anchor(tag|prompt) (avg per token, so longer tags are
                # not penalised), then take the first tag != y_w as y_l.
                for qid in qids:
                    q = qmap[qid]
                    prompt = builder.build(q["x"])
                    tag_scores = score_candidates(
                        peft_model, tokenizer, prompt, MOVIE_TAGS, args.cutoff_len,
                        args.batch_size, length_normalized=True,
                    )
                    order = sorted(range(len(MOVIE_TAGS)), key=lambda i: tag_scores[i], reverse=True)
                    ranked_tags = [[MOVIE_TAGS[i], round(tag_scores[i], 4)] for i in order]
                    y_l = next((MOVIE_TAGS[i] for i in order if MOVIE_TAGS[i] != q["y_w"]), MOVIE_TAGS[order[0]])
                    result[qid] = {"y_l": y_l, "tag_ranking": ranked_tags}
            else:
                prompts = [builder.build(qmap[qid]["x"]) for qid in qids]
                y_ls = generate_batch(
                    peft_model, tokenizer, prompts, args.task_name, gen_params,
                    pair_tmpl, args.batch_size,
                )
                for qid, y_l in zip(qids, y_ls):
                    result[qid] = {"y_l": y_l}

            save_json(result, cache_path)
        except Exception as e:
            print(f"  [error] anchor {anchor_id}: {e}")
        finally:
            del peft_model
            gc.collect()
            torch.cuda.empty_cache()

    print("[gen] done for this shard.")


def run_assemble(args, out_dir):
    qmap, rmap = load_state(out_dir)
    _, per_query_sel = selected_anchor_qids(rmap)

    gen_dir = out_dir / "gen"
    gen_cache = {}
    for gf in gen_dir.glob("anchor_*.json"):
        aid = int(gf.stem.split("_", 1)[1])
        gen_cache[aid] = load_json(str(gf))

    log_by_user = defaultdict(list)     # verbose -> dpo_pair_log.json
    clean_by_user = defaultdict(list)   # minimal {x, y_w, y_l} -> dpo_pair.json
    n_missing = 0
    for qid, q in qmap.items():
        sel = per_query_sel.get(qid)
        if not sel:
            continue
        detail_pairs = []
        clean_triples = []
        for role in ("similar", "dissimilar"):
            aid, rank, logp = sel[role]
            rec = gen_cache.get(aid, {}).get(qid, {})
            y_l = rec.get("y_l")
            if y_l is None or not str(y_l).strip():
                n_missing += 1
                continue
            pair = {
                "anchor_id": aid,
                "anchor_type": role,
                "rank": rank,
                "logp": logp,
                "y_l": y_l,
            }
            # Attach the 15-tag ranking for BOTH roles when present (movie_tagging):
            # the dissimilar anchor is scored exactly like the similar one.
            if "tag_ranking" in rec:
                pair["tag_ranking"] = rec["tag_ranking"]
            detail_pairs.append(pair)
            clean_triples.append({"x": q["x"], "y_w": q["y_w"], "y_l": y_l})
        if not detail_pairs:
            continue
        log_by_user[q["user_id"]].append({
            "entry_idx": q["entry_idx"],
            "x": q["x"],
            "y_w": q["y_w"],
            "pairs": detail_pairs,
        })
        # qmap iterates in queries.json order (user-major, entry-ascending),
        # so triples are already appended in stable entry order.
        clean_by_user[q["user_id"]].extend(clean_triples)

    # Optional user limit for debugging assemblies.
    if args.max_users is not None:
        keep = list(log_by_user.keys())[:args.max_users]
        log_by_user = {u: log_by_user[u] for u in keep}
        clean_by_user = {u: clean_by_user[u] for u in keep if u in clean_by_user}

    # Sort each user's log entries by entry_idx for stable output.
    log_result = {u: sorted(v, key=lambda e: e["entry_idx"]) for u, v in log_by_user.items()}
    save_json(log_result, out_dir / "dpo_pair_log.json")
    save_json(dict(clean_by_user), out_dir / "dpo_pair.json")
    total_pairs = sum(len(e["pairs"]) for v in log_result.values() for e in v)
    total_triples = sum(len(v) for v in clean_by_user.values())
    print(f"[assemble] users={len(log_result)} entries="
          f"{sum(len(v) for v in log_result.values())} pairs={total_pairs} "
          f"clean_triples={total_triples} missing_y_l={n_missing}")


def main():
    args = parse_arguments()
    print(f"Using GPU: {args.gpu_id} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')})")
    out_dir = out_dir_for(args.data_dir, args.task_name, args.out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in ("gen", "all"):
        run_gen(args, out_dir)
    if args.phase in ("assemble", "all"):
        if args.num_shards > 1 and args.phase == "all":
            print("[warn] num_shards>1: run '--phase assemble' separately AFTER all shards finish.")
        else:
            run_assemble(args, out_dir)


if __name__ == "__main__":
    main()
