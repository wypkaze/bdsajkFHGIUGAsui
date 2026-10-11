#!/usr/bin/env python3
"""
Pass A: score every query against every anchor and rank them.

For each query (one per test-user profile entry, NO filtering) the gold answer
y_w is fixed. We compute log p_anchor(y_w | prompt_{anchor,x}) for all anchors
(dynamic count read from anchor_user_idx.json), then rank anchors from high to
low and record Top-4 similar + Bottom-4 anti-anchor.

Strategy: anchor-major (each anchor's LoRA is loaded ONCE, then all queries are
scored in batches), multi-GPU sharding by anchor (--shard_id/--num_shards), and
per-anchor shard files for resume.

Products (under PAPPO_record/data/<task>/<out_subdir>/):
    queries.json                flat query list [{qid,user_id,entry_idx,x,y_w}]
    scores/anchor_<id>.json     {qid: logp}  (one shard per anchor)
    anchor_ranking.json         {qid: {ranking, top4_similar, bottom4_antianchor}}

Usage:
    python score_anchors.py --task_name movie_tagging --gpu_id 0 --max_users 2
    python score_anchors.py --task_name news_headline --gpu_id 0 --phase score --shard_id 0 --num_shards 4
    python score_anchors.py --task_name news_headline --phase rank
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
    DEFAULT_ANCHOR_IDX_DIR,
    DEFAULT_PROMPT_DIR,
    merged_model_path,
    anchor_lora_path,
    out_dir_for,
    load_json,
    save_json,
    load_anchor_map,
    load_prompt_pair,
    build_queries,
)
from src.generation.generator import AnchorPromptBuilder
from src.scoring.logprob import compute_log_prob_batch


def parse_arguments():
    p = argparse.ArgumentParser(description="Pass A: anchor log-prob scoring + ranking")
    p.add_argument('--task_name', type=str, required=True, choices=SUPPORTED_TASKS)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--k', type=int, default=1, help='BM25 retrieval count for anchor history')
    p.add_argument('--anchor_k', type=int, default=16, help='anchor_user_idx k-folder (k16)')
    p.add_argument('--cutoff_len', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--shard_id', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--phase', type=str, default='all', choices=['score', 'rank', 'all', 'queries'])
    p.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument('--ckpt_dir', type=str, default=str(DEFAULT_CKPT_DIR))
    p.add_argument('--anchor_idx_dir', type=str, default=str(DEFAULT_ANCHOR_IDX_DIR))
    p.add_argument('--prompt_dir', type=str, default=str(DEFAULT_PROMPT_DIR))
    p.add_argument('--out_subdir', type=str, default='pappo_anchor')
    p.add_argument('--max_users', type=int, default=None)
    p.add_argument('--tweet_x_source', type=str, default='detoned_x',
                   choices=['formal_x', 'detoned_x', 'corrected_x'],
                   help='tweet_paraphrase only: which rewrite product supplies the query x '
                        '(formalized_x.json / detoned_x.json / corrected_x.json)')
    return p.parse_args()


def get_queries(args, out_dir, write=False):
    """Build (or load) the flat query list; optionally persist queries.json."""
    queries_path = out_dir / "queries.json"
    test_users = load_json(str(Path(args.data_dir) / args.task_name / "test_100" / "user_test_100.json"))

    formalized = None
    x_source_field = "formal_x"
    cfg = get_task_config(args.task_name)
    if cfg.get("x_from_formalize"):
        src = getattr(args, "tweet_x_source", "detoned_x")
        src_map = {
            "formal_x":    ("formalized_x.json", "formal_x"),
            "detoned_x":   ("detoned_x.json",    "detoned_x"),
            "corrected_x": ("corrected_x.json",  "corrected_x"),
        }
        fname, x_source_field = src_map[src]
        fx_path = out_dir / fname
        if not fx_path.exists():
            raise SystemExit(
                f"tweet x source '{src}' requires {fx_path} (missing). Run the "
                f"corresponding pass first, or pass --tweet_x_source formal_x."
            )
        formalized = load_json(str(fx_path))
        print(f"[queries] tweet x source: {src} ({fname} -> field '{x_source_field}')")

    queries = build_queries(args.task_name, test_users, formalized,
                            args.max_users, x_source_field)
    if write:
        save_json(queries, queries_path)
    return queries


def run_score(args, out_dir):
    queries = get_queries(args, out_dir, write=(args.shard_id == 0))
    print(f"Queries: {len(queries)}")
    if not queries:
        return

    anchor_ids, anchor_map = load_anchor_map(
        args.task_name, args.data_dir, args.anchor_idx_dir, k=args.anchor_k
    )
    print(f"Anchors total: {len(anchor_ids)}")

    # This shard's anchors (round-robin split keeps shards balanced).
    shard_anchor_ids = anchor_ids[args.shard_id::args.num_shards]
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(shard_anchor_ids)} anchors")

    scores_dir = out_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    pair_tmpl = load_prompt_pair(args.prompt_dir)

    # --- Load merged model (4bit) + tokenizer, inference only (NO kbit training prep) ---
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
    print("Merged model loaded.")

    completions = [q["y_w"] for q in queries]
    qids = [q["qid"] for q in queries]

    from tqdm import tqdm
    for anchor_id in tqdm(shard_anchor_ids, desc=f"anchors(shard {args.shard_id})"):
        shard_path = scores_dir / f"anchor_{anchor_id}.json"
        if shard_path.exists():
            continue  # resume

        anchor_user = anchor_map.get(anchor_id)
        if anchor_user is None:
            print(f"  [skip] anchor {anchor_id}: no profile in candidate data")
            continue

        lora_path = anchor_lora_path(args.ckpt_dir, args.task_name, anchor_id)
        if not os.path.exists(lora_path):
            print(f"  [skip] anchor {anchor_id}: LoRA not found at {lora_path}")
            continue

        peft_model = PeftModel.from_pretrained(base_model, lora_path)
        peft_model.eval()
        try:
            builder = AnchorPromptBuilder(anchor_user, args.task_name, pair_tmpl, k=args.k)
            prompts = [builder.build(q["x"]) for q in queries]
            logps = compute_log_prob_batch(
                peft_model, tokenizer, prompts, completions, args.cutoff_len, args.batch_size
            )
            save_json({qid: lp for qid, lp in zip(qids, logps)}, shard_path)
        except Exception as e:
            print(f"  [error] anchor {anchor_id}: {e}")
        finally:
            del peft_model
            gc.collect()
            torch.cuda.empty_cache()

    print("Score phase done for this shard.")


def run_rank(args, out_dir):
    queries_path = out_dir / "queries.json"
    if not queries_path.exists():
        # Build it now (rank may be run standalone after all shards finish).
        get_queries(args, out_dir, write=True)
    queries = load_json(str(queries_path))
    qids = [q["qid"] for q in queries]

    scores_dir = out_dir / "scores"
    shard_files = sorted(scores_dir.glob("anchor_*.json"))
    if not shard_files:
        raise SystemExit(f"No score shards found under {scores_dir}. Run --phase score first.")

    # qid -> list of (anchor_id, logp)
    per_query = {qid: [] for qid in qids}
    n_anchors = 0
    for sf in shard_files:
        anchor_id = int(sf.stem.split("_", 1)[1])
        data = load_json(str(sf))
        n_anchors += 1
        for qid, logp in data.items():
            if qid in per_query:
                per_query[qid].append((anchor_id, logp))

    print(f"Merging {n_anchors} anchor shards for {len(qids)} queries")

    ranking_out = {}
    for qid in qids:
        pairs = sorted(per_query[qid], key=lambda t: t[1], reverse=True)
        ranking = [[aid, round(float(lp), 4)] for aid, lp in pairs]
        ranking_out[qid] = {
            "ranking": ranking,
            "top4_similar": ranking[:4],
            "bottom4_antianchor": ranking[-4:],
        }

    save_json(ranking_out, out_dir / "anchor_ranking.json")
    covered = [len(v["ranking"]) for v in ranking_out.values()]
    print(f"Ranking written. anchors/query: min={min(covered)} max={max(covered)}")


def main():
    args = parse_arguments()
    print(f"Using GPU: {args.gpu_id} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')})")
    out_dir = out_dir_for(args.data_dir, args.task_name, args.out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "queries":
        qs = get_queries(args, out_dir, write=True)
        print(f"[queries] wrote {len(qs)} queries -> {out_dir / 'queries.json'} "
              f"(CPU-only, no scoring)")
        return

    if args.phase in ("score", "all"):
        run_score(args, out_dir)
    if args.phase in ("rank", "all"):
        if args.num_shards > 1 and args.phase == "all":
            print("[warn] num_shards>1: run '--phase rank' separately AFTER all shards finish.")
        else:
            run_rank(args, out_dir)


if __name__ == "__main__":
    main()
