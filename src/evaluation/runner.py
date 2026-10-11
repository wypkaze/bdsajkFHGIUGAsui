"""
Evaluation driver for the personalized baselines.

Refactor of the original evaluation script into the current PAPPO layout. It runs
inference on every test user's queries with that user's LoRA (or the shared
task_LoRA / merged base), then:

  1. saves each test user's answers to
     records/PAPPO_baselines/<METHOD>/<task>/responses/<eval_suffix>/<run_tag>/user_<uid>.json
  2. computes per-user metrics from those responses into
     records/PAPPO_baselines/<METHOD>/<task>/metrics/<eval_suffix>/<run_tag>/user_<uid>.json
  3. writes a macro-averaged summary at .../metrics/<eval_suffix>/<run_tag>/average.json

`<run_tag>` is a `%Y%m%d_%H%M%S` timestamp (overridable via --run_tag), so repeated
runs never overwrite each other. `--metric_only` recomputes 2/3 from an existing
responses dir without touching the model.

Prompts are built with the reused `src.data.processor.DataProcessor` (the LaMP-style
eval prompt from prompt.json), matching the gold answer format for each task.
"""

import argparse
import gc
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from ..data.processor import DataProcessor
from ..pipeline.common import (
    DEFAULT_CKPT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_PROMPT_DIR,
    REPO_ROOT,
    load_json,
    merged_model_path,
    save_json,
)
from ..utils.logging_utils import setup_logging
from .metrics import (
    GENERATIVE_TASKS,
    BertScorer,
    average_metrics,
    compute_user_metrics,
    task_family,
)

# Repo-relative roots (never hardcoded), mirroring baseline_trainer.py.
CKPTS_ROOT = Path(DEFAULT_DATA_DIR).parent / "ckpts"                 # PAPPO_record/ckpts
RECORDS_ROOT = Path(DEFAULT_DATA_DIR).parent / "records"             # PAPPO_record/records
BASELINES_ROOT = RECORDS_ROOT / "PAPPO_baselines"
DEFAULT_DEBERTA = REPO_ROOT / "assets" / "deberta"

ALL_TASKS = ["citation", "movie_tagging", "news_headline",
             "product_rating", "scholarly_title", "tweet_paraphrase"]


class MethodSpec:
    """How a --method maps to its checkpoint layout and records dir name."""

    def __init__(self, arg: str, record_name: str, kind: str):
        self.arg = arg                # CLI spelling
        self.record_name = record_name  # dir under ckpts/ and PAPPO_baselines/
        self.kind = kind              # "per_user" | "single" | "none"


METHOD_SPECS: Dict[str, MethodSpec] = {
    "base":      MethodSpec("base", "base", "none"),
    "task_lora": MethodSpec("task_lora", "task_LoRA", "single"),
    "oppu":      MethodSpec("oppu", "OPPU", "per_user"),
    "pappo":     MethodSpec("pappo", "PAPPO", "per_user"),
}


# ============================ CLI ============================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate personalized baseline models (per-user responses + metrics)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--task_name', type=str, required=True, choices=ALL_TASKS)
    p.add_argument('--method', type=str, required=True, choices=sorted(METHOD_SPECS),
                   help='base=merged model only; task_lora=shared adapter; '
                        'oppu/pappo=per-user adapters')
    # retrieval / profile configuration -> picks the eval_suffix subdir
    p.add_argument('--k', type=int, default=1, help='BM25 history count (0=disabled -> k0)')
    p.add_argument('--pag', action='store_true', default=False,
                   help='Profile-augmented generation: use profile-id2text.json (-> pag), overrides --k')
    # generation
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--max_users', type=int, default=None)
    p.add_argument('--lora_path', type=str, default=None,
                   help='Override adapter dir for --method task_lora (single adapter)')
    # metric options
    p.add_argument('--no_bertscore', action='store_true', default=False,
                   help='Skip BERTScore for generative tasks (ROUGE only)')
    p.add_argument('--deberta_path', type=str, default=str(DEFAULT_DEBERTA))
    # run control
    p.add_argument('--metric_only', action='store_true', default=False,
                   help='Recompute metrics from an existing responses dir; never loads the model')
    p.add_argument('--run_tag', type=str, default=None,
                   help='Version tag for responses/metrics subdirs (default: current timestamp; '
                        'for --metric_only: latest existing responses run)')
    p.add_argument('--gpu_id', type=int, default=0)
    # paths
    p.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument('--prompt_dir', type=str, default=str(DEFAULT_PROMPT_DIR))
    p.add_argument('--ckpt_dir', type=str, default=str(DEFAULT_CKPT_DIR),
                   help='Dir holding <task>-anchor_train/merged_model (base model)')
    p.add_argument('--records_dir', type=str, default=str(BASELINES_ROOT),
                   help='Root of PAPPO_baselines records (responses/metrics)')
    return p.parse_args()


# ============================ path helpers ============================
def resolve_eval_suffix(args) -> str:
    """pag -> 'pag'; k==0 -> 'k0'; k>=1 -> 'rag' (retrieval-augmented)."""
    if args.pag:
        return "pag"
    return "k0" if args.k == 0 else "rag"


def _is_run_tag(name: str) -> bool:
    return re.fullmatch(r"\d{8}_\d{6}", name) is not None


def latest_run_tag(parent: Path) -> Optional[str]:
    """Most recent timestamp subdir under `parent` (None if absent)."""
    if not parent.exists():
        return None
    tags = sorted(d.name for d in parent.iterdir() if d.is_dir() and _is_run_tag(d.name))
    return tags[-1] if tags else None


def resolve_run_tag(args, eval_suffix: str, records_base: Path) -> str:
    if args.run_tag:
        return args.run_tag
    if args.metric_only:
        parent = records_base / "responses" / eval_suffix
        tag = latest_run_tag(parent)
        if tag is None:
            raise SystemExit(f"--metric_only but no responses run found under {parent}")
        return tag
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_single_lora(spec: MethodSpec, task: str, args, logger) -> Optional[str]:
    """Locate the single shared adapter (task_LoRA): --lora_path or newest *_ckpt."""
    if args.lora_path:
        return args.lora_path
    root = CKPTS_ROOT / spec.record_name / f"{task}-{spec.record_name}"
    if not root.exists():
        logger.error(f"task_LoRA ckpt dir not found: {root}")
        return None
    ckpts = [d for d in root.glob("*_ckpt") if (d / "adapter_config.json").exists()]
    if not ckpts:
        logger.error(f"no *_ckpt adapter under {root}")
        return None
    return str(max(ckpts, key=lambda d: d.stat().st_mtime))


# ============================ model ============================
def load_base_model(merged_path: str, logger):
    if not Path(merged_path, "config.json").exists():
        raise FileNotFoundError(
            f"merged model not found at {merged_path}; produce it via anchor/anchor_train.py"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        merged_path, trust_remote_code=True, padding_side="left", use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        merged_path, quantization_config=quant, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    logger.info(f"base model loaded from {merged_path}")
    return tokenizer, model


@torch.no_grad()
def generate_batch(gen_model, tokenizer, prompts: List[str], max_new_tokens: int,
                   do_sample: bool) -> List[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=2048).to(gen_model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs.update(temperature=0.7, top_p=0.9)
    else:
        gen_kwargs.update(temperature=None, top_p=None)
    outputs = gen_model.generate(**inputs, **gen_kwargs)
    generated = outputs[:, inputs['input_ids'].shape[1]:]
    return [d.strip() for d in tokenizer.batch_decode(generated, skip_special_tokens=True)]


# ============================ phase 1: responses ============================
def run_generation(args, spec: MethodSpec, eval_suffix: str, run_tag: str,
                   responses_dir: Path, logger) -> None:
    task = args.task_name
    data_dir = Path(args.data_dir)

    test_users = load_json(str(data_dir / task / "test_100" / "user_test_100.json"))
    labels_data = load_json(str(data_dir / task / "test_100" / "user_test_100_label.json"))
    label_map = {g["id"]: g["output"] for g in labels_data["golds"]}
    prompt_tmpl = load_json(str(Path(args.prompt_dir) / "prompt.json"))

    if args.pag:
        profile_path = data_dir / task / "profile-id2text.json"
        if not profile_path.exists():
            raise FileNotFoundError(f"--pag set but profile not found: {profile_path}")
        all_profiles = load_json(str(profile_path))
        k, add_profile = 0, True
        logger.info(f"pag mode: loaded {len(all_profiles)} profiles from {profile_path}")
    else:
        all_profiles = {}
        k, add_profile = args.k, False

    processor = DataProcessor(task_name=task, prompt_template=prompt_tmpl,
                              k=k, add_profile=add_profile)
    use_sampling = task in GENERATIVE_TASKS

    if args.max_users is not None:
        test_users = test_users[:args.max_users]
    logger.info(f"evaluating {len(test_users)} users | eval_suffix={eval_suffix} "
                f"k={k} add_profile={add_profile} sampling={use_sampling}")

    tokenizer, base_model = load_base_model(merged_model_path(args.ckpt_dir, task), logger)

    persistent = None
    if spec.kind == "single":
        lora = resolve_single_lora(spec, task, args, logger)
        if lora is None:
            raise SystemExit(f"cannot resolve task_LoRA adapter for {task}")
        persistent = PeftModel.from_pretrained(base_model, lora)
        persistent.eval()
        logger.info(f"shared adapter loaded from {lora}")

    n_done = n_skip = 0
    for user in tqdm(test_users, desc=f"eval {spec.record_name}"):
        uid = str(user["user_id"])

        if spec.kind == "per_user":
            lora = CKPTS_ROOT / spec.record_name / task / f"user_{uid}"
            if not (lora / "adapter_config.json").exists():
                logger.warning(f"LoRA not found for user {uid}: {lora}, skipped")
                n_skip += 1
                continue
            try:
                gen_model = PeftModel.from_pretrained(base_model, str(lora))
                gen_model.eval()
            except Exception as e:
                logger.error(f"failed to load LoRA for user {uid}: {e}")
                n_skip += 1
                continue
        elif spec.kind == "single":
            gen_model = persistent
        else:
            gen_model = base_model

        responses = _generate_for_user(gen_model, tokenizer, user, processor,
                                       all_profiles, label_map, args, use_sampling)
        save_json({
            "user_id": uid,
            "method": spec.record_name,
            "task": task,
            "eval_suffix": eval_suffix,
            "run_tag": run_tag,
            "k": k,
            "add_profile": add_profile,
            "responses": responses,
        }, responses_dir / f"user_{uid}.json")
        n_done += 1

        if spec.kind == "per_user":
            del gen_model
            gc.collect()
            torch.cuda.empty_cache()

    if persistent is not None:
        del persistent
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"responses written for {n_done} users (skipped {n_skip}) -> {responses_dir}")


def _generate_for_user(gen_model, tokenizer, user, processor, all_profiles,
                       label_map, args, use_sampling) -> List[Dict]:
    """Run all of one user's queries and return per-query response records."""
    processed = processor.process_user(user, all_profiles)
    prompts, qids, golds = [], [], []
    for q, proc in zip(user["query"], processed):
        prompts.append(proc["prompt"])
        qids.append(q["id"])
        golds.append(label_map.get(q["id"], q.get("gold")))

    outputs: List[str] = []
    for i in range(0, len(prompts), args.batch_size):
        outputs.extend(generate_batch(gen_model, tokenizer, prompts[i:i + args.batch_size],
                                      args.max_new_tokens, use_sampling))

    return [{"id": qid, "prompt": p, "output": o, "gold": g}
            for qid, p, o, g in zip(qids, prompts, outputs, golds)]


# ============================ phase 2: metrics ============================
def run_metrics(args, spec: MethodSpec, eval_suffix: str, run_tag: str,
                responses_dir: Path, metrics_dir: Path, logger) -> None:
    task = args.task_name
    resp_files = sorted(responses_dir.glob("user_*.json"))
    if not resp_files:
        raise SystemExit(f"no response files under {responses_dir}")
    logger.info(f"computing metrics for {len(resp_files)} users from {responses_dir}")

    bert = None
    if task_family(task) == "generative" and not args.no_bertscore:
        bert = BertScorer(args.deberta_path)
        if not bert.available:
            logger.warning("BERTScore unavailable (missing dep or model); ROUGE only")

    per_user: Dict[str, Dict[str, float]] = {}
    total_queries = 0
    for fpath in tqdm(resp_files, desc="metrics"):
        data = load_json(str(fpath))
        uid = str(data["user_id"])
        pairs = [(r["output"], r["gold"]) for r in data.get("responses", [])
                 if r.get("gold") is not None]
        preds = [p for p, _ in pairs]
        labels = [l for _, l in pairs]
        metrics = compute_user_metrics(task, preds, labels, bert_scorer=bert)
        per_user[uid] = metrics
        total_queries += len(pairs)
        save_json({
            "user_id": uid, "method": spec.record_name, "task": task,
            "eval_suffix": eval_suffix, "run_tag": run_tag,
            "num_queries": len(pairs), **metrics,
        }, metrics_dir / f"user_{uid}.json")

    avg = average_metrics(per_user)
    save_json({
        "method": spec.record_name, "task": task,
        "eval_suffix": eval_suffix, "run_tag": run_tag,
        "num_users": len(per_user), "num_queries_total": total_queries,
        "average": avg,
        "per_user": per_user,
    }, metrics_dir / "average.json")

    logger.info(f"average over {len(per_user)} users ({total_queries} queries):")
    for k, v in avg.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info(f"metrics written -> {metrics_dir}")


# ============================ entry ============================
def main() -> None:
    args = parse_args()
    spec = METHOD_SPECS[args.method]
    eval_suffix = resolve_eval_suffix(args)

    records_base = Path(args.records_dir) / spec.record_name / args.task_name
    run_tag = resolve_run_tag(args, eval_suffix, records_base)
    responses_dir = records_base / "responses" / eval_suffix / run_tag
    metrics_dir = records_base / "metrics" / eval_suffix / run_tag

    if not args.metric_only:
        responses_dir.mkdir(parents=True, exist_ok=True)
    elif not responses_dir.exists():
        raise SystemExit(f"--metric_only: responses dir missing: {responses_dir}")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(
        str(metrics_dir), f"{args.task_name}_{spec.record_name}_{eval_suffix}_{run_tag}"
    )
    logger.info("=" * 70)
    logger.info(f"run config | method={spec.record_name} task={args.task_name} "
                f"eval_suffix={eval_suffix} run_tag={run_tag} metric_only={args.metric_only}")
    logger.info(f"responses_dir={responses_dir}")
    logger.info(f"metrics_dir={metrics_dir}")
    logger.info("=" * 70)

    try:
        if not args.metric_only:
            run_generation(args, spec, eval_suffix, run_tag, responses_dir, logger)
        run_metrics(args, spec, eval_suffix, run_tag, responses_dir, metrics_dir, logger)
    except Exception:
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    logger.info("evaluation finished.")
    sys.exit(0)
