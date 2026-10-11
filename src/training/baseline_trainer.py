"""
PAPPO per-user preference-optimization driver.

This is the shared driver trimmed down to ONLY the PAPPO path. Everything the
per-user script needs lives here:

  * argument parser,
  * 4-bit loading of the task_LoRA merged model (policy init == π_ref base),
  * per-user data assembly from dpo_pair.json + records/precompute
    (PAPPO reads the record-level aggregated delta_ref_hat = sp_val_pappo_*),
  * per-user 90/10 split, per-epoch shuffling, gradient accumulation,
  * early stopping on validation loss + best-weight restore,
  * logging of hyper-parameters, per-step loss / gradient norm / lr,
  * OPPU-style checkpoint layout: ckpts/PAPPO/<task>/user_<uid>/{adapter, checkpoint-<step>},
  * --train_missing resume and --shard_id/--num_shards user-level multi-GPU split.

Trimmed: the DPO / HyPO (most-relevant-anchor π_ref), SimPO / IPO (reference-free)
and MRPO (harmonic-mean sp_val) code paths, the SimPO/HyPO γ target-margin and the
ablation re-aggregation hooks are all gone.
"""

import argparse
import gc
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from ..config import SUPPORTED_TASKS, get_task_config
from ..data.loader import load_json_file
from ..data.precompute_loader import (
    TargetUserPrompter,
    build_user_items,
    dpo_pair_path,
    load_entry_idx_index,
    load_spval_index,
    load_user_profiles,
    precompute_strategy_dir,
)
from ..models.lora_config import create_lora_config
from ..pipeline.common import (
    DEFAULT_CKPT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_PROMPT_DIR,
    merged_model_path,
    out_dir_for,
)
from ..utils.clean_output import load_prompt_templates
from ..utils.logging_utils import setup_logging
from .baseline_losses import BASELINE_LOSS_FN_MAP

DEFAULT_RECORDS_DIR = Path(DEFAULT_DATA_DIR).parent / "records"
DEFAULT_CKPTS_ROOT = Path(DEFAULT_DATA_DIR).parent / "ckpts"


# ============================ CLI ============================
def build_arg_parser(method):
    """Argument surface for the PAPPO per-user run."""
    p = argparse.ArgumentParser(
        description=f"{method.upper()} per-user preference optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # task / method
    p.add_argument('--task_name', type=str, required=True, choices=SUPPORTED_TASKS)
    p.add_argument('--k', type=int, default=1,
                   help='BM25 history count for the target-user prompt. '
                        'Must match the precompute run (k=1).')
    p.add_argument('--anchor_selection', type=str, default='similar',
                   choices=['similar', 'noisy', 'random'],
                   help='Which precompute strategy directory supplies the aggregated ref margin')

    # optimization
    p.add_argument('--dpo_epoch', type=int, default=6)
    p.add_argument('--dpo_lr', type=float, default=5e-5)
    p.add_argument('--batch_size', type=int, default=4,
                   help='Preference pairs per micro-batch (2 forwards each)')
    p.add_argument('--gradient_accumulation_steps', type=int, default=2)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--max_grad_norm', type=float, default=1.0,
                   help='Gradient clipping norm; <=0 means only measure, never clip')
    p.add_argument('--warmup_ratio', type=float, default=0.0,
                   help='Linear warmup share of the training steps')
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--cut_off', type=int, default=512,
                   help='Max sequence length. Keep 512 to match the precompute run.')

    # length normalization
    p.add_argument('--norm', dest='norm', action='store_true', default=True,
                   help='Use length-normalized (per-token) log-probs and sp_val_pappo_norm')
    p.add_argument('--no_norm', dest='norm', action='store_false',
                   help='Use raw sum log-probs and sp_val_pappo_unorm')

    # data split / early stopping
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--early_stopping_patience', type=int, default=1,
                   help='Epochs without val improvement before stopping; 0 disables')
    p.add_argument('--early_stopping_min_delta', type=float, default=1e-4)

    # LoRA
    p.add_argument('--lora_r', type=int, default=8)
    p.add_argument('--lora_alpha', type=int, default=8)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--grad_checkpointing', action='store_true', default=True)

    # run control
    p.add_argument('--max_users', type=int, default=None)
    p.add_argument('--max_pairs', type=int, default=None,
                   help='Cap preference pairs kept per user (smoke tests only)')
    p.add_argument('--train_missing', action='store_true', default=False,
                   help='Skip users that already have a checkpoint')
    p.add_argument('--require_ref', dest='require_ref', action='store_true', default=True,
                   help='Drop pairs whose precompute delta_ref_hat record is missing')
    p.add_argument('--no_require_ref', dest='require_ref', action='store_false',
                   help='Keep such pairs with reference margin=0 '
                        '(entry_idx/lengths recovered from queries.json)')
    p.add_argument('--logging_steps', type=int, default=10,
                   help='Log every N optimizer steps')
    p.add_argument('--log_first_pairs', type=int, default=0,
                   help='Dump per-pair details for the first N pairs of user #1')
    p.add_argument('--dry_run', action='store_true', default=False,
                   help='Build and split the data only; never load the model')
    p.add_argument('--shard_id', type=int, default=0)
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--gpu_id', type=int, default=0)

    # paths
    p.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument('--ckpt_dir', type=str, default=str(DEFAULT_CKPT_DIR),
                   help='Dir holding <task>-anchor_train/merged_model (policy init)')
    p.add_argument('--prompt_dir', type=str, default=str(DEFAULT_PROMPT_DIR))
    p.add_argument('--records_dir', type=str, default=str(DEFAULT_RECORDS_DIR))
    p.add_argument('--out_subdir', type=str, default='pappo_anchor')
    p.add_argument('--out_dir', type=str, default=None,
                   help=f'Default: {DEFAULT_CKPTS_ROOT}/<METHOD>')
    return p


def resolve_args(args, method):
    """Fill method-dependent defaults after parsing."""
    if args.out_dir is None:
        args.out_dir = str(DEFAULT_CKPTS_ROOT / method.upper())
    return args


# ============================ model loading ============================
def load_tokenizer(args):
    """Tokenizer only (CPU). Also used to recover token lengths without a model."""
    mm_path = merged_model_path(args.ckpt_dir, args.task_name)
    if not Path(mm_path, "config.json").exists():
        raise FileNotFoundError(
            f"task_LoRA merged model not found at {mm_path}. "
            f"It is produced by the anchor-training stage (Base + TaskLoRA -> merged_model)."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        mm_path, trust_remote_code=True, padding_side="left", use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_merged_base(args, logger):
    """4-bit load of the task_LoRA merged model (policy init and π_ref base)."""
    mm_path = merged_model_path(args.ckpt_dir, args.task_name)
    tokenizer = load_tokenizer(args)
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        mm_path, quantization_config=quant, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # incompatible with the backward passes here
    model.config.pad_token_id = tokenizer.pad_token_id
    # fp32 layernorms + gradient checkpointing wiring; get_peft_model then re-freezes
    # the base and leaves only the per-user LoRA trainable.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.grad_checkpointing,
    )
    model.eval()
    logger.info(f"Merged model loaded from {mm_path}")
    return tokenizer, model


# ============================ data ============================
def prepare_user_items(args, uid, dpo_triples, profiles, prompt_tmpl,
                       entry_idx_index, logger, tokenizer=None):
    """Precompute index + rebuilt prompts -> training items for a single user.

    PAPPO reads the record-level aggregated delta_ref_hat (sp_val_pappo_*) via
    load_spval_index; no anchor model is forwarded.
    """
    profile_list = profiles.get(str(uid), [])
    if not profile_list:
        logger.warning(f"[data] user {uid}: no profile entries in user_test_100.json")
    prompter = TargetUserPrompter(
        args.task_name, profile_list, prompt_tmpl[args.task_name],
        args.k, get_task_config(args.task_name)["x_field"],
    )
    index = load_spval_index(args.task_name, uid, args.records_dir,
                             strategy=args.anchor_selection, logger=logger)
    need_index = args.require_ref
    if not index:
        logger.error(f"[data] user {uid}: empty precompute sp_val index "
                     f"({precompute_strategy_dir(args.records_dir, args.task_name, args.anchor_selection)}); "
                     f"require_ref={args.require_ref}")
    return build_user_items(uid, dpo_triples, index, prompter,
                            require_ref=need_index,
                            entry_idx_index=entry_idx_index, tokenizer=tokenizer,
                            logger=logger)


def split_items(items, uid, seed, val_ratio):
    """Per-user shuffle then 90/10 split (validation subset stays fixed)."""
    rng = random.Random(seed + int(uid))
    order = list(range(len(items)))
    rng.shuffle(order)
    n_val = max(1, int(round(len(items) * val_ratio))) if len(items) > 1 else 0
    val = [items[i] for i in order[:n_val]]
    train = [items[i] for i in order[n_val:]]
    return train, val


# ============================ adapter lifecycle ============================
def attach_user_lora(base_model, args):
    """Mount a fresh trainable 'default' LoRA adapter on the shared base model."""
    if getattr(base_model, "peft_config", None) and "default" in base_model.peft_config:
        detach_user_lora(base_model)
    if args.grad_checkpointing:
        base_model.gradient_checkpointing_enable()
    cfg = create_lora_config(r=args.lora_r, lora_alpha=args.lora_alpha,
                             lora_dropout=args.lora_dropout)
    model = get_peft_model(base_model, cfg)
    model.train()
    return model


def detach_user_lora(base_model, model=None):
    """Remove the per-user adapter so the base model can serve the next user."""
    try:
        if model is not None:
            model.delete_adapter("default")
    except Exception:
        pass
    peft_cfg = getattr(base_model, "peft_config", None)
    if isinstance(peft_cfg, dict):
        peft_cfg.pop("default", None)


# ============================ training ============================
def _snapshot_lora(model):
    return {n: p.detach().cpu().clone()
            for n, p in model.named_parameters() if p.requires_grad}


def _restore_lora(model, snapshot):
    with torch.no_grad():
        n = 0
        for name, p in model.named_parameters():
            if name in snapshot:
                p.copy_(snapshot[name].to(p.device))
                n += 1
    return n


def _grad_norm(params, max_grad_norm):
    """Clip (optionally) and return the total gradient norm."""
    clip = max_grad_norm if (max_grad_norm and max_grad_norm > 0) else float("inf")
    return torch.nn.utils.clip_grad_norm_(params, clip).item()


def evaluate(model, tokenizer, loss_fn, items, args, batch_size):
    """Mean loss over the held-out pairs (no gradient, no optimizer state)."""
    model.eval()
    total_loss = 0.0
    total_pairs = 0
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            loss, stats = loss_fn(model, tokenizer, items[i:i + batch_size],
                                  beta=args.beta, cut_off=args.cut_off, norm=args.norm,
                                  no_grad=True)
            if loss is None:
                continue
            n = stats["valid_pairs"]
            total_loss += stats["loss"] * n
            total_pairs += n
    model.train()
    if total_pairs == 0:
        return None
    return total_loss / total_pairs


def train_user(model, tokenizer, loss_fn, uid, items, user_out_dir, args, logger,
               method, verbose_pairs=0):
    """Train one user's LoRA with early stopping, then save the best weights."""
    train_items, val_items = split_items(items, uid, args.seed, args.val_ratio)
    if not train_items:
        logger.warning(f"[{method}] user {uid}: empty train split, skipped")
        return None
    logger.info(f"[{method}] user {uid}: pairs={len(items)} "
                f"train={len(train_items)} val={len(val_items)} "
                f"epochs<={args.dpo_epoch} lr={args.dpo_lr} beta={args.beta} "
                f"batch={args.batch_size} grad_accum={args.gradient_accumulation_steps} "
                f"norm={args.norm} "
                f"val_ratio={args.val_ratio} patience={args.early_stopping_patience}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.dpo_lr, weight_decay=args.weight_decay)
    accum = max(1, args.gradient_accumulation_steps)
    steps_per_epoch = (len(train_items) + args.batch_size - 1) // args.batch_size // accum
    total_steps = max(1, steps_per_epoch * args.dpo_epoch)
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / max(1, warmup) if warmup and s < warmup
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup)),
    )

    best_val = float("inf")
    best_snapshot = None
    best_step = 0
    bad_epochs = 0
    global_step = 0
    stopped_at = None

    for epoch in range(args.dpo_epoch):
        rng = random.Random(args.seed * 1000 + int(uid) + epoch)
        order = list(range(len(train_items)))
        rng.shuffle(order)
        shuffled = [train_items[i] for i in order]

        accum_loss, accum_pairs, step_count = 0.0, 0, 0
        epoch_loss, epoch_pairs = 0.0, 0
        grad_norms = []
        verbose_left = verbose_pairs
        for i in range(0, len(shuffled), args.batch_size):
            batch = shuffled[i:i + args.batch_size]
            verbose = verbose_left > 0
            if verbose:
                verbose_left -= len(batch)
            loss, stats = loss_fn(model, tokenizer, batch, beta=args.beta,
                                  cut_off=args.cut_off, norm=args.norm,
                                  verbose=verbose, logger=logger)
            if loss is None:
                continue
            (loss / accum).backward()
            accum_loss += stats["loss"] * stats["valid_pairs"]
            accum_pairs += stats["valid_pairs"]
            epoch_loss += stats["loss"] * stats["valid_pairs"]
            epoch_pairs += stats["valid_pairs"]
            step_count += 1

            if step_count % accum == 0:
                gnorm = _grad_norm(params, args.max_grad_norm)
                grad_norms.append(gnorm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.logging_steps == 0:
                    logger.info(f"  step {global_step} | epoch {epoch + 1} | "
                                f"train_loss={accum_loss / max(1, accum_pairs):.4f} | "
                                f"grad_norm={gnorm:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")
                accum_loss, accum_pairs = 0.0, 0
            del loss
            gc.collect()
            torch.cuda.empty_cache()

        # flush remaining accumulated gradients
        if step_count % accum != 0:
            gnorm = _grad_norm(params, args.max_grad_norm)
            grad_norms.append(gnorm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        train_avg = epoch_loss / epoch_pairs if epoch_pairs else None
        val_loss = evaluate(model, tokenizer, loss_fn, val_items, args, args.batch_size) if val_items else None
        mean_gnorm = sum(grad_norms) / len(grad_norms) if grad_norms else float("nan")
        logger.info(f"[{method}] user {uid} epoch {epoch + 1}/{args.dpo_epoch}: "
                    f"train_loss={train_avg if train_avg is not None else float('nan'):.4f} "
                    f"val_loss={val_loss if val_loss is not None else float('nan'):.4f} "
                    f"grad_norm={mean_gnorm:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                    f"steps={global_step}")

        improved = val_loss is not None and val_loss < best_val - args.early_stopping_min_delta
        if improved:
            best_val = val_loss
            best_snapshot = _snapshot_lora(model)
            best_step = global_step
            bad_epochs = 0
        elif val_loss is not None:
            bad_epochs += 1
            logger.info(f"[{method}] user {uid}: no val improvement "
                        f"({bad_epochs}/{args.early_stopping_patience}), "
                        f"best_val={best_val:.4f} current_val={val_loss:.4f}")
        if args.early_stopping_patience and bad_epochs >= args.early_stopping_patience:
            stopped_at = epoch + 1
            logger.info(f"[{method}] user {uid}: early stopping at epoch {stopped_at}")
            break

    if best_snapshot is not None:
        restored = _restore_lora(model, best_snapshot)
        logger.info(f"[{method}] user {uid}: restored best weights "
                    f"(val_loss={best_val:.4f} @ step {best_step}, {restored} tensors)")
    else:
        logger.warning(f"[{method}] user {uid}: no valid improvement recorded "
                       f"(val pairs={len(val_items)}); saving final weights")

    Path(user_out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(user_out_dir)
    if best_step:
        model.save_pretrained(str(Path(user_out_dir) / f"checkpoint-{best_step}"))
    with open(Path(user_out_dir) / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "method": method, "task_name": args.task_name, "user_id": str(uid),
            "k": args.k, "beta": args.beta, "norm": args.norm,
            "lr": args.dpo_lr, "epochs_requested": args.dpo_epoch,
            "epochs_run": (stopped_at or args.dpo_epoch),
            "early_stopped_at": stopped_at,
            "pairs": len(items), "train_pairs": len(train_items), "val_pairs": len(val_items),
            "best_val_loss": None if best_val == float("inf") else best_val,
            "best_step": best_step, "total_steps": global_step,
            "anchor_selection": args.anchor_selection,
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"[{method}] user {uid}: saved to {user_out_dir}")
    return user_out_dir


# ============================ driver ============================
def run_baseline(args, method, logger):
    """Full per-user PAPPO driver."""
    loss_fn = BASELINE_LOSS_FN_MAP[method]
    strat_dir = precompute_strategy_dir(args.records_dir, args.task_name, args.anchor_selection)
    if not strat_dir.exists():
        raise FileNotFoundError(
            f"{method.upper()} needs precompute records for the aggregated ref margin but "
            f"{strat_dir} does not exist. Run the precompute stage first "
            f"(--task_name {args.task_name} --anchor_selection {args.anchor_selection}), "
            f"or pass --no_require_ref for the degenerate zero-ref mode."
        )

    dpo_pairs = load_json_file(str(dpo_pair_path(args.data_dir, args.task_name, args.out_subdir)))
    pa_dir = out_dir_for(args.data_dir, args.task_name, args.out_subdir)
    prompt_tmpl = load_prompt_templates(str(Path(args.prompt_dir) / "prompt.json"))
    profiles = load_user_profiles(args.data_dir, args.task_name)
    # Pairs that miss the precompute index have to fall back on queries.json for
    # entry_idx and on the tokenizer for the length-normalization denominators.
    need_recovery = not args.require_ref
    entry_idx_index = load_entry_idx_index(pa_dir) if need_recovery else {}
    prep_tokenizer = None
    if need_recovery:
        try:
            prep_tokenizer = load_tokenizer(args)
        except FileNotFoundError as e:
            logger.warning(f"[data] no tokenizer for the recovery fallback: {e}")
    logger.info(f"users_with_pairs={len(dpo_pairs)} profiles={len(profiles)} "
                f"queries_idx={len(entry_idx_index)}")

    user_ids = sorted(dpo_pairs.keys())
    if args.max_users is not None:
        user_ids = user_ids[:args.max_users]
    if args.num_shards > 1:
        user_ids = user_ids[args.shard_id::args.num_shards]
        logger.info(f"shard {args.shard_id}/{args.num_shards} -> {len(user_ids)} users")

    method_root = Path(args.out_dir) / args.task_name
    method_root.mkdir(parents=True, exist_ok=True)

    # ---- data-only pass (also used by --dry_run and --train_missing) ----
    model = None
    items_by_user = {}
    for uid in user_ids:
        out_path = method_root / f"user_{uid}"
        if args.train_missing and (out_path / "adapter_config.json").exists():
            logger.info(f"[{method}] user {uid}: checkpoint exists, skipped (--train_missing)")
            continue
        items = prepare_user_items(args, uid, dpo_pairs[uid], profiles,
                                   prompt_tmpl, entry_idx_index, logger,
                                   tokenizer=prep_tokenizer)
        if args.max_pairs:
            items = items[:args.max_pairs]
        if not items:
            logger.error(f"[{method}] user {uid}: no usable pairs, skipped")
            continue
        items_by_user[uid] = items
    logger.info(f"[{method}] prepared {len(items_by_user)} users "
                f"(total pairs={sum(len(v) for v in items_by_user.values())})")

    if args.dry_run:
        for uid, items in list(items_by_user.items())[:5]:
            tr, va = split_items(items, uid, args.seed, args.val_ratio)
            logger.info(f"[dry_run] user {uid}: pairs={len(items)} train={len(tr)} val={len(va)} "
                        f"example_prompt={tr[0]['prompt'][:80]!r}")
        logger.info("[dry_run] model not loaded, exiting")
        return 0, 0, len(user_ids)

    tokenizer, model = load_merged_base(args, logger)

    success = failed = 0
    from tqdm import tqdm
    for uid, items in tqdm(items_by_user.items(), desc=f"{method} users"):
        logger.info(f"=== user {uid} ({len(items)} pairs) ===")
        peft_model = None
        try:
            peft_model = attach_user_lora(model, args)
            result = train_user(peft_model, tokenizer, loss_fn, uid, items,
                                str(method_root / f"user_{uid}"), args, logger, method,
                                verbose_pairs=args.log_first_pairs if success == 0 else 0)
            success += 1 if result else 0
            failed += 0 if result else 1
        except Exception as e:
            failed += 1
            logger.error(f"[{method}] user {uid} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            if peft_model is not None:
                detach_user_lora(model, peft_model)
            del peft_model
            gc.collect()
            torch.cuda.empty_cache()

    skipped = len(user_ids) - len(items_by_user)
    logger.info(f"[{method}] complete: success={success} failed={failed} skipped={skipped}")
    return success, failed, skipped


# ============================ entry point ============================
def log_run_config(args, method, logger):
    """Dump everything this run depends on before the first GPU forward."""
    logger.info("=" * 70)
    logger.info(f"run config | method={method.upper()} task={args.task_name}")
    logger.info(json.dumps(vars(args), indent=2, sort_keys=True, default=str))
    logger.info(f"policy init (task_LoRA merged model): "
                f"{merged_model_path(args.ckpt_dir, args.task_name)}")
    logger.info(f"pi_ref: precompute {args.anchor_selection} records "
                f"({precompute_strategy_dir(args.records_dir, args.task_name, args.anchor_selection)}) "
                f"-> aggregated sp_val (delta_ref_hat) per triple")
    logger.info(f"ckpt/log root: {Path(args.out_dir) / args.task_name}")
    logger.info("=" * 70)


def main(method):
    """__main__ body for target_user/PAPPO.py."""
    args = resolve_args(build_arg_parser(method).parse_args(), method)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    log_dir = Path(args.out_dir) / args.task_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(str(log_dir), f"{args.task_name}_{method.upper()}_k{args.k}_{stamp}")
    log_run_config(args, method, logger)

    try:
        ok, fail, skipped = run_baseline(args, method, logger)
    except Exception:
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    logger.info(f"[{method.upper()}] finished: success={ok} failed={fail} skipped={skipped}")
    sys.exit(0 if fail == 0 else 1)
