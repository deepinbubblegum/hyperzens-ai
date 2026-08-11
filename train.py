"""Train a BPE-scale FFFTransformer (GPT-2 vocab, WikiText).

Loss
----
``L = CrossEntropy + λ(t) · L_balance``
with dynamic ``λ`` and FFF temperature annealing ``τ: 1.0 → 0.02``.

Optimisation
------------
* AdamW + **LR warmup + cosine decay** (``lr → min_lr``)
* Micro-batch ``batch_size=8`` × ``grad_accum_steps=8`` (effective 64) for ~12GB VRAM
* Mixed precision: CUDA autocast ``float16`` for forward + CE (logits stay fp16)
* ``torch.compile(..., mode="reduce-overhead")`` on CUDA with fallback
* DataLoader workers + ``pin_memory`` + ``persistent_workers`` on CUDA
* Caps: ``max_epochs=5``, ``max_steps=3000``
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Reduce CUDA allocator fragmentation before importing torch.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset_loader import (
    GPT2_VOCAB_SIZE,
    BPEDataset,
    build_wikitext_datasets,
    get_gpt2_encoding,
)
from device_utils import (
    amp_autocast,
    apply_hardware_optimizations,
    make_grad_scaler,
    pin_memory_for,
    print_device_info,
    resolve_device,
    to_device,
)
from models.transformer import FFFConfig, FFFTransformer

DATA_DIR = Path(__file__).resolve().parent / "data"
CHECKPOINT_NAME = "fff_checkpoint.pt"


@dataclass
class TrainConfig:
    """CLI / runtime training hyperparameters (BPE SLM defaults)."""

    # Model — aligned with scaled :class:`FFFConfig`
    n_embd: int = 512
    n_layer: int = 8
    n_head: int = 8
    fff_depth: int = 6
    block_size: int = 256
    dropout: float = 0.1
    vocab_size: int = GPT2_VOCAB_SIZE

    # Data
    wiki_variant: str = "wikitext-2"  # or wikitext-103

    # Optimisation (RTX 3060 12GB — small micro-batch, same effective batch 64)
    batch_size: int = 8
    grad_accum_steps: int = 8  # effective batch = 8 * 8 = 64
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    max_epochs: int = 5
    grad_clip: float = 1.0
    compile_model: bool = True  # torch.compile on CUDA only
    compile_mode: str = "reduce-overhead"  # lower warm-up VRAM vs "default"
    lr_warmup_frac: float = 0.05  # fraction of optimizer steps for LR warmup

    # FFF routing schedule
    init_temp: float = 1.0
    min_temp: float = 0.02
    temp_warmup_frac: float = 0.15  # keep τ=1.0 for first 15% of steps
    lambda_balance: float = 0.1
    lambda_balance_end: float = 0.01

    # Logging / eval / IO
    eval_interval: int = 200  # optimizer steps between val runs
    eval_batches: int = 20
    seed: int = 42
    device: str = "auto"
    num_workers: int = 4
    checkpoint_path: str = CHECKPOINT_NAME
    wandb: bool = False
    wandb_project: str = "fff-transformer"
    data_dir: str = str(DATA_DIR)
    steps_per_epoch: int | None = None
    max_steps: int | None = 3000  # hard cap so training finishes in a reasonable time


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def annealed_temperature(
    step: int,
    total_steps: int,
    init_temp: float,
    min_temp: float,
    warmup_frac: float = 0.15,
) -> float:
    """Keep ``τ = init_temp`` during warmup, then exponential decay to ``min_temp``."""
    if total_steps <= 1:
        return float(min_temp)
    warmup_steps = int(total_steps * warmup_frac)
    if step < warmup_steps:
        return float(init_temp)
    decay_steps = max(total_steps - warmup_steps, 1)
    t = min(max(step - warmup_steps, 0), decay_steps - 1)
    ratio = t / float(max(decay_steps - 1, 1))
    return float(init_temp * ((min_temp / init_temp) ** ratio))


def dynamic_lambda_balance(
    step: int,
    total_steps: int,
    lambda_start: float,
    lambda_end: float,
) -> float:
    """Linearly decay balance weight ``λ_start → λ_end``."""
    if total_steps <= 1:
        return float(lambda_end)
    ratio = min(max(step, 0), total_steps - 1) / float(total_steps - 1)
    return float(lambda_start + (lambda_end - lambda_start) * ratio)


def cosine_lr(
    step: int,
    total_steps: int,
    lr: float,
    min_lr: float,
    warmup_frac: float,
) -> float:
    """Linear warmup then cosine decay to ``min_lr``.

    ``step`` is the optimizer-step index (after accumulation), 0-based.
    """
    if total_steps <= 0:
        return min_lr
    warmup_steps = max(int(total_steps * warmup_frac), 1)
    if step < warmup_steps:
        return float(lr * float(step + 1) / float(warmup_steps))
    if step >= total_steps - 1:
        return float(min_lr)
    progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps - 1, 1))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + (lr - min_lr) * cosine)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


# ---------------------------------------------------------------------------
# Train helpers
# ---------------------------------------------------------------------------


def maybe_compile(
    model: FFFTransformer,
    enabled: bool,
    device: torch.device,
    mode: str = "reduce-overhead",
) -> FFFTransformer:
    """Compile with ``torch.compile`` on CUDA when requested.

    Tries ``mode`` first (default ``reduce-overhead`` to limit warm-up VRAM),
    then falls back to ``"default"``, then returns the eager model.
    """
    if not enabled:
        return model
    if device.type != "cuda":
        print(f"torch.compile skipped (device={device.type}; CUDA only)")
        return model
    if not hasattr(torch, "compile"):
        print("torch.compile unavailable — skipping")
        return model

    modes = [mode]
    if mode != "default":
        modes.append("default")

    for compile_mode in modes:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            compiled = torch.compile(model, mode=compile_mode)  # type: ignore[assignment]
            print(f"torch.compile enabled (mode={compile_mode!r}, device={device.type})")
            return compiled  # type: ignore[return-value]
        except Exception as exc:  # pragma: no cover
            print(f"torch.compile mode={compile_mode!r} failed ({exc}); trying fallback...")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("torch.compile skipped — using eager model")
    return model


def make_dataloader(
    dataset: BPEDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
    device: torch.device,
) -> DataLoader:
    """Build a DataLoader with CUDA-friendly worker / pin_memory settings."""
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
    }
    # persistent_workers requires num_workers > 0; enable on CUDA for throughput.
    if num_workers > 0 and device.type == "cuda":
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def compute_batch_loss(
    model: FFFTransformer,
    input_ids: Tensor,
    targets: Tensor,
    lambda_balance: float,
    mode: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """``input_ids, targets: (B, T)`` → ``(total, ce, balance)``.

    Caller must wrap this in :func:`amp_autocast` so the forward pass and
    ``F.cross_entropy`` run under CUDA ``float16`` autocast (logits stay fp16).
    """
    logits, balance_loss = model(input_ids, mode=mode)
    if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
        raise RuntimeError(
            f"logits shape {tuple(logits.shape)} incompatible with "
            f"input_ids {tuple(input_ids.shape)}"
        )
    if logits.size(-1) != unwrap_model(model).config.vocab_size:  # type: ignore[union-attr]
        raise RuntimeError(
            f"LM head width {logits.size(-1)} != vocab_size "
            f"{unwrap_model(model).config.vocab_size}"  # type: ignore[union-attr]
        )
    # Keep CE inside the outer autocast context (do not cast logits to fp32 here).
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    if mode == "soft":
        total = ce + lambda_balance * balance_loss
    else:
        total = ce
        balance_loss = balance_loss.detach()
    return total, ce, balance_loss


@torch.no_grad()
def evaluate(
    model: FFFTransformer,
    loader: DataLoader,
    device: torch.device,
    lambda_balance: float,
    max_batches: int,
    mode: str,
) -> dict[str, float]:
    model.eval()
    total_ce = 0.0
    total_bal = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = to_device(x, device)
        y = to_device(y, device)
        with amp_autocast(device):
            logits, balance_loss = model(x, mode=mode)
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        preds = logits.argmax(dim=-1)
        total_correct += int((preds == y).sum().item())
        total_tokens += y.numel()
        total_ce += float(ce.item())
        total_bal += float(balance_loss.item())
        n_batches += 1

    if n_batches == 0:
        return {"ce": float("nan"), "ppl": float("nan"), "acc": float("nan"), "balance": 0.0}

    mean_ce = total_ce / n_batches
    mean_bal = total_bal / n_batches
    return {
        "ce": mean_ce,
        "ppl": math.exp(min(mean_ce, 100.0)),
        "acc": total_correct / max(total_tokens, 1),
        "balance": mean_bal,
        "loss": mean_ce + (lambda_balance * mean_bal if mode == "soft" else 0.0),
    }


def format_diagnostic_report(
    epoch: int,
    step: int,
    tau: float,
    lam: float,
    lr: float,
    mean_loss: float,
    mean_ce: float,
    diag: dict[str, float],
) -> str:
    util = diag["leaf_utilization_pct"]
    util_min = diag["leaf_utilization_pct_min"]
    split = diag["router_split_ratio_mean"]
    collapse_flag = "COLLAPSE?" if util_min < 80.0 else "ok"
    split_flag = "ok" if abs(split - 0.5) < 0.15 else "SKEWED?"
    return "\n".join(
        [
            "",
            "=" * 72,
            f"DIAGNOSTIC REPORT — epoch {epoch} (opt_step {step})",
            "=" * 72,
            f"  train loss (avg):     {mean_loss:.4f}  |  CE={mean_ce:.4f}",
            f"  lr / τ / λ:           {lr:.2e} / {tau:.4f} / {lam:.4f}",
            f"  leaf utilization:     {util:.1f}% avg | {util_min:.1f}% worst  [{collapse_flag}]",
            f"  router split ratio:   mean p={split:.4f}  [{split_flag}]",
            f"  grad ‖router‖₂:       {diag['router_grad_norm']:.4e}",
            f"  grad ‖leaf‖₂:         {diag['leaf_grad_norm']:.4e}",
            f"  leaf entropy (norm):  {diag['leaf_entropy_norm']:.4f}",
            "=" * 72,
        ]
    )


def save_checkpoint(
    path: Path,
    model: FFFTransformer,
    optimizer: torch.optim.Optimizer,
    train_cfg: TrainConfig,
    step: int,
    extras: dict[str, Any] | None = None,
) -> None:
    core = unwrap_model(model)
    payload: dict[str, Any] = {
        "model_state_dict": core.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(core.config),  # type: ignore[union-attr]
        "train_config": asdict(train_cfg),
        "tokenizer": {
            "encoding": "gpt2",
            "vocab_size": GPT2_VOCAB_SIZE,
        },
        "step": step,
    }
    if extras:
        payload.update(extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"Saved checkpoint → {path}")


def assert_embed_lm_alignment(model: FFFTransformer) -> None:
    """Verify embedding / LM-head vocab dims match GPT-2 BPE size."""
    cfg = model.config
    assert model.wte.num_embeddings == cfg.vocab_size, (
        f"wte vocab {model.wte.num_embeddings} != config {cfg.vocab_size}"
    )
    assert model.lm_head.out_features == cfg.vocab_size, (
        f"lm_head out {model.lm_head.out_features} != config {cfg.vocab_size}"
    )
    assert model.wte.embedding_dim == cfg.n_embd == model.lm_head.in_features, (
        f"d_model mismatch: emb={model.wte.embedding_dim}, "
        f"cfg={cfg.n_embd}, lm_in={model.lm_head.in_features}"
    )
    if cfg.tie_weights:
        assert model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr(), (
            "tie_weights=True but lm_head.weight is not shared with wte"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    d = TrainConfig()
    p = argparse.ArgumentParser(description="Train BPE-scale FFFTransformer on WikiText")
    p.add_argument("--n-embd", type=int, default=d.n_embd)
    p.add_argument("--n-layer", type=int, default=d.n_layer)
    p.add_argument("--n-head", type=int, default=d.n_head)
    p.add_argument("--fff-depth", type=int, default=d.fff_depth)
    p.add_argument("--block-size", type=int, default=d.block_size)
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--wiki-variant", choices=["wikitext-2", "wikitext-103"], default=d.wiki_variant)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--grad-accum-steps", type=int, default=d.grad_accum_steps)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--min-lr", type=float, default=d.min_lr)
    p.add_argument("--lr-warmup-frac", type=float, default=d.lr_warmup_frac)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--max-epochs", type=int, default=d.max_epochs)
    p.add_argument("--grad-clip", type=float, default=d.grad_clip)
    p.add_argument("--init-temp", type=float, default=d.init_temp)
    p.add_argument("--min-temp", type=float, default=d.min_temp)
    p.add_argument("--temp-warmup-frac", type=float, default=d.temp_warmup_frac)
    p.add_argument("--lambda-balance", type=float, default=d.lambda_balance)
    p.add_argument("--lambda-balance-end", type=float, default=d.lambda_balance_end)
    p.add_argument("--eval-interval", type=int, default=d.eval_interval)
    p.add_argument("--eval-batches", type=int, default=d.eval_batches)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", type=str, default=d.device)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=d.compile_model)
    p.add_argument(
        "--compile-mode",
        type=str,
        default=d.compile_mode,
        choices=["reduce-overhead", "default", "max-autotune"],
        help="torch.compile mode (CUDA); reduce-overhead uses less warm-up VRAM",
    )
    p.add_argument("--checkpoint-path", type=str, default=d.checkpoint_path)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default=d.wandb_project)
    p.add_argument("--data-dir", type=str, default=d.data_dir)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument(
        "--max-steps",
        type=int,
        default=d.max_steps,
        help="Cap on optimizer steps (default 3000; use 0 for no cap)",
    )
    return p


def train_config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        fff_depth=args.fff_depth,
        block_size=args.block_size,
        dropout=args.dropout,
        wiki_variant=args.wiki_variant,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_frac=args.lr_warmup_frac,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        grad_clip=args.grad_clip,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        init_temp=args.init_temp,
        min_temp=args.min_temp,
        temp_warmup_frac=args.temp_warmup_frac,
        lambda_balance=args.lambda_balance,
        lambda_balance_end=args.lambda_balance_end,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        checkpoint_path=args.checkpoint_path,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        data_dir=args.data_dir,
        steps_per_epoch=args.steps_per_epoch,
        max_steps=None if args.max_steps == 0 else args.max_steps,
    )


def main() -> None:
    args = build_argparser().parse_args()
    cfg = train_config_from_args(args)

    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    apply_hardware_optimizations(device)
    print_device_info(device)

    # --- BPE WikiText ---
    print(f"Preparing {cfg.wiki_variant} (GPT-2 BPE, block_size={cfg.block_size})...")
    train_ds, val_ds, _test_ds, meta = build_wikitext_datasets(
        variant=cfg.wiki_variant,  # type: ignore[arg-type]
        block_size=cfg.block_size,
        cache_dir=Path(cfg.data_dir) / "cache",
    )
    assert int(meta["vocab_size"]) == cfg.vocab_size == GPT2_VOCAB_SIZE
    _ = get_gpt2_encoding()  # ensure tiktoken is importable early

    pin_memory = bool(pin_memory_for(device) or device.type == "cuda")
    train_loader = make_dataloader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        device=device,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        device=device,
    )

    model_cfg = FFFConfig(
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
        fff_depth=cfg.fff_depth,
        init_temp=cfg.init_temp,
        tie_weights=True,
    )
    model = FFFTransformer(model_cfg).to(device)
    assert_embed_lm_alignment(model)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = maybe_compile(
        model,
        enabled=cfg.compile_model,
        device=device,
        mode=cfg.compile_mode,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = make_grad_scaler(device)

    micro_per_epoch = cfg.steps_per_epoch or len(train_loader)
    micro_per_epoch = min(micro_per_epoch, len(train_loader))
    # Optimizer steps ≈ ceil(micro / accum) per epoch
    opt_per_epoch = max((micro_per_epoch + cfg.grad_accum_steps - 1) // cfg.grad_accum_steps, 1)
    total_opt_steps = opt_per_epoch * cfg.max_epochs
    if cfg.max_steps is not None:
        total_opt_steps = min(total_opt_steps, cfg.max_steps)

    raw = unwrap_model(model)
    n_params = sum(p.numel() for p in raw.parameters())
    print(
        f"Device={device} | vocab={cfg.vocab_size} | params={n_params:,} "
        f"(~{n_params / 1e6:.1f}M) | "
        f"D={cfg.n_embd} L={cfg.n_layer} H={cfg.n_head} fff_depth={cfg.fff_depth} | "
        f"batch={cfg.batch_size}×accum={cfg.grad_accum_steps} "
        f"(eff={cfg.batch_size * cfg.grad_accum_steps}) | "
        f"train_chunks={len(train_ds):,} (stride={train_ds.stride}) | "
        f"opt_steps={total_opt_steps} | "
        f"τ={cfg.init_temp}→{cfg.min_temp} | lr={cfg.lr}→{cfg.min_lr}"
    )

    wandb_run = None
    if cfg.wandb:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb_project,
            config={**asdict(cfg), **asdict(model_cfg)},
        )

    global_opt_step = 0
    best_hard_ppl = float("inf")
    epoch_reports: list[dict[str, Any]] = []
    t0 = time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.max_epochs}", leave=True)
        epoch_loss_sum = 0.0
        epoch_ce_sum = 0.0
        epoch_micro = 0
        accum_count = 0
        last_diag: dict[str, float] | None = None
        last_tau = cfg.init_temp
        last_lam = cfg.lambda_balance
        last_lr = cfg.lr

        optimizer.zero_grad(set_to_none=True)

        for input_ids, targets in pbar:
            if cfg.max_steps is not None and global_opt_step >= cfg.max_steps:
                break
            if epoch_micro >= micro_per_epoch:
                break

            input_ids = to_device(input_ids, device, non_blocking=pin_memory)
            targets = to_device(targets, device, non_blocking=pin_memory)

            # Schedules keyed by upcoming optimizer step index
            tau = annealed_temperature(
                global_opt_step,
                total_opt_steps,
                cfg.init_temp,
                cfg.min_temp,
                warmup_frac=cfg.temp_warmup_frac,
            )
            unwrap_model(model).set_temperature(tau)  # type: ignore[union-attr]
            lam = dynamic_lambda_balance(
                global_opt_step,
                total_opt_steps,
                cfg.lambda_balance,
                cfg.lambda_balance_end,
            )
            lr = cosine_lr(
                global_opt_step,
                total_opt_steps,
                cfg.lr,
                cfg.min_lr,
                cfg.lr_warmup_frac,
            )
            set_lr(optimizer, lr)
            last_tau, last_lam, last_lr = tau, lam, lr

            with amp_autocast(device):
                total_loss, ce_loss, bal_loss = compute_batch_loss(
                    model,
                    input_ids,
                    targets,
                    lambda_balance=lam,
                    mode="soft",
                )
                loss = total_loss / float(cfg.grad_accum_steps)

            scaler.scale(loss).backward()
            accum_count += 1
            epoch_micro += 1
            epoch_loss_sum += float(total_loss.detach())
            epoch_ce_sum += float(ce_loss.detach())

            do_step = accum_count >= cfg.grad_accum_steps or epoch_micro >= micro_per_epoch
            if do_step:
                last_diag = unwrap_model(model).get_routing_diagnostics()  # type: ignore[union-attr]
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                global_opt_step += 1

                pbar.set_postfix(
                    loss=f"{float(total_loss.detach()):.3f}",
                    ce=f"{float(ce_loss.detach()):.3f}",
                    lr=f"{lr:.1e}",
                    tau=f"{tau:.3f}",
                    util=f"{last_diag['leaf_utilization_pct']:.0f}%",
                )

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": float(total_loss.detach()),
                            "train/ce": float(ce_loss.detach()),
                            "train/balance": float(bal_loss.detach()),
                            "train/lr": lr,
                            "train/temperature": tau,
                            "train/lambda_balance": lam,
                            "train/leaf_utilization_pct": last_diag["leaf_utilization_pct"],
                            "epoch": epoch,
                        },
                        step=global_opt_step,
                    )

                if global_opt_step % cfg.eval_interval == 0 or global_opt_step == total_opt_steps:
                    soft_m = evaluate(
                        model, val_loader, device, lam, cfg.eval_batches, mode="soft"
                    )
                    hard_m = evaluate(
                        model, val_loader, device, lam, cfg.eval_batches, mode="hard"
                    )
                    print(
                        f"\n[opt_step {global_opt_step}] lr={lr:.2e} τ={tau:.4f} λ={lam:.4f} | "
                        f"soft ppl={soft_m['ppl']:.2f} | hard ppl={hard_m['ppl']:.2f} | "
                        f"util={last_diag['leaf_utilization_pct']:.1f}%"
                    )
                    if hard_m["ppl"] < best_hard_ppl:
                        best_hard_ppl = hard_m["ppl"]
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "val/soft_ppl": soft_m["ppl"],
                                "val/hard_ppl": hard_m["ppl"],
                                "val/soft_acc": soft_m["acc"],
                                "val/hard_acc": hard_m["acc"],
                            },
                            step=global_opt_step,
                        )
                    model.train()

                if cfg.max_steps is not None and global_opt_step >= cfg.max_steps:
                    break

        if epoch_micro > 0 and last_diag is not None:
            mean_loss = epoch_loss_sum / epoch_micro
            mean_ce = epoch_ce_sum / epoch_micro
            report = {
                "epoch": epoch + 1,
                "step": global_opt_step,
                "mean_loss": mean_loss,
                "mean_ce": mean_ce,
                "tau": last_tau,
                "lambda_balance": last_lam,
                "lr": last_lr,
                **last_diag,
            }
            epoch_reports.append(report)
            print(
                format_diagnostic_report(
                    epoch + 1,
                    global_opt_step,
                    last_tau,
                    last_lam,
                    last_lr,
                    mean_loss,
                    mean_ce,
                    last_diag,
                )
            )

        if cfg.max_steps is not None and global_opt_step >= cfg.max_steps:
            break

    elapsed = time.time() - t0
    soft_m = evaluate(
        model, val_loader, device, cfg.lambda_balance_end, cfg.eval_batches, mode="soft"
    )
    hard_m = evaluate(
        model, val_loader, device, cfg.lambda_balance_end, cfg.eval_batches, mode="hard"
    )
    print(
        f"\nDone in {elapsed:.1f}s | soft ppl={soft_m['ppl']:.2f} | hard ppl={hard_m['ppl']:.2f}"
    )

    save_checkpoint(
        Path(cfg.checkpoint_path),
        model,
        optimizer,
        cfg,
        step=global_opt_step,
        extras={
            "final_val_soft": soft_m,
            "final_val_hard": hard_m,
            "best_hard_ppl": best_hard_ppl,
            "epoch_diagnostic_reports": epoch_reports,
        },
    )

    if wandb_run is not None:
        wandb_run.summary["best_hard_ppl"] = best_hard_ppl
        wandb_run.finish()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
