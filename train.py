"""Train an FFFTransformer on TinyShakespeare (toy LM demo).

Loss
----
``L = CrossEntropy(logits, targets) + λ(t) · L_balance``
where ``L_balance`` combines router MSE-to-0.5, entropy maximization, and
leaf uniformity. ``λ`` starts higher (default ``0.1``) and decays to
``lambda_balance_end`` (default ``0.01``).

Temperature schedule
--------------------
1. **Warmup** (first ``warmup_frac`` of steps, default 15%): keep ``τ = 1.0``.
2. **Decay**: exponential ``τ: 1.0 → min_temp`` only after warmup so routers
   learn meaningful splits before hardening.

Diagnostics
-----------
Each epoch logs leaf utilization, router split ratio, and router/leaf grad
norms via ``model.get_routing_diagnostics()`` (collapse / stagnation detectors).
"""

from __future__ import annotations

import argparse
import math
import os
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.transformer import FFFConfig, FFFTransformer

# ---------------------------------------------------------------------------
# Data: TinyShakespeare (char-level)
# ---------------------------------------------------------------------------

TINYSPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DATA_DIR = Path(__file__).resolve().parent / "data"
CHECKPOINT_NAME = "fff_checkpoint.pt"


@dataclass
class TrainConfig:
    """CLI / runtime training hyperparameters."""

    # Model (reasonable defaults for a toy demo)
    n_embd: int = 256
    n_layer: int = 4
    n_head: int = 4
    fff_depth: int = 4
    block_size: int = 128
    dropout: float = 0.1

    # Optimisation
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 5
    grad_clip: float = 1.0

    # FFF extras
    init_temp: float = 1.0
    min_temp: float = 0.05
    warmup_frac: float = 0.15  # keep τ=init_temp for first 15% of steps
    lambda_balance: float = 0.1  # start high to fight collapse
    lambda_balance_end: float = 0.01  # anneal λ toward this

    # Logging / eval / IO
    eval_interval: int = 200  # optimizer steps between val runs
    eval_batches: int = 20
    seed: int = 42
    device: str = "cpu"
    num_workers: int = 0
    checkpoint_path: str = CHECKPOINT_NAME
    wandb: bool = False
    wandb_project: str = "fff-transformer"
    data_dir: str = str(DATA_DIR)
    steps_per_epoch: int | None = None  # optional cap (dry-runs / diagnostics)


class CharTokenizer:
    """Simple character-level vocabulary over the corpus alphabet."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.chars: list[str] = chars
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size: int = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int] | Tensor) -> str:
        if isinstance(ids, Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)


class CharLMDataset(Dataset):
    """Sliding windows of ``block_size`` tokens → next-token LM pairs.

    Each item:
        ``x``: ``(T,)`` input ids
        ``y``: ``(T,)`` targets (x shifted by 1 in the underlying stream)
    """

    def __init__(self, token_ids: Tensor, block_size: int) -> None:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a 1-D LongTensor")
        if len(token_ids) <= block_size:
            raise ValueError(
                f"corpus length {len(token_ids)} must exceed block_size {block_size}"
            )
        self.data = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        # Number of starting positions such that x[i:i+T], y[i+1:i+1+T] fit.
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1].clone()
        y = chunk[1:].clone()
        return x, y


def download_tinyshakespeare(data_dir: Path) -> Path:
    """Download TinyShakespeare into ``data_dir`` if missing; return path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "tinyshakespeare.txt"
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"Downloading TinyShakespeare → {path}")
    urllib.request.urlretrieve(TINYSPEARE_URL, path)
    return path


def load_datasets(
    data_dir: Path,
    block_size: int,
    split_ratio: float = 0.9,
) -> tuple[CharTokenizer, CharLMDataset, CharLMDataset]:
    """Build char tokenizer + train/val datasets (90/10 chronological split)."""
    path = download_tinyshakespeare(data_dir)
    text = path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = len(ids)
    n_train = int(n * split_ratio)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]
    train_ds = CharLMDataset(train_ids, block_size)
    val_ds = CharLMDataset(val_ids, block_size)
    return tokenizer, train_ds, val_ds


# ---------------------------------------------------------------------------
# Schedules: temperature (warmup + decay) and λ_balance
# ---------------------------------------------------------------------------


def annealed_temperature(
    step: int,
    total_steps: int,
    init_temp: float,
    min_temp: float,
    warmup_frac: float = 0.15,
) -> float:
    """Warmup at ``init_temp``, then exponential decay to ``min_temp``.

    Phase 1 — warmup (``t < warmup_frac · T``): ``τ = init_temp`` (fully soft).
    Phase 2 — decay: ``τ(u) = τ₀ · (τ_min/τ₀)^u`` for ``u ∈ [0, 1]`` over the
    remaining steps.
    """
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
    """Linearly decay balance weight ``λ_start → λ_end`` over training."""
    if total_steps <= 1:
        return float(lambda_end)
    ratio = min(max(step, 0), total_steps - 1) / float(total_steps - 1)
    return float(lambda_start + (lambda_end - lambda_start) * ratio)


# ---------------------------------------------------------------------------
# Train / eval / diagnostics
# ---------------------------------------------------------------------------


def compute_batch_loss(
    model: FFFTransformer,
    input_ids: Tensor,
    targets: Tensor,
    lambda_balance: float,
    mode: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(total_loss, ce_loss, balance_loss)`` for one batch.

    Shapes: ``input_ids, targets`` are ``(B, T)``.
    """
    logits, balance_loss = model(input_ids, mode=mode)  # logits: (B, T, V)
    ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )
    if mode == "soft":
        total = ce + lambda_balance * balance_loss
    else:
        # Hard routing has no differentiable balance term.
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
    """Evaluate CE / perplexity / next-token accuracy under soft or hard routing."""
    model.eval()
    total_ce = 0.0
    total_bal = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
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
    acc = total_correct / max(total_tokens, 1)
    ppl = math.exp(min(mean_ce, 100.0))  # clamp for numerical safety
    return {
        "ce": mean_ce,
        "ppl": ppl,
        "acc": acc,
        "balance": mean_bal,
        "loss": mean_ce + (lambda_balance * mean_bal if mode == "soft" else 0.0),
    }


def format_diagnostic_report(
    epoch: int,
    step: int,
    tau: float,
    lam: float,
    mean_loss: float,
    mean_ce: float,
    diag: dict[str, float],
) -> str:
    """Human-readable per-epoch routing health report."""
    util = diag["leaf_utilization_pct"]
    util_min = diag["leaf_utilization_pct_min"]
    split = diag["router_split_ratio_mean"]
    split_std = diag["router_split_ratio_std"]
    collapse_flag = "COLLAPSE?" if util_min < 80.0 else "ok"
    split_flag = "ok" if abs(split - 0.5) < 0.15 else "SKEWED?"
    lines = [
        "",
        "=" * 72,
        f"DIAGNOSTIC REPORT — epoch {epoch} (step {step})",
        "=" * 72,
        f"  train loss (avg):     {mean_loss:.4f}  |  CE={mean_ce:.4f}",
        f"  temperature τ:        {tau:.4f}",
        f"  λ_balance:            {lam:.4f}",
        f"  leaf utilization:     {util:.1f}% avg | {util_min:.1f}% worst-layer  [{collapse_flag}]",
        f"    (target > 80% active leaves)",
        f"  router split ratio:   mean p={split:.4f}  std={split_std:.4f}  [{split_flag}]",
        f"    (target ≈ 0.5 early on)",
        f"  grad ‖router‖₂:       {diag['router_grad_norm']:.4e}",
        f"  grad ‖leaf‖₂:         {diag['leaf_grad_norm']:.4e}",
        f"  leaf entropy (norm):  {diag['leaf_entropy_norm']:.4f}  (1.0 = uniform)",
        f"  balance loss:         {diag['balance_loss']:.4f}",
        "=" * 72,
    ]
    return "\n".join(lines)


def save_checkpoint(
    path: Path,
    model: FFFTransformer,
    optimizer: torch.optim.Optimizer,
    tokenizer: CharTokenizer,
    train_cfg: TrainConfig,
    step: int,
    extras: dict[str, Any] | None = None,
) -> None:
    """Persist model + optimizer + tokenizer metadata for ``infer.py``."""
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model.config),
        "train_config": asdict(train_cfg),
        "tokenizer": {
            "chars": tokenizer.chars,
            "vocab_size": tokenizer.vocab_size,
        },
        "step": step,
    }
    if extras:
        payload.update(extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"Saved checkpoint → {path}")


def build_argparser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description="Train FFFTransformer on TinyShakespeare")
    p.add_argument("--n-embd", type=int, default=defaults.n_embd)
    p.add_argument("--n-layer", type=int, default=defaults.n_layer)
    p.add_argument("--n-head", type=int, default=defaults.n_head)
    p.add_argument("--fff-depth", type=int, default=defaults.fff_depth)
    p.add_argument("--block-size", type=int, default=defaults.block_size)
    p.add_argument("--dropout", type=float, default=defaults.dropout)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    p.add_argument("--max-epochs", type=int, default=defaults.max_epochs)
    p.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    p.add_argument("--init-temp", type=float, default=defaults.init_temp)
    p.add_argument("--min-temp", type=float, default=defaults.min_temp)
    p.add_argument("--warmup-frac", type=float, default=defaults.warmup_frac)
    p.add_argument(
        "--lambda-balance",
        type=float,
        default=defaults.lambda_balance,
        help="Initial λ for balance loss (decays toward --lambda-balance-end)",
    )
    p.add_argument(
        "--lambda-balance-end",
        type=float,
        default=defaults.lambda_balance_end,
    )
    p.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
    p.add_argument("--eval-batches", type=int, default=defaults.eval_batches)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--num-workers", type=int, default=defaults.num_workers)
    p.add_argument("--checkpoint-path", type=str, default=defaults.checkpoint_path)
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    p.add_argument("--wandb-project", type=str, default=defaults.wandb_project)
    p.add_argument("--data-dir", type=str, default=defaults.data_dir)
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional global cap on optimizer steps (smoke tests)",
    )
    p.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Optional per-epoch step cap (diagnostic dry-runs)",
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
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        grad_clip=args.grad_clip,
        init_temp=args.init_temp,
        min_temp=args.min_temp,
        warmup_frac=args.warmup_frac,
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
    )


def main() -> None:
    args = build_argparser().parse_args()
    cfg = train_config_from_args(args)
    max_steps_cap: int | None = args.max_steps

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    data_dir = Path(cfg.data_dir)
    tokenizer, train_ds, val_ds = load_datasets(data_dir, cfg.block_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    model_cfg = FFFConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
        fff_depth=cfg.fff_depth,
        init_temp=cfg.init_temp,
    )
    model = FFFTransformer(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    full_steps_per_epoch = len(train_loader)
    epoch_step_cap = cfg.steps_per_epoch or full_steps_per_epoch
    epoch_step_cap = min(epoch_step_cap, full_steps_per_epoch)
    total_steps = epoch_step_cap * cfg.max_epochs
    if max_steps_cap is not None:
        total_steps = min(total_steps, max_steps_cap)

    warmup_steps = int(total_steps * cfg.warmup_frac)
    print(
        f"Device={device} | vocab={tokenizer.vocab_size} | "
        f"params={model.get_num_params():,} | "
        f"train_windows={len(train_ds):,} | val_windows={len(val_ds):,} | "
        f"steps/epoch={epoch_step_cap} | total_steps={total_steps} | "
        f"temp_warmup_steps={warmup_steps} | "
        f"λ={cfg.lambda_balance}→{cfg.lambda_balance_end}"
    )

    wandb_run = None
    if cfg.wandb:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb_project,
            config={**asdict(cfg), **asdict(model_cfg)},
        )

    global_step = 0
    best_hard_ppl = float("inf")
    epoch_reports: list[dict[str, Any]] = []
    t0 = time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.max_epochs}", leave=True)
        epoch_loss_sum = 0.0
        epoch_ce_sum = 0.0
        epoch_steps = 0
        last_diag: dict[str, float] | None = None
        last_tau = cfg.init_temp
        last_lam = cfg.lambda_balance

        for input_ids, targets in pbar:
            if max_steps_cap is not None and global_step >= max_steps_cap:
                break
            if epoch_steps >= epoch_step_cap:
                break

            # --- temperature: warmup then exponential decay ---
            tau = annealed_temperature(
                global_step,
                total_steps,
                cfg.init_temp,
                cfg.min_temp,
                warmup_frac=cfg.warmup_frac,
            )
            model.set_temperature(tau)
            lam = dynamic_lambda_balance(
                global_step,
                total_steps,
                cfg.lambda_balance,
                cfg.lambda_balance_end,
            )
            last_tau, last_lam = tau, lam

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            total_loss, ce_loss, bal_loss = compute_batch_loss(
                model,
                input_ids,
                targets,
                lambda_balance=lam,
                mode="soft",
            )
            total_loss.backward()

            # Diagnostics need grads — capture before next zero_grad.
            last_diag = model.get_routing_diagnostics()

            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            global_step += 1
            epoch_steps += 1
            epoch_loss_sum += float(total_loss.detach())
            epoch_ce_sum += float(ce_loss.detach())

            pbar.set_postfix(
                loss=f"{float(total_loss.detach()):.3f}",
                ce=f"{float(ce_loss.detach()):.3f}",
                bal=f"{float(bal_loss.detach()):.4f}",
                tau=f"{tau:.3f}",
                util=f"{last_diag['leaf_utilization_pct']:.0f}%",
                split=f"{last_diag['router_split_ratio_mean']:.2f}",
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": float(total_loss.detach()),
                        "train/ce": float(ce_loss.detach()),
                        "train/balance": float(bal_loss.detach()),
                        "train/temperature": tau,
                        "train/lambda_balance": lam,
                        "train/leaf_utilization_pct": last_diag["leaf_utilization_pct"],
                        "train/router_split_ratio_mean": last_diag[
                            "router_split_ratio_mean"
                        ],
                        "train/router_grad_norm": last_diag["router_grad_norm"],
                        "train/leaf_grad_norm": last_diag["leaf_grad_norm"],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            # --- periodic soft vs hard validation ---
            if global_step % cfg.eval_interval == 0 or global_step == total_steps:
                soft_metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    lam,
                    cfg.eval_batches,
                    mode="soft",
                )
                hard_metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    lam,
                    cfg.eval_batches,
                    mode="hard",
                )
                ppl_gap = hard_metrics["ppl"] - soft_metrics["ppl"]
                acc_gap = soft_metrics["acc"] - hard_metrics["acc"]
                print(
                    f"\n[step {global_step}] τ={tau:.4f} λ={lam:.4f} | "
                    f"soft ppl={soft_metrics['ppl']:.2f} acc={soft_metrics['acc']:.3f} | "
                    f"hard ppl={hard_metrics['ppl']:.2f} acc={hard_metrics['acc']:.3f} | "
                    f"Δppl(hard-soft)={ppl_gap:.2f} Δacc(soft-hard)={acc_gap:.3f} | "
                    f"util={last_diag['leaf_utilization_pct']:.1f}% "
                    f"split={last_diag['router_split_ratio_mean']:.3f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "val/soft_ce": soft_metrics["ce"],
                            "val/soft_ppl": soft_metrics["ppl"],
                            "val/soft_acc": soft_metrics["acc"],
                            "val/soft_balance": soft_metrics["balance"],
                            "val/hard_ce": hard_metrics["ce"],
                            "val/hard_ppl": hard_metrics["ppl"],
                            "val/hard_acc": hard_metrics["acc"],
                            "val/ppl_gap_hard_minus_soft": ppl_gap,
                            "val/acc_gap_soft_minus_hard": acc_gap,
                            "train/temperature": tau,
                            "train/lambda_balance": lam,
                        },
                        step=global_step,
                    )

                if hard_metrics["ppl"] < best_hard_ppl:
                    best_hard_ppl = hard_metrics["ppl"]

                model.train()

        # --- end-of-epoch diagnostic report (especially first 3 epochs) ---
        if epoch_steps > 0 and last_diag is not None:
            mean_loss = epoch_loss_sum / epoch_steps
            mean_ce = epoch_ce_sum / epoch_steps
            report = {
                "epoch": epoch + 1,
                "step": global_step,
                "mean_loss": mean_loss,
                "mean_ce": mean_ce,
                "tau": last_tau,
                "lambda_balance": last_lam,
                **last_diag,
            }
            epoch_reports.append(report)
            print(
                format_diagnostic_report(
                    epoch + 1,
                    global_step,
                    last_tau,
                    last_lam,
                    mean_loss,
                    mean_ce,
                    last_diag,
                )
            )

        if max_steps_cap is not None and global_step >= max_steps_cap:
            break

    elapsed = time.time() - t0
    # Final eval + checkpoint
    soft_metrics = evaluate(
        model,
        val_loader,
        device,
        cfg.lambda_balance_end,
        cfg.eval_batches,
        mode="soft",
    )
    hard_metrics = evaluate(
        model,
        val_loader,
        device,
        cfg.lambda_balance_end,
        cfg.eval_batches,
        mode="hard",
    )
    print(
        f"\nDone in {elapsed:.1f}s | final soft ppl={soft_metrics['ppl']:.2f} | "
        f"final hard ppl={hard_metrics['ppl']:.2f}"
    )

    if epoch_reports:
        print("\n### Summary — first epoch diagnostics ###")
        for r in epoch_reports[:3]:
            print(
                f"  epoch {r['epoch']}: loss={r['mean_loss']:.4f} "
                f"util={r['leaf_utilization_pct']:.1f}% "
                f"split={r['router_split_ratio_mean']:.3f} "
                f"τ={r['tau']:.3f} λ={r['lambda_balance']:.3f} "
                f"‖g_r‖={r['router_grad_norm']:.2e} ‖g_ℓ‖={r['leaf_grad_norm']:.2e}"
            )

    ckpt_path = Path(cfg.checkpoint_path)
    save_checkpoint(
        ckpt_path,
        model,
        optimizer,
        tokenizer,
        cfg,
        step=global_step,
        extras={
            "final_val_soft": soft_metrics,
            "final_val_hard": hard_metrics,
            "best_hard_ppl": best_hard_ppl,
            "epoch_diagnostic_reports": epoch_reports,
        },
    )

    if wandb_run is not None:
        wandb_run.summary["best_hard_ppl"] = best_hard_ppl
        wandb_run.finish()


if __name__ == "__main__":
    # Avoid thread over-subscription on small CPU demos.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
