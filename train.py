"""
Training loop for boolean grokking experiments.

Key features:
  - Logs train/test accuracy AND loss at every epoch
  - Saves checkpoints at regular intervals (for post-hoc analysis)
  - Detects grokking automatically (when test acc jumps > 90% after train acc was 100%)
  - Saves the full training history for plotting
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

from models.transformer import BooleanTransformer, TransformerConfig
from data.dataset import (
    SingleOpDataset, AllOpsDataset, make_loaders,
    VOCAB_SIZE, SEQ_LEN
)


@dataclass
class TrainConfig:
    # Model
    d_model: int     = 64
    n_heads: int     = 2
    n_layers: int    = 1
    d_mlp: int       = 256
    use_mlp: bool    = True

    # Training
    lr: float        = 1e-3
    weight_decay: float = 1.0   # High WD is KEY to grokking
    n_epochs: int    = 5000
    batch_size: int  = 64
    train_frac: float = 0.7

    # Experiment
    op_name: str     = 'XOR'    # 'XOR', 'AND', 'OR', 'ALL'
    n_copies: int    = 1000     # dataset size multiplier
    save_dir: str    = 'checkpoints'
    checkpoint_every: int = 100  # save model weights every N epochs
    log_every: int   = 10


def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for tokens, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            logits = model(tokens)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def train(cfg: TrainConfig, device: Optional[str] = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    # --- Dataset ---
    if cfg.op_name == 'ALL':
        dataset = AllOpsDataset(n_copies=cfg.n_copies)
        seq_len = SEQ_LEN
    else:
        dataset = SingleOpDataset(cfg.op_name, n_copies=cfg.n_copies)
        seq_len = SEQ_LEN

    train_loader, test_loader = make_loaders(dataset, cfg.train_frac, cfg.batch_size)
    print(f"Dataset: {cfg.op_name} | Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # --- Model ---
    model_cfg = TransformerConfig(
        vocab_size=VOCAB_SIZE,
        seq_len=seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_mlp=cfg.d_mlp,
        use_mlp=cfg.use_mlp,
    )
    model = BooleanTransformer(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # --- Optimizer ---
    # AdamW with high weight decay is the standard grokking recipe
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # --- Checkpoint dir ---
    save_dir = Path(cfg.save_dir) / cfg.op_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(asdict(cfg), f, indent=2)

    # --- History ---
    history = {
        'epoch': [], 'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [], 'weight_norm': []
    }

    grokked = False
    grokked_at = None
    train_saturated_at = None  # epoch when train acc first hit 100%

    print(f"\n{'Epoch':>6} | {'Train Acc':>10} | {'Test Acc':>10} | {'Train Loss':>12} | {'Test Loss':>12} | {'||W||':>8}")
    print("-" * 72)

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(tokens)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        if epoch % cfg.log_every == 0:
            train_loss, train_acc = evaluate(model, train_loader, device)
            test_loss, test_acc   = evaluate(model, test_loader, device)
            weight_norm = sum(p.norm().item() for p in model.parameters())

            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['weight_norm'].append(weight_norm)

            # Detect grokking
            if train_acc >= 0.999 and train_saturated_at is None:
                train_saturated_at = epoch
                print(f"\n>>> Train accuracy saturated at epoch {epoch} <<<\n")

            if train_saturated_at is not None and test_acc >= 0.95 and not grokked:
                grokked = True
                grokked_at = epoch
                print(f"\n🎉 GROKKING at epoch {epoch}! (train saturated at {train_saturated_at})\n")

            print(f"{epoch:>6} | {train_acc:>10.4f} | {test_acc:>10.4f} | {train_loss:>12.6f} | {test_loss:>12.6f} | {weight_norm:>8.2f}")

        # Save checkpoint
        if epoch % cfg.checkpoint_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_dir / f'checkpoint_epoch_{epoch}.pt')

    # Save final model and history
    torch.save(model.state_dict(), save_dir / 'final_model.pt')
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f)

    print(f"\nTraining complete.")
    if grokked:
        print(f"  Grokked at epoch {grokked_at} (after {grokked_at - train_saturated_at} epochs of memorization)")
    else:
        print(f"  No grokking detected. Try increasing weight_decay or n_epochs.")

    return model, history, model_cfg


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', type=str, default='XOR',
                        choices=['XOR', 'AND', 'OR', 'NAND', 'NOR', 'XNOR', 'ALL'])
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--wd', type=float, default=1.0,
                        help='Weight decay (higher = more likely to grok)')
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--no_mlp', action='store_true')
    args = parser.parse_args()

    cfg = TrainConfig(
        op_name=args.op,
        n_epochs=args.epochs,
        weight_decay=args.wd,
        d_model=args.d_model,
        use_mlp=not args.no_mlp,
    )
    train(cfg)
