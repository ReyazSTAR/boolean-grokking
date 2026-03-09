"""
Run all boolean operations and compare their grokking behavior.

Research question: Does grokking speed/dynamics differ by operation?
  - XOR is not linearly separable → needs a different internal algorithm
  - AND/OR are linearly separable → might use a simpler circuit
  - Does the model learn different circuits for each?

Run this after training all operations.
"""

import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


OPS = ['AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR']
COLORS = {
    'AND': 'steelblue', 'OR': 'crimson', 'XOR': 'purple',
    'NAND': 'darkorange', 'NOR': 'forestgreen', 'XNOR': 'brown'
}


def train_all(epochs=5000, weight_decay=1.0):
    """Train a model for each boolean operation."""
    for op in OPS:
        print(f"\n{'='*50}")
        print(f"Training {op}...")
        print(f"{'='*50}")
        cmd = [
            'python', 'train.py',
            '--op', op,
            '--epochs', str(epochs),
            '--wd', str(weight_decay),
        ]
        subprocess.run(cmd, check=True)


def compare_grokking_curves(checkpoint_dir: str = 'checkpoints'):
    """
    Plot test accuracy for all operations on the same axes.
    Key visualization: which operations grok faster? does XOR grok at all?
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    grokking_info = {}

    for op in OPS:
        history_path = Path(checkpoint_dir) / op / 'history.json'
        if not history_path.exists():
            print(f"Skipping {op} (no history found)")
            continue

        with open(history_path) as f:
            h = json.load(f)

        epochs = h['epoch']
        color = COLORS[op]

        axes[0].plot(epochs, h['test_acc'], label=op, color=color, linewidth=2)
        axes[1].plot(epochs, h['train_acc'], label=op, color=color,
                     linewidth=2, linestyle='--', alpha=0.7)
        axes[1].plot(epochs, h['test_acc'], color=color, linewidth=2)

        # Find grokking epoch
        train_acc = np.array(h['train_acc'])
        test_acc = np.array(h['test_acc'])
        ep_arr = np.array(epochs)

        train_saturated = ep_arr[train_acc >= 0.99][0] if any(train_acc >= 0.99) else None
        grokked = ep_arr[test_acc >= 0.95][0] if any(test_acc >= 0.95) else None

        grokking_info[op] = {
            'train_saturated': int(train_saturated) if train_saturated is not None else None,
            'grokked_at': int(grokked) if grokked is not None else None,
            'grokking_gap': int(grokked - train_saturated) if (grokked and train_saturated is not None) else None,
            'final_test_acc': float(test_acc[-1]),
        }

    axes[0].axhline(0.95, color='gray', linestyle=':', alpha=0.5, label='Grokked threshold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Test Accuracy by Operation', fontweight='bold')
    axes[0].legend(); axes[0].set_ylim(-0.05, 1.05)

    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Train (dashed) vs Test (solid)', fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(Path(checkpoint_dir) / 'comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary table
    print(f"\n{'Operation':<10} | {'Train Sat.':<12} | {'Grokked At':<12} | {'Gap':<10} | {'Final Test Acc'}")
    print("-" * 65)
    for op, info in grokking_info.items():
        ts  = info['train_saturated'] or 'N/A'
        ga  = info['grokked_at'] or 'N/A'
        gap = info['grokking_gap'] or 'N/A'
        acc = f"{info['final_test_acc']:.4f}"
        print(f"{op:<10} | {str(ts):<12} | {str(ga):<12} | {str(gap):<10} | {acc}")

    return grokking_info


def compare_attention_patterns(checkpoint_dir: str = 'checkpoints'):
    """
    Side-by-side attention patterns for XOR vs AND vs OR.
    Key question: do different operations learn different attention circuits?
    """
    import torch
    from models.transformer import BooleanTransformer, TransformerConfig
    from data.dataset import VOCAB_SIZE, SEQ_LEN, OPERATIONS, EQ_TOKEN

    fig, axes = plt.subplots(len(OPS), 4, figsize=(16, 3 * len(OPS)))

    for row, op in enumerate(OPS):
        model_path = Path(checkpoint_dir) / op / 'final_model.pt'
        config_path = Path(checkpoint_dir) / op / 'config.json'
        if not model_path.exists():
            continue

        with open(config_path) as f:
            cfg_dict = json.load(f)

        model_cfg = TransformerConfig(
            vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN,
            d_model=cfg_dict['d_model'], n_heads=cfg_dict['n_heads'],
            n_layers=cfg_dict['n_layers'], d_mlp=cfg_dict['d_mlp'],
        )
        model = BooleanTransformer(model_cfg)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        op_token = OPERATIONS[op][0]
        inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        pos_labels = ['a', 'op', 'b', '=']

        for col, (a, b) in enumerate(inputs):
            tokens = torch.tensor([[a, op_token, b, EQ_TOKEN]], dtype=torch.long)
            with torch.no_grad():
                model(tokens)

            # Average attention across heads
            pattern = model.blocks[0].attn.attn_pattern[0].mean(0).numpy()  # (seq, seq)
            result = OPERATIONS[op][1](a, b)

            ax = axes[row, col]
            ax.imshow(pattern, vmin=0, vmax=1, cmap='Blues', aspect='auto')
            ax.set_xticks(range(4)); ax.set_xticklabels(pos_labels, fontsize=8)
            ax.set_yticks(range(4)); ax.set_yticklabels(pos_labels, fontsize=8)

            if col == 0:
                ax.set_ylabel(op, fontsize=11, fontweight='bold', rotation=0,
                              labelpad=35, va='center')
            ax.set_title(f'{a} {op} {b}={result}', fontsize=9)

    fig.suptitle('Attention Patterns Across Operations\n(avg over heads, each row = one op)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Path(checkpoint_dir) / 'attention_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train all operations first')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--compare', action='store_true', help='Compare grokking curves')
    parser.add_argument('--attn', action='store_true', help='Compare attention patterns')
    args = parser.parse_args()

    if args.train:
        train_all(args.epochs)
    if args.compare:
        compare_grokking_curves()
    if args.attn:
        compare_attention_patterns()
    if not any([args.train, args.compare, args.attn]):
        print("Usage:")
        print("  python experiments.py --train       # train all ops")
        print("  python experiments.py --compare     # plot grokking comparison")
        print("  python experiments.py --attn        # compare attention patterns")
        print("  python experiments.py --train --compare --attn  # full pipeline")
