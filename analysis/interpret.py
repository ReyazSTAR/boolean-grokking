"""
Mechanistic Interpretability Analysis for Boolean Grokking
==========================================================

This is where the research happens. After training, load a model and run these
analyses to reverse-engineer what circuit it learned.

Key analyses:
  1. Loss curves & grokking visualization
  2. Attention pattern heatmaps
  3. Embedding space visualization (PCA)
  4. Logit lens (what does each layer "think" the answer is?)
  5. Ablation studies (zero out heads/MLP to find load-bearing components)
  6. Weight matrix analysis (what structure is in W_E, W_Q, W_K, W_V?)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import json
from typing import Optional, List, Dict

from models.transformer import BooleanTransformer, TransformerConfig
from data.dataset import (
    VOCAB_SIZE, SEQ_LEN, TOKEN_NAMES,
    FALSE_TOKEN, TRUE_TOKEN, EQ_TOKEN,
    OPERATIONS, decode_sequence, SingleOpDataset, AllOpsDataset
)


# ─────────────────────────────────────────────────────────────────
# 1. TRAINING DYNAMICS
# ─────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, op_name: str, save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Training Dynamics: {op_name}', fontsize=14, fontweight='bold')

    epochs = history['epoch']

    axes[0].plot(epochs, history['train_acc'], label='Train', color='steelblue', linewidth=2)
    axes[0].plot(epochs, history['test_acc'],  label='Test',  color='crimson',   linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy (looking for grokking jump)')
    axes[0].legend(); axes[0].set_ylim(-0.05, 1.05)
    axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    axes[1].plot(epochs, history['train_loss'], label='Train', color='steelblue', linewidth=2)
    axes[1].plot(epochs, history['test_loss'],  label='Test',  color='crimson',   linewidth=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss')
    axes[1].legend(); axes[1].set_yscale('log')

    axes[2].plot(epochs, history['weight_norm'], color='darkorange', linewidth=2)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('||W||')
    axes[2].set_title('Weight Norm (drops during grokking cleanup)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 2. ATTENTION PATTERNS
# ─────────────────────────────────────────────────────────────────

def plot_attention_patterns(model: BooleanTransformer, op_name: str,
                             device: str = 'cpu', save_path: Optional[str] = None):
    op_token = OPERATIONS[op_name][0]
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers

    fig, axes = plt.subplots(
        n_layers * n_heads, len(inputs),
        figsize=(4 * len(inputs), 3 * n_layers * n_heads)
    )
    if n_layers * n_heads == 1:
        axes = np.array([[axes[i] for i in range(len(inputs))]])

    pos_labels = ['a', 'op', 'b', '=']

    for inp_idx, (a, b) in enumerate(inputs):
        tokens = torch.tensor([[a, op_token, b, EQ_TOKEN]], dtype=torch.long).to(device)
        with torch.no_grad():
            model(tokens)

        for layer in range(n_layers):
            pattern = model.blocks[layer].attn.attn_pattern[0]
            for head in range(n_heads):
                ax = axes[layer * n_heads + head, inp_idx]
                attn = pattern[head].cpu().numpy()
                im = ax.imshow(attn, vmin=0, vmax=1, cmap='Blues', aspect='auto')

                ax.set_xticks(range(4)); ax.set_xticklabels(pos_labels, fontsize=9)
                ax.set_yticks(range(4)); ax.set_yticklabels(pos_labels, fontsize=9)

                result = OPERATIONS[op_name][1](a, b)
                if inp_idx == 0:
                    ax.set_ylabel(f'L{layer}H{head}', fontsize=10, fontweight='bold')
                if layer == 0:
                    ax.set_title(f'{a} {op_name} {b} = {result}', fontsize=10)

                for r in range(4):
                    for c in range(4):
                        ax.text(c, r, f'{attn[r,c]:.2f}', ha='center', va='center',
                                fontsize=7, color='white' if attn[r,c] > 0.6 else 'black')

    fig.suptitle(f'Attention Patterns: {op_name}', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 3. EMBEDDING SPACE
# ─────────────────────────────────────────────────────────────────

def plot_embedding_space(model: BooleanTransformer, save_path: Optional[str] = None):
    from sklearn.decomposition import PCA

    W_E = model.W_E.weight.detach().cpu().numpy()

    pca = PCA(n_components=2)
    coords = pca.fit_transform(W_E)

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        FALSE_TOKEN: 'steelblue',
        TRUE_TOKEN: 'crimson',
        2: 'forestgreen', 3: 'darkorange', 4: 'purple',
        5: 'brown', 6: 'pink', 7: 'gray',
        EQ_TOKEN: 'black',
    }

    for tok_id, name in TOKEN_NAMES.items():
        if tok_id >= VOCAB_SIZE:
            continue
        c = coords[tok_id]
        ax.scatter(c[0], c[1], s=200, color=colors.get(tok_id, 'gray'), zorder=3)
        ax.annotate(name, c, textcoords="offset points", xytext=(5, 5), fontsize=12)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title('Token Embedding Space (PCA)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 4. LOGIT LENS
# ─────────────────────────────────────────────────────────────────

def logit_lens(model: BooleanTransformer, op_name: str,
               device: str = 'cpu', save_path: Optional[str] = None):
    op_token = OPERATIONS[op_name][0]
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    W_U = model.W_U.weight.detach()
    ln_final = model.ln_final

    fig, axes = plt.subplots(len(inputs), 1, figsize=(10, 3 * len(inputs)))

    for inp_idx, (a, b) in enumerate(inputs):
        tokens = torch.tensor([[a, op_token, b, EQ_TOKEN]], dtype=torch.long).to(device)
        with torch.no_grad():
            model(tokens)

        true_result = OPERATIONS[op_name][1](a, b)
        ax = axes[inp_idx]

        stages = ['embed'] + [f'block_{i}' for i in range(model.cfg.n_layers)]
        prob_true = []

        for stage in stages:
            h = model.residual_cache[stage]
            h_last = h[0, -1, :]
            h_normed = ln_final(h_last.unsqueeze(0)).squeeze(0)
            logits = W_U @ h_normed
            probs = torch.softmax(logits, dim=0)
            prob_true.append(probs[TRUE_TOKEN].item())

        ax.bar(range(len(stages)), prob_true,
               color=['steelblue' if p < 0.5 else 'crimson' for p in prob_true])
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, rotation=30, ha='right')
        ax.set_ylabel('P(True)')
        ax.set_ylim(0, 1)
        expected = 'T' if true_result else 'F'
        ax.set_title(f'Input: {a} {op_name} {b} = {expected}')

    fig.suptitle(f'Logit Lens: {op_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 5. ABLATION STUDIES
# ─────────────────────────────────────────────────────────────────

def ablation_study(model: BooleanTransformer, op_name: str, device: str = 'cpu'):
    from data.dataset import make_loaders
    dataset = SingleOpDataset(op_name, n_copies=100)
    _, test_loader = make_loaders(dataset, train_frac=0.5)

    def get_acc(m):
        m.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for tokens, labels in test_loader:
                tokens, labels = tokens.to(device), labels.to(device)
                logits = m(tokens)
                correct += (logits.argmax(-1) == labels).sum().item()
                total += len(labels)
        return correct / total

    baseline = get_acc(model)
    print(f"Baseline accuracy: {baseline:.4f}")
    print(f"\nAblation results (zeroing out components):")
    print(f"{'Component':<30} {'Accuracy':<12} {'Drop':<10} {'Load-bearing?'}")
    print("-" * 65)

    for layer_idx, block in enumerate(model.blocks):
        for head_idx in range(model.cfg.n_heads):
            orig_W_O = block.attn.W_O.data.clone()
            block.attn.W_O.data[head_idx] = 0.0
            acc = get_acc(model)
            drop = baseline - acc
            block.attn.W_O.data = orig_W_O
            important = "✓ YES" if drop > 0.1 else "no"
            print(f"  Layer {layer_idx} Head {head_idx}              {acc:.4f}       {drop:+.4f}     {important}")

        if block.mlp is not None:
            orig_W_out = block.mlp.W_out.data.clone()
            block.mlp.W_out.data = torch.zeros_like(block.mlp.W_out.data)
            acc = get_acc(model)
            drop = baseline - acc
            block.mlp.W_out.data = orig_W_out
            important = "✓ YES" if drop > 0.1 else "no"
            print(f"  Layer {layer_idx} MLP                    {acc:.4f}       {drop:+.4f}     {important}")

    return baseline


# ─────────────────────────────────────────────────────────────────
# 6. WEIGHT MATRIX STRUCTURE
# ─────────────────────────────────────────────────────────────────

def plot_weight_matrices(model: BooleanTransformer, save_path: Optional[str] = None):
    layer = model.blocks[0]
    matrices = {
        'W_E (Embedding)': model.W_E.weight.detach().cpu().numpy(),
        'W_Q Head 0': layer.attn.W_Q[0].detach().cpu().numpy(),
        'W_K Head 0': layer.attn.W_K[0].detach().cpu().numpy(),
        'W_V Head 0': layer.attn.W_V[0].detach().cpu().numpy(),
    }

    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 5))

    for ax, (name, mat) in zip(axes, matrices.items()):
        norm = TwoSlopeNorm(vmin=mat.min(), vcenter=0, vmax=mat.max())
        im = ax.imshow(mat, cmap='RdBu_r', norm=norm, aspect='auto')
        ax.set_title(name, fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle('Weight Matrix Structure', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 7. MAIN ANALYSIS RUNNER
# ─────────────────────────────────────────────────────────────────

def run_full_analysis(op_name: str, checkpoint_dir: str = 'checkpoints',
                      device: str = 'cpu'):
    """
    Run all analyses on a trained model.
    
    If op_name is 'ALL', loads the ALL ops model and runs per-operation
    analysis (attention, logit lens, ablation) for each operation separately.
    Embedding and weight matrix plots are shared and saved once.
    """
    # Determine which operations to analyze
    ops_to_analyze = list(OPERATIONS.keys()) if op_name == 'ALL' else [op_name]

    save_dir = Path(checkpoint_dir) / op_name
    plots_dir = save_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Load config and history
    with open(save_dir / 'config.json') as f:
        cfg_dict = json.load(f)
    with open(save_dir / 'history.json') as f:
        history = json.load(f)

    # Rebuild model
    model_cfg = TransformerConfig(
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        d_model=cfg_dict['d_model'],
        n_heads=cfg_dict['n_heads'],
        n_layers=cfg_dict['n_layers'],
        d_mlp=cfg_dict['d_mlp'],
        use_mlp=cfg_dict['use_mlp'],
    )
    model = BooleanTransformer(model_cfg).to(device)
    model.load_state_dict(torch.load(save_dir / 'final_model.pt', map_location=device))
    model.eval()
    print(f"Loaded model for {op_name}")

    # 1. Training curves — one plot for the whole model
    print("\n[1/6] Plotting training curves...")
    plot_training_curves(history, op_name,
                         save_path=str(plots_dir / 'training_curves.png'))

    # 2. Attention patterns — one plot per operation
    print("\n[2/6] Plotting attention patterns...")
    for op in ops_to_analyze:
        op_dir = plots_dir / op if op_name == 'ALL' else plots_dir
        op_dir.mkdir(exist_ok=True)
        print(f"  -> {op}")
        plot_attention_patterns(model, op, device,
                                save_path=str(op_dir / 'attention_patterns.png'))

    # 3. Embedding space — shared across all ops, saved once
    print("\n[3/6] Plotting embedding space (shared)...")
    plot_embedding_space(model, save_path=str(plots_dir / 'embeddings.png'))

    # 4. Logit lens — one plot per operation
    print("\n[4/6] Running logit lens...")
    for op in ops_to_analyze:
        op_dir = plots_dir / op if op_name == 'ALL' else plots_dir
        op_dir.mkdir(exist_ok=True)
        print(f"  -> {op}")
        logit_lens(model, op, device,
                   save_path=str(op_dir / 'logit_lens.png'))

    # 5. Ablation — one study per operation
    print("\n[5/6] Running ablation study...")
    for op in ops_to_analyze:
        print(f"\n  === Ablation for {op} ===")
        ablation_study(model, op, device)

    # 6. Weight matrices — shared, saved once
    print("\n[6/6] Plotting weight matrices...")
    plot_weight_matrices(model, save_path=str(plots_dir / 'weight_matrices.png'))

    print(f"\n✓ All plots saved to {plots_dir}")
    return model, history


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', type=str, default='XOR')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    run_full_analysis(args.op, args.checkpoint_dir)
