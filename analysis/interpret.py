"""
Mechanistic Interpretability Analysis for Boolean Grokking
==========================================================

Key analyses:
  1. Loss curves & grokking visualization
  2. Attention pattern heatmaps
  3. Embedding space visualization (PCA)
  4. Logit lens (what does each layer "think" the answer is?)
  5. Ablation studies (zero out heads/MLP to find load-bearing components)
  6. Weight matrix analysis
  7. HTML report (all plots + all numbers in one file)

Usage:
  python -m analysis.interpret --op XOR --wd 1.0
  python -m analysis.interpret --op ALL --wd 0.0
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import json
import base64
from typing import Optional

from models.transformer import BooleanTransformer, TransformerConfig
from data.dataset import (
    VOCAB_SIZE, SEQ_LEN, TOKEN_NAMES,
    FALSE_TOKEN, TRUE_TOKEN, EQ_TOKEN,
    OPERATIONS, SingleOpDataset, AllOpsDataset
)


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def _tag(op_name: str, wd: float) -> str:
    """Canonical experiment tag used in all filenames e.g. XOR_wd1.0"""
    return f"{op_name}_wd{wd}"


def _img_to_base64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ─────────────────────────────────────────────────────────────────
# 1. TRAINING DYNAMICS
# ─────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, op_name: str, wd: float,
                         save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    tag = _tag(op_name, wd)
    fig.suptitle(f'Training Dynamics: {tag}', fontsize=14, fontweight='bold')

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
        print(f"  Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 2. ATTENTION PATTERNS
# ─────────────────────────────────────────────────────────────────

def plot_attention_patterns(model: BooleanTransformer, op_name: str, wd: float,
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

    tag = _tag(op_name, wd)
    fig.suptitle(f'Attention Patterns: {tag}', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────
# 3. EMBEDDING SPACE
# ─────────────────────────────────────────────────────────────────

def plot_embedding_space(model: BooleanTransformer, op_name: str, wd: float,
                         save_path: Optional[str] = None):
    from sklearn.decomposition import PCA

    W_E = model.W_E.weight.detach().cpu().numpy()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(W_E)

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        FALSE_TOKEN: 'steelblue', TRUE_TOKEN: 'crimson',
        2: 'forestgreen', 3: 'darkorange', 4: 'purple',
        5: 'brown', 6: 'pink', 7: 'gray', EQ_TOKEN: 'black',
    }

    for tok_id, name in TOKEN_NAMES.items():
        if tok_id >= VOCAB_SIZE:
            continue
        c = coords[tok_id]
        ax.scatter(c[0], c[1], s=200, color=colors.get(tok_id, 'gray'), zorder=3)
        ax.annotate(name, c, textcoords="offset points", xytext=(5, 5), fontsize=12)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    tag = _tag(op_name, wd)
    ax.set_title(f'Token Embedding Space (PCA): {tag}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()

    return {
        'pc1_variance': float(pca.explained_variance_ratio_[0]),
        'pc2_variance': float(pca.explained_variance_ratio_[1]),
        'coords': {
            TOKEN_NAMES[i]: coords[i].tolist()
            for i in range(min(len(TOKEN_NAMES), VOCAB_SIZE))
        }
    }


# ─────────────────────────────────────────────────────────────────
# 4. LOGIT LENS
# ─────────────────────────────────────────────────────────────────

def logit_lens(model: BooleanTransformer, op_name: str, wd: float,
               device: str = 'cpu', save_path: Optional[str] = None):
    op_token = OPERATIONS[op_name][0]
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    W_U = model.W_U.weight.detach()
    ln_final = model.ln_final

    fig, axes = plt.subplots(len(inputs), 1, figsize=(10, 3 * len(inputs)))
    results = {}

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

        results[f'{a}_{op_name}_{b}'] = {
            'expected': bool(true_result),
            'prob_true_per_stage': {s: float(p) for s, p in zip(stages, prob_true)}
        }

        ax.bar(range(len(stages)), prob_true,
               color=['steelblue' if p < 0.5 else 'crimson' for p in prob_true])
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, rotation=30, ha='right')
        ax.set_ylabel('P(True)')
        ax.set_ylim(0, 1)
        expected = 'T' if true_result else 'F'
        ax.set_title(f'Input: {a} {op_name} {b} = {expected}')

    tag = _tag(op_name, wd)
    fig.suptitle(f'Logit Lens: {tag}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()

    return results


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
    results = {'baseline': float(baseline), 'components': {}}

    print(f"  Baseline: {baseline:.4f}")
    print(f"  {'Component':<20} {'Acc':<10} {'Drop':<10} {'Load-bearing?'}")
    print(f"  {'-' * 55}")

    for layer_idx, block in enumerate(model.blocks):
        for head_idx in range(model.cfg.n_heads):
            orig_W_O = block.attn.W_O.data.clone()
            block.attn.W_O.data[head_idx] = 0.0
            acc = get_acc(model)
            drop = baseline - acc
            block.attn.W_O.data = orig_W_O
            important = drop > 0.1
            key = f'L{layer_idx}H{head_idx}'
            results['components'][key] = {
                'accuracy': float(acc), 'drop': float(drop), 'load_bearing': important
            }
            print(f"  {key:<20} {acc:.4f}     {drop:+.4f}     {'✓ YES' if important else 'no'}")

        if block.mlp is not None:
            orig_W_out = block.mlp.W_out.data.clone()
            block.mlp.W_out.data = torch.zeros_like(block.mlp.W_out.data)
            acc = get_acc(model)
            drop = baseline - acc
            block.mlp.W_out.data = orig_W_out
            important = drop > 0.1
            key = f'L{layer_idx}_MLP'
            results['components'][key] = {
                'accuracy': float(acc), 'drop': float(drop), 'load_bearing': important
            }
            print(f"  {key:<20} {acc:.4f}     {drop:+.4f}     {'✓ YES' if important else 'no'}")

    return results


# ─────────────────────────────────────────────────────────────────
# 6. WEIGHT MATRIX STRUCTURE
# ─────────────────────────────────────────────────────────────────

def plot_weight_matrices(model: BooleanTransformer, op_name: str, wd: float,
                         save_path: Optional[str] = None):
    layer = model.blocks[0]
    matrices = {
        'W_E (Embedding)': model.W_E.weight.detach().cpu().numpy(),
        'W_Q Head 0': layer.attn.W_Q[0].detach().cpu().numpy(),
        'W_K Head 0': layer.attn.W_K[0].detach().cpu().numpy(),
        'W_V Head 0': layer.attn.W_V[0].detach().cpu().numpy(),
    }

    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 5))
    stats = {}

    for ax, (name, mat) in zip(axes, matrices.items()):
        norm = TwoSlopeNorm(vmin=mat.min(), vcenter=0, vmax=mat.max())
        im = ax.imshow(mat, cmap='RdBu_r', norm=norm, aspect='auto')
        ax.set_title(name, fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.7)
        stats[name] = {
            'min': float(mat.min()),
            'max': float(mat.max()),
            'mean': float(mat.mean()),
            'std': float(mat.std()),
            'frobenius_norm': float(np.linalg.norm(mat)),
            'sparsity_pct': float((np.abs(mat) < 0.01).mean()),
        }

    tag = _tag(op_name, wd)
    fig.suptitle(f'Weight Matrix Structure: {tag}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()

    return stats


# ─────────────────────────────────────────────────────────────────
# 7. NEURON ANALYSIS
# ─────────────────────────────────────────────────────────────────

def neuron_analysis(model: BooleanTransformer, op_name: str, wd: float,
                    device: str = 'cpu', save_dir: Optional[Path] = None,
                    top_n: int = 10):
    """
    Neuron-level analysis of the MLP hidden layer.

    For each input, records the activation of every neuron in the MLP
    hidden layer (after the nonlinearity). Then:
      - Ranks neurons by variance across inputs (high variance = more discriminative)
      - Ablates each neuron one at a time and measures accuracy drop
      - Saves a heatmap + bar chart and returns numerical results
    """
    from data.dataset import make_loaders

    op_token = OPERATIONS[op_name][0]
    inputs   = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # ── Collect MLP hidden activations via hook ──────────────────
    activation_store = {}

    def mlp_hook(module, input, output):
        activation_store['mlp_hidden'] = output.detach().cpu()

    # Hook onto the first MLP's hidden linear (W_in output, before W_out)
    hooks = []
    for block in model.blocks:
        if block.mlp is not None:
            hooks.append(block.mlp.W_in.register_forward_hook(mlp_hook))
            break  # only first layer

    acts_per_input = []
    input_labels   = []

    model.eval()
    with torch.no_grad():
        for (a, b) in inputs:
            tokens = torch.tensor([[a, op_token, b, EQ_TOKEN]],
                                   dtype=torch.long).to(device)
            activation_store.clear()
            model(tokens)
            # shape: (1, seq_len, d_mlp) — take last position
            act = activation_store['mlp_hidden'][0, -1, :].numpy()
            acts_per_input.append(act)
            result = OPERATIONS[op_name][1](a, b)
            input_labels.append(f'{a} {op_name} {b} = {int(result)}')

    for h in hooks:
        h.remove()

    acts = np.stack(acts_per_input)   # (4, d_mlp)
    n_neurons  = acts.shape[1]
    variance   = np.var(acts,       axis=0)
    mean_abs   = np.mean(np.abs(acts), axis=0)
    top_idx    = np.argsort(variance)[::-1][:top_n]

    print(f"\n  Top {top_n} neurons by input-variance ({op_name}):")
    print(f"  {'Neuron':>8}  {'Variance':>10}  {'Mean |act|':>12}")
    neuron_stats = []
    for idx in top_idx:
        print(f"  Neuron {idx:>3}   {variance[idx]:>10.4f}   {mean_abs[idx]:>12.4f}")
        neuron_stats.append({
            'neuron': int(idx),
            'variance': float(variance[idx]),
            'mean_abs': float(mean_abs[idx]),
        })

    # ── Per-neuron ablation ──────────────────────────────────────
    dataset     = SingleOpDataset(op_name, n_copies=100)
    _, test_loader = make_loaders(dataset, train_frac=0.5)

    def get_acc(m):
        m.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for toks, labels in test_loader:
                toks, labels = toks.to(device), labels.to(device)
                out = m(toks)
                correct += (out.argmax(-1) == labels).sum().item()
                total   += len(labels)
        return correct / total

    # identify the MLP W_in layer
    mlp_layer = None
    for block in model.blocks:
        if block.mlp is not None:
            mlp_layer = block.mlp.W_in
            break

    baseline_acc = get_acc(model)
    print(f"\n  Neuron ablation baseline: {baseline_acc:.4f}")
    print(f"  {'Neuron':>8}  {'Acc':>8}  {'Drop':>8}  {'Load-bearing?'}")
    print(f"  {'-'*50}")

    ablation_drops = []
    for neuron_idx in range(n_neurons):
        orig_w = mlp_layer.weight.data[neuron_idx].clone()
        orig_b = mlp_layer.bias.data[neuron_idx].clone() if mlp_layer.bias is not None else None

        mlp_layer.weight.data[neuron_idx] = 0.0
        if mlp_layer.bias is not None:
            mlp_layer.bias.data[neuron_idx] = 0.0

        acc  = get_acc(model)
        drop = baseline_acc - acc

        mlp_layer.weight.data[neuron_idx] = orig_w
        if mlp_layer.bias is not None:
            mlp_layer.bias.data[neuron_idx] = orig_b

        ablation_drops.append({
            'neuron': neuron_idx,
            'accuracy': float(acc),
            'drop': float(drop),
            'load_bearing': drop > 0.05,
        })

        if drop > 0.05:
            print(f"  Neuron {neuron_idx:>3}   {acc:>8.4f}  {drop:>8.4f}  ✓ YES")

    load_bearing_neurons = [d for d in ablation_drops if d['load_bearing']]
    print(f"\n  Load-bearing neurons: {len(load_bearing_neurons)} / {n_neurons}")

    # ── Plots ────────────────────────────────────────────────────
    tag = _tag(op_name, wd)
    saved_paths = {}

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Heatmap: inputs × top neurons
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle(f'Neuron Analysis: {tag} / {op_name}', fontsize=13, fontweight='bold')

        ax = axes[0]
        im = ax.imshow(acts[:, top_idx], aspect='auto', cmap='RdBu_r')
        ax.set_xticks(range(top_n))
        ax.set_xticklabels([f'N{i}' for i in top_idx], rotation=45, fontsize=8)
        ax.set_yticks(range(len(input_labels)))
        ax.set_yticklabels(input_labels, fontsize=9)
        ax.set_xlabel('Neuron (sorted by variance)')
        ax.set_ylabel('Input')
        ax.set_title(f'Top {top_n} neuron activations')
        plt.colorbar(im, ax=ax)

        ax = axes[1]
        ax.bar(range(top_n), variance[top_idx], color='steelblue')
        ax.set_xticks(range(top_n))
        ax.set_xticklabels([f'N{i}' for i in top_idx], rotation=45, fontsize=8)
        ax.set_xlabel('Neuron index')
        ax.set_ylabel('Variance across inputs')
        ax.set_title('Input variance (higher = more discriminative)')

        plt.tight_layout()
        heatmap_path = str(save_dir / f'neuron_heatmap_{tag}_{op_name}.png')
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_paths['heatmap'] = heatmap_path
        print(f"  Saved: {heatmap_path}")

        # Bar chart: ablation drops for all neurons
        neurons   = [d['neuron'] for d in ablation_drops]
        drop_vals = [d['drop']   for d in ablation_drops]
        colors    = ['crimson' if d > 0.05 else 'steelblue' for d in drop_vals]

        fig, ax = plt.subplots(figsize=(max(8, n_neurons * 0.3), 4))
        ax.bar(neurons, drop_vals, color=colors)
        ax.axhline(0.05, linestyle='--', color='gray', alpha=0.7, label='5% threshold')
        ax.set_xlabel('Neuron index')
        ax.set_ylabel('Accuracy drop when ablated')
        ax.set_title(f'Per-neuron ablation: {tag} / {op_name}')
        ax.legend()
        plt.tight_layout()
        ablation_path = str(save_dir / f'neuron_ablation_{tag}_{op_name}.png')
        plt.savefig(ablation_path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_paths['ablation'] = ablation_path
        print(f"  Saved: {ablation_path}")

    return {
        'baseline_accuracy': float(baseline_acc),
        'n_neurons': n_neurons,
        'top_neurons_by_variance': neuron_stats,
        'ablation_drops': ablation_drops,
        'load_bearing_count': len(load_bearing_neurons),
        'saved_paths': saved_paths,
    }


# ─────────────────────────────────────────────────────────────────
# 8. HTML REPORT
# ─────────────────────────────────────────────────────────────────

def generate_html_report(
    op_name: str, wd: float, cfg_dict: dict, history: dict,
    embedding_stats: dict, weight_stats: dict,
    ablation_results: dict, logit_results: dict,
    neuron_results: dict,
    plots_dir: Path, ops_to_analyze: list, save_path: str
):
    tag = _tag(op_name, wd)

    def img_tag(path, caption=''):
        if not Path(path).exists():
            return f'<p style="color:red;font-size:12px">Missing: {path}</p>'
        b64 = _img_to_base64(path)
        return (
            f'<figure style="margin:0 0 16px 0">'
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,0.12)"/>'
            f'<figcaption style="color:#666;font-size:12px;margin-top:4px">{caption}</figcaption>'
            f'</figure>'
        )

    def kv_table(rows: dict):
        rows_html = ''.join(
            f'<tr><td style="padding:5px 12px;font-weight:500;color:#444">{k}</td>'
            f'<td style="padding:5px 12px;font-family:monospace;color:#222">{v}</td></tr>'
            for k, v in rows.items()
        )
        return (
            f'<table style="border-collapse:collapse;width:100%;'
            f'background:#f7f7f7;border-radius:6px;overflow:hidden;margin-bottom:16px">'
            f'{rows_html}</table>'
        )

    # Config
    config_rows = {
        'Operation': op_name, 'Weight Decay': str(wd),
        'Epochs': str(cfg_dict.get('n_epochs', '?')),
        'd_model': str(cfg_dict.get('d_model', '?')),
        'n_heads': str(cfg_dict.get('n_heads', '?')),
        'n_layers': str(cfg_dict.get('n_layers', '?')),
        'd_mlp': str(cfg_dict.get('d_mlp', '?')),
        'Learning Rate': str(cfg_dict.get('lr', '?')),
    }

    # Training summary
    training_rows = {
        'Final Train Accuracy': f"{history['train_acc'][-1]:.4f}",
        'Final Test Accuracy': f"{history['test_acc'][-1]:.4f}",
        'Final Train Loss': f"{history['train_loss'][-1]:.6f}",
        'Final Test Loss': f"{history['test_loss'][-1]:.6f}",
        'Final Weight Norm': f"{history['weight_norm'][-1]:.4f}",
        'Total Epochs': str(history['epoch'][-1]),
    }

    # Embedding stats
    emb_rows = {
        'PC1 Variance': f"{embedding_stats['pc1_variance']:.1%}",
        'PC2 Variance': f"{embedding_stats['pc2_variance']:.1%}",
    }
    for tok, coord in embedding_stats['coords'].items():
        emb_rows[f'{tok} (PC1, PC2)'] = f"({coord[0]:.4f}, {coord[1]:.4f})"

    # Weight matrix stats
    wm_html = ''
    for mat_name, s in weight_stats.items():
        wm_html += f'<h4 style="margin:12px 0 4px 0;color:#333">{mat_name}</h4>'
        wm_html += kv_table({
            'Min / Max': f"{s['min']:.6f} / {s['max']:.6f}",
            'Mean / Std': f"{s['mean']:.6f} / {s['std']:.6f}",
            'Frobenius Norm': f"{s['frobenius_norm']:.6f}",
            'Sparsity (|w|<0.01)': f"{s['sparsity_pct']:.1%}",
        })

    # Per-operation sections
    op_sections = ''
    for op in ops_to_analyze:
        abl = ablation_results.get(op, {})
        log = logit_results.get(op, {})

        if op_name == 'ALL':
            attn_path = str(plots_dir / op / f'attention_patterns_{tag}_{op}.png')
            logit_path = str(plots_dir / op / f'logit_lens_{tag}_{op}.png')
        else:
            attn_path = str(plots_dir / f'attention_patterns_{tag}_{op}.png')
            logit_path = str(plots_dir / f'logit_lens_{tag}_{op}.png')

        # Ablation table
        abl_html = ''
        if abl:
            baseline = abl.get('baseline', 0)
            abl_html += (
                f'<p style="font-weight:bold;color:#333">Baseline Accuracy: {baseline:.4f}</p>'
                f'<table style="border-collapse:collapse;width:100%;margin-bottom:12px">'
                f'<tr style="background:#dde"><th style="padding:5px 10px">Component</th>'
                f'<th>Accuracy</th><th>Drop</th><th>Load-bearing?</th></tr>'
            )
            for comp, vals in abl.get('components', {}).items():
                bg = '#ffe8e8' if vals['load_bearing'] else '#f9f9f9'
                flag = '✓ YES' if vals['load_bearing'] else 'no'
                abl_html += (
                    f'<tr style="background:{bg}">'
                    f'<td style="padding:5px 10px;font-family:monospace">{comp}</td>'
                    f'<td style="padding:5px 10px;font-family:monospace">{vals["accuracy"]:.4f}</td>'
                    f'<td style="padding:5px 10px;font-family:monospace">{vals["drop"]:+.4f}</td>'
                    f'<td style="padding:5px 10px">{flag}</td>'
                    f'</tr>'
                )
            abl_html += '</table>'

        # Logit lens table
        log_html = ''
        if log:
            log_html += (
                f'<table style="border-collapse:collapse;width:100%;margin-bottom:12px">'
                f'<tr style="background:#dde"><th style="padding:5px 10px">Input</th>'
                f'<th>Expected</th><th>Stage</th><th>P(True)</th><th>Correct?</th></tr>'
            )
            for inp_key, vals in log.items():
                expected = 'T' if vals['expected'] else 'F'
                for stage, p in vals['prob_true_per_stage'].items():
                    correct = (p > 0.5) == vals['expected']
                    bg = '#e8ffe8' if correct else '#ffe8e8'
                    log_html += (
                        f'<tr style="background:{bg}">'
                        f'<td style="padding:4px 10px;font-family:monospace">{inp_key}</td>'
                        f'<td style="padding:4px 10px">{expected}</td>'
                        f'<td style="padding:4px 10px">{stage}</td>'
                        f'<td style="padding:4px 10px;font-family:monospace">{p:.4f}</td>'
                        f'<td style="padding:4px 10px">{"✓" if correct else "✗"}</td>'
                        f'</tr>'
                    )
            log_html += '</table>'

        # Neuron analysis section
        nr = neuron_results.get(op, {})
        neuron_html = ''
        if nr:
            heatmap_path  = nr.get('saved_paths', {}).get('heatmap', '')
            ablation_path = nr.get('saved_paths', {}).get('ablation', '')
            neuron_html += (
                f'<p style="color:#333"><b>Total neurons:</b> {nr.get("n_neurons","?")} &nbsp;|&nbsp; '
                f'<b>Load-bearing:</b> {nr.get("load_bearing_count","?")} &nbsp;|&nbsp; '
                f'<b>Baseline acc:</b> {nr.get("baseline_accuracy", 0):.4f}</p>'
            )
            neuron_html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">'
            neuron_html += f'<div>{img_tag(heatmap_path, "Top neuron activations heatmap")}</div>'
            neuron_html += f'<div>{img_tag(ablation_path, "Per-neuron ablation drops")}</div>'
            neuron_html += '</div>'

            # Top neurons table
            top_neurons = nr.get('top_neurons_by_variance', [])
            if top_neurons:
                neuron_html += (
                    '<table style="border-collapse:collapse;width:100%;margin-top:8px">'
                    '<tr style="background:#dde"><th style="padding:5px 10px">Neuron</th>'
                    '<th>Variance</th><th>Mean |act|</th><th>Load-bearing?</th></tr>'
                )
                abl_by_neuron = {d['neuron']: d for d in nr.get('ablation_drops', [])}
                for n in top_neurons:
                    abl = abl_by_neuron.get(n['neuron'], {})
                    lb  = abl.get('load_bearing', False)
                    bg  = '#ffe8e8' if lb else '#f9f9f9'
                    neuron_html += (
                        f'<tr style="background:{bg}">'
                        f'<td style="padding:5px 10px;font-family:monospace">N{n["neuron"]}</td>'
                        f'<td style="padding:5px 10px;font-family:monospace">{n["variance"]:.4f}</td>'
                        f'<td style="padding:5px 10px;font-family:monospace">{n["mean_abs"]:.4f}</td>'
                        f'<td style="padding:5px 10px">{"✓ YES" if lb else "no"}</td>'
                        f'</tr>'
                    )
                neuron_html += '</table>'

        op_sections += f'''
        <div style="border:1px solid #ccc;border-radius:8px;padding:20px;margin-bottom:24px">
            <h3 style="color:#2c5f8a;margin-top:0">Operation: {op}</h3>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
                <div>{img_tag(attn_path, f'Attention Patterns — {op}')}</div>
                <div>{img_tag(logit_path, f'Logit Lens — {op}')}</div>
            </div>
            <h4 style="margin:0 0 8px 0">Ablation Study</h4>
            {abl_html}
            <h4 style="margin:0 0 8px 0">Logit Lens (Numerical)</h4>
            {log_html}
            <h4 style="margin:0 0 8px 0">Neuron Analysis</h4>
            {neuron_html}
        </div>'''

    training_path = str(plots_dir / f'training_curves_{tag}.png')
    embed_path    = str(plots_dir / f'embeddings_{tag}.png')
    wm_path       = str(plots_dir / f'weight_matrices_{tag}.png')

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Report: {tag}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 1200px; margin: 0 auto; padding: 24px; color: #222; background: #fff;
  }}
  h1 {{ color: #1a3a5c; border-bottom: 3px solid #2c5f8a; padding-bottom: 8px; }}
  h2 {{ color: #2c5f8a; border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 32px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
</style>
</head>
<body>

<h1>Experiment Report: {tag}</h1>
<p style="color:#888;font-size:13px">Auto-generated by interpret.py</p>

<h2>1. Configuration</h2>
<div class="grid2">
  <div><h4>Model & Training Config</h4>{kv_table(config_rows)}</div>
  <div><h4>Training Summary</h4>{kv_table(training_rows)}</div>
</div>

<h2>2. Training Dynamics</h2>
{img_tag(training_path, 'Accuracy / Loss / Weight Norm over epochs')}

<h2>3. Embedding Space</h2>
<div class="grid2">
  <div>{img_tag(embed_path, 'PCA of token embeddings')}</div>
  <div><h4>Embedding Coordinates</h4>{kv_table(emb_rows)}</div>
</div>

<h2>4. Weight Matrices</h2>
{img_tag(wm_path, 'W_E, W_Q, W_K, W_V heatmaps')}
{wm_html}

<h2>5. Per-Operation Analysis</h2>
{op_sections}

</body>
</html>'''

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n  ✓ HTML report: {save_path}")


# ─────────────────────────────────────────────────────────────────
# 8. MAIN RUNNER
# ─────────────────────────────────────────────────────────────────

def run_full_analysis(op_name: str, wd: float = 1.0,
                      checkpoint_dir: str = 'checkpoints',
                      device: str = 'cpu'):
    tag = _tag(op_name, wd)
    ops_to_analyze = list(OPERATIONS.keys()) if op_name == 'ALL' else [op_name]

    save_dir  = Path(checkpoint_dir) / op_name
    plots_dir = save_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'config.json') as f:
        cfg_dict = json.load(f)
    with open(save_dir / 'history.json') as f:
        history = json.load(f)

    model_cfg = TransformerConfig(
        vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN,
        d_model=cfg_dict['d_model'], n_heads=cfg_dict['n_heads'],
        n_layers=cfg_dict['n_layers'], d_mlp=cfg_dict['d_mlp'],
        use_mlp=cfg_dict['use_mlp'],
    )
    model = BooleanTransformer(model_cfg).to(device)
    model.load_state_dict(torch.load(save_dir / 'final_model.pt', map_location=device))
    model.eval()
    print(f"Loaded model: {tag}\n")

    print("[1/7] Training curves...")
    plot_training_curves(history, op_name, wd,
        save_path=str(plots_dir / f'training_curves_{tag}.png'))

    print("\n[2/7] Attention patterns...")
    for op in ops_to_analyze:
        op_dir = (plots_dir / op) if op_name == 'ALL' else plots_dir
        op_dir.mkdir(exist_ok=True)
        print(f"  -> {op}")
        plot_attention_patterns(model, op, wd, device,
            save_path=str(op_dir / f'attention_patterns_{tag}_{op}.png'))

    print("\n[3/7] Embedding space...")
    embedding_stats = plot_embedding_space(model, op_name, wd,
        save_path=str(plots_dir / f'embeddings_{tag}.png'))

    print("\n[4/7] Logit lens...")
    logit_results = {}
    for op in ops_to_analyze:
        op_dir = (plots_dir / op) if op_name == 'ALL' else plots_dir
        op_dir.mkdir(exist_ok=True)
        print(f"  -> {op}")
        logit_results[op] = logit_lens(model, op, wd, device,
            save_path=str(op_dir / f'logit_lens_{tag}_{op}.png'))

    print("\n[5/7] Ablation study...")
    ablation_results = {}
    for op in ops_to_analyze:
        print(f"\n  === {op} ===")
        ablation_results[op] = ablation_study(model, op, device)

    print("\n[6/8] Weight matrices...")
    weight_stats = plot_weight_matrices(model, op_name, wd,
        save_path=str(plots_dir / f'weight_matrices_{tag}.png'))

    print("\n[7/8] Neuron analysis...")
    neuron_results = {}
    for op in ops_to_analyze:
        op_dir = (plots_dir / op) if op_name == 'ALL' else plots_dir
        op_dir.mkdir(exist_ok=True)
        print(f"\n  === {op} ===")
        neuron_results[op] = neuron_analysis(
            model, op, wd, device, save_dir=op_dir)

    print("\n[8/8] HTML report...")
    generate_html_report(
        op_name=op_name, wd=wd, cfg_dict=cfg_dict, history=history,
        embedding_stats=embedding_stats, weight_stats=weight_stats,
        ablation_results=ablation_results, logit_results=logit_results,
        neuron_results=neuron_results,
        plots_dir=plots_dir, ops_to_analyze=ops_to_analyze,
        save_path=str(save_dir / f'report_{tag}.html')
    )

    print(f"\n✓ All outputs saved to: {save_dir}")
    return model, history


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--op',             type=str,   default='XOR')
    parser.add_argument('--wd',             type=float, default=1.0,
                        help='Weight decay used during training (for file naming)')
    parser.add_argument('--checkpoint_dir', type=str,   default='checkpoints')
    args = parser.parse_args()
    run_full_analysis(args.op, args.wd, args.checkpoint_dir)
