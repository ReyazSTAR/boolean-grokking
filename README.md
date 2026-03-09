# Boolean Grokking: Mechanistic Interpretability of Boolean Logic in Transformers

A research project investigating **what internal circuit a transformer learns when it groks boolean logic**, and how it compares to the Fourier-rotation algorithm learned for modular arithmetic.

## Research Question

Nanda et al. (2023) showed that transformers trained on modular addition learn a specific, beautiful algorithm: they represent numbers as points on a circle and use discrete Fourier transforms to add them.

**This project asks**: What algorithm does a transformer learn for boolean logic (XOR, AND, OR, etc.)? Does it grok? And is the circuit fundamentally different from the arithmetic case?

XOR is especially interesting because it's the simplest function that is **not linearly separable** — the model *must* learn a non-trivial internal representation to generalize it.

---

## Project Structure

```
boolean_grokking/
├── models/
│   └── transformer.py      # Minimal transformer with full interpretability access
├── data/
│   └── dataset.py          # Dataset generators for all boolean operations
├── analysis/
│   └── interpret.py        # Mechanistic interpretability toolkit
├── train.py                # Training loop with grokking detection
├── experiments.py          # Multi-op comparison experiments
└── README.md
```

---

## Quick Start

### Install dependencies
```bash
pip install torch numpy matplotlib scikit-learn
```

### Train a single operation (start here)
```bash
python train.py --op XOR --epochs 5000 --wd 1.0
```

This will print a table showing train vs test accuracy over time.
**Watch for grokking**: train acc hits 100% early, test acc stays ~50%, then suddenly jumps.

### Run the full interpretability analysis
```bash
python analysis/interpret.py --op XOR
```

This produces 6 plots saved to `checkpoints/XOR/plots/`:
1. `training_curves.png` — the grokking jump
2. `attention_patterns.png` — what each head attends to
3. `embeddings.png` — token embedding space (PCA)
4. `logit_lens.png` — information flow through layers
5. `weight_matrices.png` — raw weight structure
6. (ablation results printed to console)

### Train and compare ALL operations
```bash
python experiments.py --train --epochs 5000
python experiments.py --compare --attn
```

---

## The Research Agenda (step by step)

### Phase 1: Does boolean logic grok? (Week 1)
- Train XOR, AND, OR, NAND, NOR, XNOR
- Record: which operations grok? how fast?
- **Expected finding**: XOR might need more epochs (harder circuit)

### Phase 2: What's the circuit? (Week 2)
- Run attention pattern analysis on grokked models
- Run ablation studies: which heads are load-bearing?
- **Key question**: Does attention implement the logic directly, or does it route inputs to MLP?

### Phase 3: Embedding geometry (Week 2-3)
- PCA / t-SNE of token embeddings
- In modular arithmetic: embeddings form a **circle**
- **Key question**: What geometry do boolean embeddings form? A hypercube? A simplex?

### Phase 4: Attention-only vs MLP ablation (Week 3)
- Train with `--no_mlp` flag (attention-only model)
- Can attention alone implement XOR? (Hint: probably not, since XOR requires non-linear combination)
- This tells you about the role of the MLP as a "logic gate"

### Phase 5: Compositional logic (Week 4)
- Train on multi-operation sequences: (a XOR b) AND c
- Does it grok faster because it reuses circuits?
- Run `data/dataset.py MultiOpDataset`

---

## Key Hyperparameters for Grokking

| Parameter | Effect |
|-----------|--------|
| `weight_decay` | **Most important.** Higher WD → more likely to grok. Try 0.1, 1.0, 5.0 |
| `n_epochs` | Grokking can take thousands of epochs after memorization |
| `train_frac` | Lower → harder to generalize → more pronounced grokking |
| `d_model` | Larger → more capacity → might skip grokking (memorizes easily) |

If you're not seeing grokking, try increasing `weight_decay` to 5.0 or 10.0.

---

## Expected Results & What to Write About

### If XOR groks differently from AND/OR:
That's your main finding. XOR requires a fundamentally different internal representation.
Write: "We find that linearly inseparable boolean functions require qualitatively different circuits..."

### If the embedding geometry is different from the modular arithmetic circle:
Also publishable. Characterize the geometry. Is it a simplex (vertices of a triangle for 0, 1)?
Write: "Unlike modular arithmetic, which uses Fourier representations, boolean logic uses..."

### If MLP is essential for XOR but not AND/OR:
Strong finding. Tells us about what MLPs compute (they're thought to be "key-value memories").
Write: "We show that attention alone cannot implement XOR, but can implement AND/OR..."

---

## References

- Nanda et al. (2023) — *Progress Measures for Grokking* — https://arxiv.org/abs/2301.05217
- Power et al. (2022) — *Grokking* — https://arxiv.org/abs/2201.02177
- Elhage et al. (2021) — *Mathematical Framework for Transformer Circuits* — https://transformer-circuits.pub/2021/framework/index.html
- Pan et al. (2024) — *Can Transformers Reason Logically? A Study in SAT Solving* — https://arxiv.org/abs/2410.07432
