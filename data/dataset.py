"""
Dataset generation for boolean grokking experiments.

Vocabulary:
  0 -> token 0  (FALSE)
  1 -> token 1  (TRUE)
  2 -> AND
  3 -> OR
  4 -> XOR
  5 -> NAND
  6 -> NOR
  7 -> XNOR
  8 -> = (equals / query token)
  9 -> PAD

Input format: [a, op, b, =]  (seq_len = 4)
Target: result of op(a, b)  -> 0 or 1

For multi-op tasks (phase 2):
  [a, op1, b, op2, c, =]  (seq_len = 6)
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, List, Optional
import random

# Token vocabulary
FALSE_TOKEN  = 0
TRUE_TOKEN   = 1
AND_TOKEN    = 2
OR_TOKEN     = 3
XOR_TOKEN    = 4
NAND_TOKEN   = 5
NOR_TOKEN    = 6
XNOR_TOKEN   = 7
EQ_TOKEN     = 8
PAD_TOKEN    = 9

VOCAB_SIZE   = 10
SEQ_LEN      = 4  # [a, op, b, =]

TOKEN_NAMES = {
    0: 'F', 1: 'T', 2: 'AND', 3: 'OR', 4: 'XOR',
    5: 'NAND', 6: 'NOR', 7: 'XNOR', 8: '=', 9: 'PAD'
}

# All single boolean operations
OPERATIONS = {
    'AND':  (AND_TOKEN,  lambda a, b: int(a and b)),
    'OR':   (OR_TOKEN,   lambda a, b: int(a or b)),
    'XOR':  (XOR_TOKEN,  lambda a, b: int(a ^ b)),
    'NAND': (NAND_TOKEN, lambda a, b: int(not (a and b))),
    'NOR':  (NOR_TOKEN,  lambda a, b: int(not (a or b))),
    'XNOR': (XNOR_TOKEN, lambda a, b: int(not (a ^ b))),
}


def compute_op(op_token: int, a: int, b: int) -> int:
    op_map = {v[0]: v[1] for v in OPERATIONS.values()}
    return op_map[op_token](a, b)


class SingleOpDataset(Dataset):
    """
    Dataset for a single boolean operation.
    There are only 4 possible inputs: (0,0), (0,1), (1,0), (1,1).
    We replicate them many times so the model sees enough training signal.
    """
    def __init__(self, op_name: str, n_copies: int = 1000):
        self.op_name = op_name
        op_token, op_fn = OPERATIONS[op_name]

        self.data = []
        for a in [0, 1]:
            for b in [0, 1]:
                result = op_fn(a, b)
                tokens = torch.tensor([a, op_token, b, EQ_TOKEN], dtype=torch.long)
                label  = torch.tensor(result, dtype=torch.long)
                self.data.append((tokens, label))

        # Replicate for sufficient gradient signal
        self.data = self.data * n_copies
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MultiOpDataset(Dataset):
    """
    Dataset for combinations of multiple operations.
    e.g. (a XOR b) AND c
    Input: [a, XOR, b, AND, c, =]  -> result
    This is for phase 2: does grokking happen on compositional logic?
    """
    def __init__(self, op_names: List[str], n_copies: int = 500):
        self.op_names = op_names
        self.data = []

        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    for op1_name in op_names:
                        for op2_name in op_names:
                            op1_tok, op1_fn = OPERATIONS[op1_name]
                            op2_tok, op2_fn = OPERATIONS[op2_name]
                            intermediate = op1_fn(a, b)
                            result = op2_fn(intermediate, c)
                            tokens = torch.tensor(
                                [a, op1_tok, b, op2_tok, c, EQ_TOKEN],
                                dtype=torch.long
                            )
                            label = torch.tensor(result, dtype=torch.long)
                            self.data.append((tokens, label))

        self.data = self.data * n_copies
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AllOpsDataset(Dataset):
    """
    Dataset with ALL operations mixed together.
    The model must learn to condition on the operator token.
    This is the most interesting for circuit analysis:
    does it learn one circuit per op, or a unified circuit?
    """
    def __init__(self, ops: Optional[List[str]] = None, n_copies: int = 500):
        if ops is None:
            ops = list(OPERATIONS.keys())
        self.data = []

        for op_name in ops:
            op_token, op_fn = OPERATIONS[op_name]
            for a in [0, 1]:
                for b in [0, 1]:
                    result = op_fn(a, b)
                    tokens = torch.tensor([a, op_token, b, EQ_TOKEN], dtype=torch.long)
                    label  = torch.tensor(result, dtype=torch.long)
                    self.data.append((tokens, label))

        self.data = self.data * n_copies
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def make_loaders(dataset, train_frac=0.7, batch_size=64):
    """Split dataset and return train/test DataLoaders."""
    n_train = int(len(dataset) * train_frac)
    n_test  = len(dataset) - n_train
    train_ds, test_ds = random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def decode_sequence(tokens: torch.Tensor) -> str:
    """Human-readable representation of a token sequence."""
    return ' '.join(TOKEN_NAMES[t.item()] for t in tokens)


if __name__ == '__main__':
    # Quick sanity check
    ds = AllOpsDataset()
    print(f"Dataset size: {len(ds)}")
    tokens, label = ds[0]
    print(f"Example: {decode_sequence(tokens)} -> {TOKEN_NAMES[label.item()]}")

    train_loader, test_loader = make_loaders(ds)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
