"""
A real, from-scratch GraphSAGE-style Graph Neural Network implemented in
plain PyTorch (no PyTorch Geometric dependency, to keep the environment
lightweight and easy to install/reproduce).

Each GraphSAGELayer performs:
    1. Neighbor aggregation: mean of neighbor features, weighted by edge
       weight (so strongly-connected shilling-clique neighbors dominate the
       aggregated signal).
    2. Concatenation of the node's own (previous-layer) features with the
       aggregated neighbor features.
    3. A learned linear transform + nonlinearity.

Stacking layers lets information propagate across the reviewer-reviewer
collusion graph, so a node's final representation depends on its NLP
features AND the behaviour of reviewers it is graph-connected to -- this is
the actual "graph neural network" the repo title promises.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_normalized_adjacency(n_nodes: int, edge_index: np.ndarray,
                                edge_weight: np.ndarray) -> torch.Tensor:
    """Build a dense weighted adjacency matrix (with self-loops) normalized
    so each row sums to 1 (mean aggregation). Dense is fine at this node
    count (hundreds); for larger graphs a sparse implementation would be
    used instead.
    """
    A = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    src, dst = edge_index
    A[src, dst] = torch.tensor(edge_weight, dtype=torch.float32)
    A[dst, src] = torch.tensor(edge_weight, dtype=torch.float32)
    # self loops so a node always retains some of its own signal even with
    # no / few neighbors
    A += torch.eye(n_nodes)
    row_sum = A.sum(dim=1, keepdim=True).clamp(min=1e-8)
    A_norm = A / row_sum
    return A_norm


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        neighbor_agg = A_norm @ x           # mean-aggregate neighbor features
        combined = torch.cat([x, neighbor_agg], dim=1)
        return F.relu(self.linear(combined))


class ShillingGNN(nn.Module):
    """2-layer GraphSAGE encoder + MLP classification head producing a
    per-reviewer shilling probability (logit).
    """
    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        self.layer1 = GraphSAGELayer(in_dim, hidden_dim)
        self.layer2 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x, A_norm)
        h = self.dropout(h)
        h = self.layer2(h, A_norm)
        h = self.dropout(h)
        logits = self.classifier(h).squeeze(-1)
        return logits

    def embed(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """Return the hidden representation (pre-classifier) for fusion."""
        h = self.layer1(x, A_norm)
        h = self.layer2(h, A_norm)
        return h


def train_gnn(model: ShillingGNN, x: torch.Tensor, A_norm: torch.Tensor,
              y: torch.Tensor, train_mask: torch.Tensor,
              epochs: int = 200, lr: float = 0.01,
              weight_decay: float = 5e-4) -> list[float]:
    """Train with binary cross-entropy on the training mask only, using
    class-weighted loss (shilling reviewers are a minority class).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_pos = y[train_mask].sum().clamp(min=1)
    n_neg = (train_mask.sum() - n_pos).clamp(min=1)
    pos_weight = (n_neg / n_pos).detach()

    losses = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(x, A_norm)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], y[train_mask], pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses
