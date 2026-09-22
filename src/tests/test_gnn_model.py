import numpy as np
import torch

from src.gnn_model import ShillingGNN, build_normalized_adjacency, train_gnn


def test_build_normalized_adjacency_rows_sum_to_one():
    edge_index = np.array([[0, 1], [1, 2]])
    edge_weight = np.array([1.0, 1.0], dtype=np.float32)
    A = build_normalized_adjacency(3, edge_index, edge_weight)
    row_sums = A.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(3), atol=1e-5)


def test_gnn_forward_shape():
    torch.manual_seed(0)
    x = torch.randn(5, 4)
    edge_index = np.array([[0, 1, 2], [1, 2, 3]])
    edge_weight = np.ones(3, dtype=np.float32)
    A = build_normalized_adjacency(5, edge_index, edge_weight)
    model = ShillingGNN(in_dim=4, hidden_dim=8)
    logits = model(x, A)
    assert logits.shape == (5,)


def test_train_gnn_reduces_loss():
    torch.manual_seed(0)
    n = 20
    x = torch.randn(n, 4)
    y = torch.tensor([1.0] * 5 + [0.0] * 15)
    edge_index = np.array([[i for i in range(4)], [i + 1 for i in range(4)]])
    edge_weight = np.ones(4, dtype=np.float32)
    A = build_normalized_adjacency(n, edge_index, edge_weight)
    train_mask = torch.ones(n, dtype=torch.bool)
    model = ShillingGNN(in_dim=4, hidden_dim=8)
    losses = train_gnn(model, x, A, y, train_mask, epochs=50, lr=0.05)
    assert losses[-1] < losses[0]
