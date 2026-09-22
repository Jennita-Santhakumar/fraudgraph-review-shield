"""
End-to-end pipeline: synthetic data -> NLP features -> graph construction ->
GNN training -> fusion & evaluation of three models:

  1. NLP-only        : Logistic Regression on NLP/behavioural features alone.
  2. Graph-only       : Logistic Regression on classic graph statistics alone
                        (degree, weighted degree, clustering, avg edge weight).
  3. Hybrid (NLP+GNN) : the from-scratch GraphSAGE GNN consuming NLP features
                        AS NODE FEATURES propagated over the reviewer-reviewer
                        collusion graph -- this is the "hybrid" the repo title
                        promises.

Run with:  python -m src.pipeline
Produces:  outputs/metrics.json, outputs/metrics.md, prints a comparison table.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.gnn_model import ShillingGNN, build_normalized_adjacency, train_gnn
from src.graph_construction import build_reviewer_graph, graph_structural_features
from src.nlp_features import extract_review_features, extract_reviewer_features
from src.synthetic_data import GeneratorConfig, generate_dataset

RANDOM_STATE = 42
NLP_FEATURE_COLS = [
    "n_reviews", "avg_rating", "rating_std", "avg_sent_compound",
    "avg_excl_ratio", "avg_upper_ratio", "avg_readability",
    "avg_max_text_similarity", "min_inter_review_gap_h",
    "median_inter_review_gap_h", "burst_fraction", "n_unique_products",
]
GRAPH_FEATURE_COLS = ["degree", "weighted_degree", "clustering", "avg_edge_weight"]


def evaluate(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return dict(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        auc=roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        accuracy=accuracy_score(y_true, y_pred),
    )


def run_pipeline(out_dir: str = "outputs", data_dir: str = "data") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # ---- 1. Data ----
    df = generate_dataset(GeneratorConfig())
    df.to_csv(os.path.join(data_dir, "synthetic_reviews.csv"), index=False)

    # ---- 2. NLP features ----
    review_feats = extract_review_features(df)
    review_feats.to_csv(os.path.join(data_dir, "review_features.csv"), index=False)
    reviewer_feats = extract_reviewer_features(review_feats)  # indexed by reviewer_id

    # ---- 3. Graph construction ----
    G = build_reviewer_graph(review_feats)
    graph_feats = graph_structural_features(G)

    node_ids = list(reviewer_feats.index)
    node_index = {rid: i for i, rid in enumerate(node_ids)}
    y_all = reviewer_feats.loc[node_ids, "reviewer_is_shill"].to_numpy().astype(np.float32)

    graph_feats = graph_feats.reindex(node_ids).fillna(0.0)

    # ---- 4. Train/test split (reviewer-level, stratified) ----
    idx_all = np.arange(len(node_ids))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=0.3, random_state=RANDOM_STATE, stratify=y_all
    )
    train_mask = np.zeros(len(node_ids), dtype=bool)
    train_mask[idx_train] = True

    results = {}

    # ---- Model 1: NLP-only ----
    X_nlp = reviewer_feats.loc[node_ids, NLP_FEATURE_COLS].fillna(0.0).to_numpy()
    scaler_nlp = StandardScaler().fit(X_nlp[idx_train])
    X_nlp_scaled = scaler_nlp.transform(X_nlp)
    clf_nlp = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    clf_nlp.fit(X_nlp_scaled[idx_train], y_all[idx_train])
    prob_nlp = clf_nlp.predict_proba(X_nlp_scaled[idx_test])[:, 1]
    results["nlp_only"] = evaluate(y_all[idx_test], prob_nlp)

    # ---- Model 2: Graph-only ----
    X_graph = graph_feats.to_numpy()
    scaler_graph = StandardScaler().fit(X_graph[idx_train])
    X_graph_scaled = scaler_graph.transform(X_graph)
    clf_graph = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    clf_graph.fit(X_graph_scaled[idx_train], y_all[idx_train])
    prob_graph = clf_graph.predict_proba(X_graph_scaled[idx_test])[:, 1]
    results["graph_only"] = evaluate(y_all[idx_test], prob_graph)

    # ---- Model 3: Hybrid NLP+GNN ----
    edge_list = list(G.edges(data=True))
    if len(edge_list) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_weight = np.zeros((0,), dtype=np.float32)
    else:
        mapped = np.array([[node_index[u], node_index[v]] for u, v, _ in edge_list], dtype=np.int64)
        edge_index = mapped.T
        edge_weight = np.array([d["weight"] for _, _, d in edge_list], dtype=np.float32)

    A_norm = build_normalized_adjacency(len(node_ids), edge_index, edge_weight)
    x_tensor = torch.tensor(X_nlp_scaled, dtype=torch.float32)  # NLP features as node features
    y_tensor = torch.tensor(y_all, dtype=torch.float32)
    train_mask_t = torch.tensor(train_mask)

    torch.manual_seed(RANDOM_STATE)
    model = ShillingGNN(in_dim=x_tensor.shape[1], hidden_dim=32, dropout=0.3)
    losses = train_gnn(model, x_tensor, A_norm, y_tensor, train_mask_t, epochs=200, lr=0.01)

    model.eval()
    with torch.no_grad():
        logits = model(x_tensor, A_norm)
        prob_gnn_all = torch.sigmoid(logits).numpy()
    results["hybrid_nlp_gnn"] = evaluate(y_all[idx_test], prob_gnn_all[idx_test])
    results["hybrid_nlp_gnn"]["final_train_loss"] = losses[-1]

    # ---- Save outputs ----
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    md_lines = ["| Model | Precision | Recall | F1 | AUC | Accuracy |",
                "|---|---|---|---|---|---|"]
    for name, m in results.items():
        md_lines.append(
            f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
            f"| {m['auc']:.3f} | {m['accuracy']:.3f} |"
        )
    with open(os.path.join(out_dir, "metrics.md"), "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Dataset: {len(df)} reviews, {len(node_ids)} reviewers "
          f"({y_all.sum():.0f} shill / {len(y_all)} total, "
          f"{y_all.mean():.1%} positive rate)")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print()
    print("\n".join(md_lines))
    return results


if __name__ == "__main__":
    run_pipeline()
