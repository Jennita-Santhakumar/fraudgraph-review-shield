"""
Graph construction for shilling-attack detection.

Builds a reviewer-reviewer graph (projected from the reviewer-product
bipartite review graph) where an edge between two reviewers is created when
they co-review the same product, weighted by:
  - number of shared products
  - temporal proximity of their reviews on shared products (close-in-time
    co-reviews are a strong collusion signal)
  - text similarity of their reviews on shared products

This reviewer-reviewer graph is what we feed into the GNN: each reviewer is
a node, node features come from `nlp_features.extract_reviewer_features`,
and edges encode the co-review collusion signal.
"""
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_bipartite_graph(review_feats: pd.DataFrame) -> nx.Graph:
    """Reviewer-product bipartite graph, mostly useful for inspection/export."""
    B = nx.Graph()
    for _, row in review_feats.iterrows():
        B.add_node(f"user::{row.reviewer_id}", bipartite=0)
        B.add_node(f"prod::{row.product_id}", bipartite=1)
        B.add_edge(f"user::{row.reviewer_id}", f"prod::{row.product_id}",
                   rating=row.rating, timestamp=row.timestamp)
    return B


def build_reviewer_graph(review_feats: pd.DataFrame,
                          min_shared_products: int = 1,
                          time_decay_hours: float = 72.0) -> nx.Graph:
    """Project the bipartite reviewer-product graph onto a reviewer-reviewer
    graph, connecting reviewers who reviewed the same product(s).

    Edge weight = sum over shared products of:
        text_sim * exp(-|dt| / time_decay_hours)
    where dt is the time gap (hours) between the two reviewers' reviews of
    that shared product. This rewards pairs who reviewed the SAME product
    with SIMILAR text at CLOSE times -- exactly the shilling-clique pattern
    injected by synthetic_data.py.
    """
    G = nx.Graph()
    for reviewer_id, g in review_feats.groupby("reviewer_id"):
        G.add_node(reviewer_id)

    for product_id, group in review_feats.groupby("product_id"):
        if len(group) < 2:
            continue
        group = group.reset_index(drop=True)
        vect = TfidfVectorizer(stop_words="english", min_df=1)
        try:
            tfidf = vect.fit_transform(group["text"].tolist())
            sims = cosine_similarity(tfidf)
        except ValueError:
            sims = np.zeros((len(group), len(group)))

        for i, j in itertools.combinations(range(len(group)), 2):
            r_i, r_j = group.loc[i, "reviewer_id"], group.loc[j, "reviewer_id"]
            if r_i == r_j:
                continue
            dt_hours = abs((group.loc[i, "timestamp"] - group.loc[j, "timestamp"]).total_seconds()) / 3600.0
            time_weight = np.exp(-dt_hours / time_decay_hours)
            text_sim = sims[i, j]
            edge_w = (0.3 + text_sim) * time_weight  # base weight so co-review alone still counts
            if G.has_edge(r_i, r_j):
                G[r_i][r_j]["weight"] += edge_w
                G[r_i][r_j]["shared_products"] += 1
            else:
                G.add_edge(r_i, r_j, weight=edge_w, shared_products=1)

    # drop edges that don't meet the minimum shared-products threshold
    to_remove = [(u, v) for u, v, d in G.edges(data=True)
                 if d["shared_products"] < min_shared_products]
    G.remove_edges_from(to_remove)
    return G


def graph_structural_features(G: nx.Graph) -> pd.DataFrame:
    """Simple graph-only features per node: degree, weighted degree, clustering
    coefficient, and average edge weight. Used to build the "graph-only"
    baseline model (no GNN, no NLP -- just classic graph statistics).
    """
    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    clustering = nx.clustering(G, weight="weight")
    avg_weight = {
        n: (weighted_degree[n] / degree[n]) if degree[n] > 0 else 0.0
        for n in G.nodes()
    }
    df = pd.DataFrame({
        "degree": degree,
        "weighted_degree": weighted_degree,
        "clustering": clustering,
        "avg_edge_weight": avg_weight,
    })
    df.index.name = "reviewer_id"
    return df
