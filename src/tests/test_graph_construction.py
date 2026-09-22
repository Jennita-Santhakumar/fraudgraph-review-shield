import networkx as nx

from src.graph_construction import (build_bipartite_graph, build_reviewer_graph,
                                     graph_structural_features)
from src.nlp_features import extract_review_features
from src.synthetic_data import GeneratorConfig, generate_dataset


def _small_review_feats():
    df = generate_dataset(GeneratorConfig(
        n_genuine_reviewers=20, n_shill_reviewers=10, clique_size=5,
        n_products=8, n_target_products=2, seed=9,
    ))
    return extract_review_features(df)


def test_build_bipartite_graph_structure():
    feats = _small_review_feats()
    B = build_bipartite_graph(feats)
    assert isinstance(B, nx.Graph)
    user_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 0]
    prod_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == 1]
    assert len(user_nodes) == feats["reviewer_id"].nunique()
    assert len(prod_nodes) == feats["product_id"].nunique()


def test_build_reviewer_graph_no_self_loops():
    feats = _small_review_feats()
    G = build_reviewer_graph(feats)
    assert nx.number_of_selfloops(G) == 0
    assert G.number_of_nodes() == feats["reviewer_id"].nunique()


def test_build_reviewer_graph_edge_weights_positive():
    feats = _small_review_feats()
    G = build_reviewer_graph(feats)
    for _, _, data in G.edges(data=True):
        assert data["weight"] > 0
        assert data["shared_products"] >= 1


def test_shill_clique_more_connected_than_average():
    feats = _small_review_feats()
    G = build_reviewer_graph(feats)
    shill_reviewers = set(feats.loc[feats.reviewer_is_shill == 1, "reviewer_id"])
    shill_reviewers = [r for r in shill_reviewers if r in G]
    genuine_reviewers = [r for r in G.nodes() if r not in shill_reviewers]
    if shill_reviewers and genuine_reviewers:
        avg_shill_degree = sum(dict(G.degree(shill_reviewers)).values()) / len(shill_reviewers)
        avg_genuine_degree = sum(dict(G.degree(genuine_reviewers)).values()) / len(genuine_reviewers)
        assert avg_shill_degree >= avg_genuine_degree


def test_graph_structural_features_shape():
    feats = _small_review_feats()
    G = build_reviewer_graph(feats)
    struct = graph_structural_features(G)
    assert set(["degree", "weighted_degree", "clustering", "avg_edge_weight"]).issubset(struct.columns)
    assert len(struct) == G.number_of_nodes()
