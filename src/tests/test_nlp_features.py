import pandas as pd

from src.nlp_features import (compute_text_similarity, extract_review_features,
                               extract_reviewer_features)
from src.synthetic_data import GeneratorConfig, generate_dataset


def _small_df():
    return generate_dataset(GeneratorConfig(
        n_genuine_reviewers=20, n_shill_reviewers=10, clique_size=5,
        n_products=8, n_target_products=2, seed=5,
    ))


def test_extract_review_features_adds_expected_columns():
    df = _small_df()
    feats = extract_review_features(df)
    for col in ["sent_compound", "n_words", "readability", "max_text_similarity"]:
        assert col in feats.columns
    assert len(feats) == len(df)


def test_sentiment_range():
    df = _small_df()
    feats = extract_review_features(df)
    assert feats["sent_compound"].between(-1.0, 1.0).all()


def test_text_similarity_detects_duplicated_shill_text():
    df = _small_df()
    feats = extract_review_features(df)
    # shill reviews target the same product with reused hype templates,
    # so their average max-similarity should exceed genuine reviews'
    shill_sim = feats.loc[feats.is_shill == 1, "max_text_similarity"].mean()
    genuine_sim = feats.loc[feats.is_shill == 0, "max_text_similarity"].mean()
    assert shill_sim > genuine_sim


def test_extract_reviewer_features_aggregation():
    df = _small_df()
    review_feats = extract_review_features(df)
    reviewer_feats = extract_reviewer_features(review_feats)
    assert reviewer_feats.index.name == "reviewer_id"
    assert set(reviewer_feats["reviewer_is_shill"].unique()).issubset({0, 1})
    assert "burst_fraction" in reviewer_feats.columns
    assert (reviewer_feats["n_reviews"] > 0).all()


def test_compute_text_similarity_empty_safe():
    df = pd.DataFrame({
        "reviewer_id": ["a"], "product_id": ["p1"], "text": ["hello world"],
    })
    sims = compute_text_similarity(df)
    assert sims.iloc[0] == 0.0
