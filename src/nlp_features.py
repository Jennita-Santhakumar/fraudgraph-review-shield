"""
NLP feature extraction for shilling-attack detection.

Produces, per review:
  - sentiment (VADER compound/pos/neg/neu)
  - text length / word count
  - readability (Flesch reading ease via textstat)
  - exclamation/uppercase ratio (a classic "hype" signal in shill reviews)
  - max TF-IDF cosine similarity to any OTHER review by a different reviewer
    on the same product (catches templated/duplicated shill text)

And, per reviewer (aggregated from their reviews):
  - review burstiness: min inter-review-time gap and fraction of reviews
    posted within a tight rolling window (bursty posting is a hallmark of
    coordinated shilling)
  - average rating, rating variance
  - average max text similarity (do they copy-paste more than others?)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def _sentiment_features(text: str) -> dict:
    scores = _analyzer.polarity_scores(text)
    return {
        "sent_compound": scores["compound"],
        "sent_pos": scores["pos"],
        "sent_neg": scores["neg"],
        "sent_neu": scores["neu"],
    }


def _surface_features(text: str) -> dict:
    n_chars = max(len(text), 1)
    n_words = max(len(text.split()), 1)
    n_upper = sum(1 for c in text if c.isupper())
    n_excl = text.count("!")
    try:
        readability = textstat.flesch_reading_ease(text)
    except Exception:
        readability = 0.0
    return {
        "n_words": n_words,
        "n_chars": n_chars,
        "upper_ratio": n_upper / n_chars,
        "excl_ratio": n_excl / n_words,
        "readability": readability,
    }


def compute_text_similarity(df: pd.DataFrame) -> pd.Series:
    """For each review, compute the max TF-IDF cosine similarity to any OTHER
    review on the SAME product written by a DIFFERENT reviewer. High values
    indicate templated / copy-pasted shill text reused across colluding
    accounts targeting the same product.
    """
    max_sim = pd.Series(0.0, index=df.index)
    for product_id, group in df.groupby("product_id"):
        if len(group) < 2:
            continue
        vect = TfidfVectorizer(stop_words="english", min_df=1)
        try:
            tfidf = vect.fit_transform(group["text"].tolist())
        except ValueError:
            continue
        sims = cosine_similarity(tfidf)
        np.fill_diagonal(sims, 0.0)
        # zero out similarity to reviews by the SAME reviewer (not a collusion signal)
        reviewers = group["reviewer_id"].to_numpy()
        same_reviewer_mask = reviewers[:, None] == reviewers[None, :]
        sims[same_reviewer_mask] = 0.0
        row_max = sims.max(axis=1)
        max_sim.loc[group.index] = row_max
    return max_sim


def extract_review_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-review NLP feature columns to a copy of `df`."""
    df = df.copy()
    sent_feats = df["text"].apply(_sentiment_features).apply(pd.Series)
    surf_feats = df["text"].apply(_surface_features).apply(pd.Series)
    df = pd.concat([df, sent_feats, surf_feats], axis=1)
    df["max_text_similarity"] = compute_text_similarity(df)
    return df


def extract_reviewer_features(review_feats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-review features into per-reviewer NLP features, including
    temporal burstiness.
    """
    rows = []
    for reviewer_id, g in review_feats.groupby("reviewer_id"):
        g = g.sort_values("timestamp")
        n_reviews = len(g)
        if n_reviews > 1:
            gaps_hours = g["timestamp"].diff().dt.total_seconds().dropna() / 3600.0
            min_gap = gaps_hours.min()
            median_gap = gaps_hours.median()
            # fraction of reviews posted within 48h of the previous one -> burst signal
            burst_frac = (gaps_hours < 48).mean()
        else:
            min_gap = np.nan
            median_gap = np.nan
            burst_frac = 0.0

        rows.append(dict(
            reviewer_id=reviewer_id,
            n_reviews=n_reviews,
            avg_rating=g["rating"].mean(),
            rating_std=g["rating"].std(ddof=0) if n_reviews > 1 else 0.0,
            avg_sent_compound=g["sent_compound"].mean(),
            avg_excl_ratio=g["excl_ratio"].mean(),
            avg_upper_ratio=g["upper_ratio"].mean(),
            avg_readability=g["readability"].mean(),
            avg_max_text_similarity=g["max_text_similarity"].mean(),
            min_inter_review_gap_h=min_gap if not np.isnan(min_gap) else 9999.0,
            median_inter_review_gap_h=median_gap if not np.isnan(median_gap) else 9999.0,
            burst_fraction=burst_frac,
            n_unique_products=g["product_id"].nunique(),
            reviewer_is_shill=g["reviewer_is_shill"].max(),
        ))
    return pd.DataFrame(rows).set_index("reviewer_id")
