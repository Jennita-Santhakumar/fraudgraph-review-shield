"""
Synthetic reviewer-product-review dataset generator with injected shilling attacks.

WHY SYNTHETIC: this repository originally shipped only two Amazon-scraping
notebooks and no labeled dataset of genuine vs. shilled reviews. Public
labeled shilling/fake-review datasets (e.g. the Yelp/Amazon fraud corpora
used in academic papers) are not reliably downloadable from this sandboxed
environment (no outbound access to most third-party hosts, and many require
manual application / licensing). Rather than fabricate a "real" data source,
this module builds a documented, parametrized SYNTHETIC generator that
encodes well-known shilling-attack signatures from the literature:

  1. Burstiness: a clique of colluding reviewer accounts posts many reviews
     for the SAME target product within a short time window.
  2. Templated / duplicated text: colluding accounts reuse near-identical
     review text (small word substitutions) instead of writing organically.
  3. Rating injection: colluding accounts give extreme (5-star) ratings to
     push a product, regardless of the product's "true" quality level.
  4. Clique co-review structure: colluding accounts review a highly
     overlapping set of products together (graph signal), unlike genuine
     reviewers whose product sets are close to random.

Every row is labeled (`is_shill`) with ground truth so we can measure real
precision/recall/F1/AUC for the NLP-only, graph-only, and hybrid models.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

GENUINE_TEMPLATES = [
    "I bought this {noun} last month and honestly it works fine, though the {part} "
    "feels a bit flimsy after regular use.",
    "Decent {noun} for the price. Shipping took a while but the product itself is okay, "
    "not amazing.",
    "Mixed feelings about this one. The {part} broke within two weeks, but customer "
    "support replaced it quickly.",
    "Solid choice if you don't need anything fancy. I've used it daily for a few weeks "
    "and it's holding up.",
    "Not what I expected from the photos, the color is slightly different, but "
    "functionally it does the job.",
    "Works as described. Nothing special but nothing wrong either. Would consider "
    "buying again if it breaks.",
    "The {part} on mine arrived slightly scratched, otherwise a reasonable {noun}.",
    "Took a star off because the instructions were confusing, but once set up it runs fine.",
    "Good value overall. My only complaint is the {part} feels cheap compared to competitors.",
    "I compared this against two other brands before buying, this one had the best "
    "reviews for durability so far it's living up to that.",
]

SHILL_TEMPLATES = [
    "AMAZING product!!! Best {noun} I have EVER purchased, 5 stars, buy it now!!!",
    "Absolutely incredible {noun}, changed my life, everyone needs this, 5/5 stars!!!",
    "Best purchase ever!!! This {noun} is perfect in every way, highly recommend to all!!!",
    "Five stars!!! Amazing quality {noun}, works perfectly, exceeded all expectations!!!",
    "Wow just wow, this {noun} is a game changer, perfect, 5 stars, must buy immediately!!!",
]

NOUNS = ["gadget", "device", "product", "item", "tool", "accessory"]
PARTS = ["cable", "casing", "button", "strap", "battery", "screen"]


@dataclass
class GeneratorConfig:
    n_genuine_reviewers: int = 260
    n_shill_reviewers: int = 40          # organized into cliques
    clique_size: int = 5                 # reviewers per shilling clique
    n_products: int = 60
    n_target_products: int = 6           # products the shill cliques attack
    genuine_reviews_per_reviewer: tuple = (1, 6)
    reviews_per_shill_in_burst: tuple = (3, 7)
    burst_window_hours: int = 36
    seed: int = 42


def _rand_text(templates: list[str], rng: random.Random) -> str:
    t = rng.choice(templates)
    return t.format(noun=rng.choice(NOUNS), part=rng.choice(PARTS))


def generate_dataset(cfg: GeneratorConfig = GeneratorConfig()) -> pd.DataFrame:
    """Generate a synthetic reviews dataframe with ground-truth shill labels.

    Returns a DataFrame with columns:
        review_id, reviewer_id, product_id, rating, text, timestamp,
        is_shill (ground truth, per-REVIEW), reviewer_is_shill (ground truth per-reviewer)
    """
    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)

    base_time = datetime(2024, 1, 1)
    products = [f"P{i:04d}" for i in range(cfg.n_products)]
    target_products = rng.sample(products, cfg.n_target_products)

    rows = []
    review_counter = 0

    # ---- genuine reviewers: organic, spread-out behaviour ----
    for r in range(cfg.n_genuine_reviewers):
        reviewer_id = f"U{r:05d}"
        n_reviews = rng.randint(*cfg.genuine_reviews_per_reviewer)
        chosen_products = rng.sample(products, min(n_reviews, len(products)))
        for prod in chosen_products:
            ts = base_time + timedelta(days=int(np_rng.uniform(0, 365)),
                                        hours=int(np_rng.uniform(0, 24)))
            rating = int(np.clip(np_rng.normal(3.6, 1.1), 1, 5))
            rows.append(dict(
                review_id=f"R{review_counter:06d}",
                reviewer_id=reviewer_id,
                product_id=prod,
                rating=rating,
                text=_rand_text(GENUINE_TEMPLATES, rng),
                timestamp=ts,
                is_shill=0,
                reviewer_is_shill=0,
            ))
            review_counter += 1

    # ---- hard negatives: genuine "launch day" buyers who coincidentally
    # cluster in time on a popular product, without colluding or reusing
    # text -- this keeps the graph-only signal from being perfectly clean ----
    popular_products = rng.sample(products, 4)
    launch_time = base_time + timedelta(days=200)
    for i in range(18):
        reviewer_id = f"L{i:05d}"
        prod = rng.choice(popular_products)
        ts = launch_time + timedelta(hours=rng.uniform(0, 96))
        rows.append(dict(
            review_id=f"R{review_counter:06d}",
            reviewer_id=reviewer_id,
            product_id=prod,
            rating=int(np.clip(np_rng.normal(4.2, 0.9), 1, 5)),
            text=_rand_text(GENUINE_TEMPLATES, rng),
            timestamp=ts,
            is_shill=0,
            reviewer_is_shill=0,
        ))
        review_counter += 1

    # ---- shilling cliques: bursty, templated, coordinated ----
    n_cliques = max(1, cfg.n_shill_reviewers // cfg.clique_size)
    shill_reviewer_counter = 0
    for c in range(n_cliques):
        clique_members = []
        for _ in range(cfg.clique_size):
            clique_members.append(f"S{shill_reviewer_counter:05d}")
            shill_reviewer_counter += 1
        target = target_products[c % len(target_products)]
        # each clique attacks its target in a short burst, then adds a
        # little "cover" noise on random other products to look less robotic
        burst_start = base_time + timedelta(days=int(rng.uniform(30, 330)))
        for member in clique_members:
            n_burst = rng.randint(*cfg.reviews_per_shill_in_burst)
            for _ in range(n_burst):
                ts = burst_start + timedelta(hours=rng.uniform(0, cfg.burst_window_hours * 2.5))
                # not every shill review is a blatant hype template / perfect
                # 5-star -- some are moderately disguised, to keep the task
                # non-trivial and avoid a degenerate 100%-separable dataset
                if rng.random() < 0.7:
                    text = _rand_text(SHILL_TEMPLATES, rng)
                    rating = 5
                else:
                    text = _rand_text(GENUINE_TEMPLATES, rng)
                    rating = rng.choice([4, 5])
                rows.append(dict(
                    review_id=f"R{review_counter:06d}",
                    reviewer_id=member,
                    product_id=target,
                    rating=rating,
                    text=text,
                    timestamp=ts,
                    is_shill=1,
                    reviewer_is_shill=1,
                ))
                review_counter += 1
            # cover noise: 0-2 genuine-looking reviews on unrelated products
            for _ in range(rng.randint(0, 2)):
                prod = rng.choice(products)
                ts = base_time + timedelta(days=int(np_rng.uniform(0, 365)))
                rows.append(dict(
                    review_id=f"R{review_counter:06d}",
                    reviewer_id=member,
                    product_id=prod,
                    rating=int(np.clip(np_rng.normal(4.0, 1.0), 1, 5)),
                    text=_rand_text(GENUINE_TEMPLATES, rng),
                    timestamp=ts,
                    is_shill=0,
                    reviewer_is_shill=1,
                ))
                review_counter += 1

    # ---- sleeper shills: a few clique members space their burst reviews out
    # over days instead of hours, and skip the hype template, so they don't
    # trip the obvious burstiness/text-similarity signals as hard -- these
    # are the "hard positives" that should separate the weaker baselines
    # from the hybrid model ----
    n_sleepers = max(1, cfg.n_shill_reviewers // 10)
    for i in range(n_sleepers):
        member = f"SL{i:05d}"
        target = rng.choice(target_products)
        start = base_time + timedelta(days=int(rng.uniform(30, 330)))
        n_reviews = rng.randint(3, 5)
        for k in range(n_reviews):
            ts = start + timedelta(days=rng.uniform(2, 9) * (k + 1))
            rows.append(dict(
                review_id=f"R{review_counter:06d}",
                reviewer_id=member,
                product_id=target if k == 0 else rng.choice(products),
                rating=rng.choice([4, 5]),
                text=_rand_text(GENUINE_TEMPLATES, rng),
                timestamp=ts,
                is_shill=1 if k == 0 else 0,
                reviewer_is_shill=1,
            ))
            review_counter += 1

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/synthetic_reviews.csv", index=False)
    print(f"Generated {len(df)} reviews from {df.reviewer_id.nunique()} reviewers "
          f"over {df.product_id.nunique()} products.")
    print(f"Reviewer-level shill rate: {df.groupby('reviewer_id').reviewer_is_shill.max().mean():.2%}")
    print(f"Review-level shill rate: {df.is_shill.mean():.2%}")
