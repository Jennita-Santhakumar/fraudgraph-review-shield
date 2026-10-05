# Hybrid Detection of Shilling Attacks Using NLP and Graph Neural Networks

> **Maintained by [Jennita S](https://github.com/Jennita-Santhakumar)** · [LinkedIn](https://linkedin.com/in/jennitas) · jennitasanthakumar0@gmail.com

Detects fake/"shilling" product reviews — coordinated fake-review campaigns
where a clique of accounts pumps a product's rating — by combining **NLP
features** (sentiment, readability, text-duplication) with a **Graph Neural
Network** over a reviewer-reviewer collusion graph.

## Honesty about the data

There is no bundled labeled real-world shilling dataset in this repo, and
public labeled fake-review corpora (Yelp/Amazon fraud datasets used in
academic papers) were not reliably downloadable from the environment this
was built in (no outbound access to most third-party hosts, and several
require manual licensing/application). Rather than pretend otherwise, this
project ships a **documented synthetic data generator**
(`src/synthetic_data.py`) that builds a reviewer-product-review graph and
injects well-known shilling signatures from the literature:

- **Burstiness** — a clique of colluding accounts reviews the same target
  product within a short time window.
- **Templated / duplicated text** — colluding accounts reuse near-identical
  hype phrasing instead of writing organically.
- **Rating injection** — colluding accounts give extreme (4-5 star) ratings
  regardless of the product's baseline quality.
- **Clique co-review structure** — colluding accounts share an unusually
  overlapping set of reviewed products.
- **Hard cases included on purpose**: some "sleeper" shills space their
  reviews out over days and skip the obvious hype template, and some
  genuine reviewers coincidentally cluster around a product's launch window
  ("hard negatives"). This keeps the task from being trivially separable
  and lets the NLP-only, graph-only, and hybrid models be meaningfully
  compared instead of all hitting 100%.

If/when a real labeled dataset becomes available, only `synthetic_data.py`
needs to be swapped for a loader that produces a dataframe with the same
schema (`review_id, reviewer_id, product_id, rating, text, timestamp,
is_shill, reviewer_is_shill`) — the feature extraction, graph construction,
GNN, and evaluation code are dataset-agnostic.

The two original notebooks (`amazon_scrape.ipynb`, `testing.ipynb`) are
**kept as-is** as a legitimate real-world data-collection option (Selenium
scraping of Amazon reviews) but are entirely separate from, and not
required by, the modeling pipeline below.

## What's actually implemented

### 1. NLP feature pipeline (`src/nlp_features.py`)
Per review:
- Sentiment (VADER: compound/pos/neg/neu)
- Surface/hype features: word count, uppercase ratio, exclamation ratio
- Readability (Flesch reading ease, via `textstat`)
- **Max TF-IDF cosine similarity** to any other review on the same product
  by a *different* reviewer — catches templated/copy-pasted shill text

Per reviewer (aggregated):
- Average rating and rating variance
- Average sentiment/hype features
- **Burstiness**: min/median inter-review time gap, fraction of reviews
  posted within 48h of the previous one
- Number of unique products reviewed

### 2. Graph construction (`src/graph_construction.py`)
Projects the reviewer-product bipartite review graph onto a
**reviewer-reviewer graph**: two reviewers are connected if they reviewed
the same product, with edge weight combining:
- text similarity of their reviews on the shared product, and
- an exponential time-decay on how close together in time the two reviews
  were posted

This directly encodes the "clique reviews the same product in a tight
window with similar text" shilling signature into graph structure. Classic
graph statistics (degree, weighted degree, clustering coefficient, average
edge weight) are also computed as a graph-only baseline.

### 3. GNN model (`src/gnn_model.py`)
A real, from-scratch **2-layer GraphSAGE-style** message-passing network in
plain PyTorch (no PyTorch Geometric dependency, so it installs anywhere):
each layer mean-aggregates a node's (weighted) neighbors, concatenates with
the node's own previous-layer representation, and applies a learned linear
transform + ReLU. This lets a reviewer's predicted "shilling" probability
depend on both their own NLP features *and* the behavior of reviewers they
are graph-connected to — the actual graph neural network the repo's title
promises. Trained with class-weighted binary cross-entropy (shills are a
minority class).

### 4. Fusion & evaluation (`src/pipeline.py`)
Trains and evaluates three models on the same held-out reviewer split:

| Model | Input |
|---|---|
| `nlp_only` | Logistic Regression on NLP/behavioural features alone |
| `graph_only` | Logistic Regression on graph statistics alone (no NLP, no learned GNN) |
| `hybrid_nlp_gnn` | GraphSAGE GNN — NLP features as node features, propagated over the collusion graph |

## Measured results (synthetic dataset, 322 reviewers, 30% held-out test split, seed=42)

| Model | Precision | Recall | F1 | AUC | Accuracy |
|---|---|---|---|---|---|
| nlp_only | 0.923 | 0.923 | 0.923 | 0.998 | 0.979 |
| graph_only | 1.000 | 0.846 | 0.917 | 0.967 | 0.979 |
| hybrid_nlp_gnn | 1.000 | 0.923 | 0.960 | 0.997 | 0.990 |

The hybrid NLP+GNN model gets the best F1 and accuracy, catching the
"sleeper" shill accounts (spaced-out, non-templated) that the graph-only
model misses on recall, while matching graph-only's precision. Regenerate
these numbers yourself with `python -m src.pipeline` (fully deterministic,
seeded) — results are written to `outputs/metrics.json` / `metrics.md`.

## How to run

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

# generate synthetic data, extract features, build the graph, train the GNN,
# evaluate all three models, and write outputs/metrics.{json,md}
python -m src.pipeline

# run the test suite
pytest src/tests -q
```

## Repository layout

```
src/
  synthetic_data.py       # documented synthetic reviewer/product/review generator
  nlp_features.py         # per-review and per-reviewer NLP feature extraction
  graph_construction.py   # reviewer-reviewer collusion graph + graph-only baseline
  gnn_model.py             # from-scratch GraphSAGE-style message-passing GNN (PyTorch)
  pipeline.py              # end-to-end run: data -> features -> graph -> GNN -> evaluation
  tests/                   # pytest coverage for the above
amazon_scrape.ipynb        # (legacy) Selenium Amazon review scraper — real-world data option
testing.ipynb              # (legacy) scraped-data testing/exploration notebook
outputs/                   # metrics.json / metrics.md written by pipeline.py
data/                      # synthetic_reviews.csv / review_features.csv written by pipeline.py
```

## Honest limitations (things to say out loud in an interview)

- **The data is synthetic.** The injected shilling patterns are based on
  real literature (burst timing, template reuse, clique co-review), but
  synthetic data is inherently easier to separate than messy real-world
  reviews — real fraud rings adapt to evade exactly these signals. Treat
  the reported metrics as a proof that the *pipeline* works end-to-end, not
  as a claim about real-world detection performance.
- **The graph is a dense-adjacency implementation.** `build_normalized_adjacency`
  materializes an NxN matrix, which is fine at a few hundred/thousand nodes
  but would need a sparse-tensor or mini-batch/neighbor-sampling
  implementation (e.g. real PyTorch Geometric `NeighborLoader`) to scale to
  a production-size review corpus.
- **No PyTorch Geometric.** The GNN is a hand-rolled GraphSAGE layer to keep
  the environment lightweight and installable anywhere; a production system
  would likely use PyG or DGL for more GNN architectures (GAT, GIN),
  better sparse-graph performance, and mini-batching.
- **Single random seed / single split reported.** The metrics table above
  is one seeded run for reproducibility; a rigorous evaluation would report
  mean +/- std over multiple seeds/splits and a proper train/val/test split
  with hyperparameter tuning on the validation set only.
- **The two legacy notebooks are a separate concern.** They scrape Amazon
  via Selenium and were not modified beyond being left in place as one
  legitimate way to eventually source real review data (subject to
  Amazon's Terms of Service — see the scraping disclaimer below).

---

## Legacy: Amazon Product Scraper & Testing Notebooks

The two notebooks below predate the modeling pipeline above and are kept as
a real-world data-collection option; they are not required to run
`src/pipeline.py`.

- `amazon_scrape.ipynb` — scrapes product name, price, rating, number of
  reviews, and availability from Amazon product listings via Selenium,
  saving to CSV.
- `testing.ipynb` — tests/analyzes the scraped data.

Requirements: `selenium`, `pandas`, `beautifulsoup4`, `webdriver_manager`.

**Disclaimer**: scraping Amazon may violate their Terms of Service. This is
for educational purposes only — use responsibly and at your own risk.

## Credits

Developed by Kabilesh Rajaselvan with contributions from Jennita S.
