# Intelligent Search Engine

Course project comparing keyword-based (TF-IDF) and semantic (MPNet) document retrieval on a news headline dataset (~209k articles).

## Features

- **Baseline TF-IDF** — Porter-stemmed token matching with cosine similarity
- **Advanced Semantic Search** — MPNet sentence embeddings via `sentence-transformers` for meaning-aware retrieval
- **Evaluation** — Precision@k with manually labeled relevance across 8 diverse queries
- **Streamlit GUI** — Side-by-side Baseline vs Advanced comparison with an evaluation chart tab

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run evaluation (generates results/evaluation_chart.png)
python scripts/run_evaluation.py

# 3. Launch the Streamlit GUI
streamlit run app.py
```

## Project Structure

```
├── data/                         # Datasets and precomputed caches
│   ├── tokenized_dataset.csv
│   ├── document_embeddings_mpnet.npz
│   └── document_embeddings_glove.npz
├── results/                      # Evaluation charts and outputs
├── scripts/                      # Runnable entry points
│   ├── baseline.py               # Standalone TF-IDF CLI
│   ├── run_evaluation.py         # Baseline vs Advanced evaluation
│   ├── run_semantic_search.py    # Semantic search CLI (GloVe or Transformer)
│   └── build_mpnet_cache.py      # One-time MPNet embedding cache builder
├── src/                          # Reusable library modules
│   ├── text_preprocessing.py     # Tokenization, stemming, stopword removal
│   └── advanced_model.py         # GloVe & Transformer semantic search engines
├── app.py                        # Streamlit GUI
├── requirements.txt
└── AGENTS.md                     # Detailed agent/developer reference
```

## Dataset

- **Source**: HuffPost News Category Dataset (news headlines)
- **Size**: ~209,000 headlines
- **Columns**: `text` (raw headline), `cleaned_text` (preprocessed stemmed tokens)

## Precomputed Caches

The `data/` directory contains heavy precomputed artifacts (~700 MB total):

| File | Size | Description |
|------|------|-------------|
| `tokenized_dataset.csv` | ~70 MB | Full headline corpus with preprocessed tokens |
| `document_embeddings_mpnet.npz` | ~400 MB | MPNet document embeddings |
| `document_embeddings_glove.npz` | ~233 MB | GloVe document embeddings |

**Regenerating caches**: If you need to rebuild the MPNet cache, run `python scripts/build_mpnet_cache.py`. See `AGENTS.md` for full details on cache management and GloVe setup.

## Evaluation Results

Sample Precision@5 comparison across 8 evaluation queries:

| Query | Baseline TF-IDF | Advanced (MPNet) |
|-------|----------------|------------------|
| food delivery problem | 0.00 | 0.40 |
| late shipment issues | 0.00 | 0.80 |
| machine learning applications | 0.00 | 0.00 |
| COVID vaccine rollout | 0.00 | 0.80 |
| hurricane damage response | 0.00 | 0.80 |
| gun control debate | 0.40 | 1.00 |
| college education costs | 0.00 | 0.60 |
| animal rescue stories | 0.00 | 1.00 |
| **Average** | **0.05** | **0.68** |

## Requirements

- Python >= 3.12
- See `requirements.txt` for full dependency list (`numpy`, `pandas`, `scikit-learn`, `nltk`, `sentence-transformers`, `torch`, `matplotlib`, `streamlit`)
- Optional: `glove.6B.300d.txt` in repo root to enable the GloVe search path (defaults to MPNet transformer otherwise)
