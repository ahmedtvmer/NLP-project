# AGENTS.md

## Dependencies
- Root `pyproject.toml` does not list project dependencies.
- Install from `requirements.txt`: `numpy`, `pandas`, `scikit-learn`, `nltk`, `sentence-transformers`, `torch`, `matplotlib`, `streamlit`.
- Optional transformer path: uncomment `sentence-transformers` in requirements, then pass `--transformer` to CLI.

## Running the search CLI
```bash
python scripts/run_semantic_search.py --glove /path/to/glove.6B.300d.txt --query "food delivery problem"
```
- If you want the transformer model instead of GloVe, add `--transformer` and omit `--glove`.
- Defaults assume CWD is the project root (`data/tokenized_dataset.csv`, `data/document_embeddings_*.npz`).

## Running evaluation
```bash
python scripts/run_evaluation.py
```
- Must be run from repo root (it imports `src.advanced_model` via `sys.path.insert(0, ".")`).
- Expects `data/tokenized_dataset.csv` and `data/document_embeddings_mpnet.npz` by default.
- Saves chart to `results/evaluation_chart.png`.

## Running the Streamlit GUI
```bash
streamlit run app.py
```
- Run from repo root.
- Side-by-side comparison of Baseline (TF-IDF) and Advanced (MPNet).
- Includes "Search Results" tab and "Evaluation Chart" tab.

## Running baseline CLI
```bash
python scripts/baseline.py --query "food delivery problem" --top-k 5
```

## Large cached artifacts
- `data/` already contains heavy precomputed files:
  - `document_embeddings_glove.npz` (~233 MB)
  - `document_embeddings_mpnet.npz` (~400 MB)
  - `tfidf_matrix.npz` (~20 MB)
  - `tfidf_vectorizer.pkl`
  - `tokenized_dataset.csv` (~70 MB)
- Avoid regenerating these unless explicitly requested.

## NLTK data
- `src/text_preprocessing.py` auto-downloads required NLTK corpora quietly.
- Standalone `scripts/toy_baseline.py` downloads noisily at import time.

## Project layout
```
data/          – datasets and precomputed caches
results/       – evaluation charts and outputs
scripts/       – runnable entry points
src/           – reusable library code
app.py         – Streamlit GUI
```
