#!/usr/bin/env python3
"""Build MPNet embedding cache standalone."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from src.advanced_model import TransformerSemanticSearch, load_corpus

CSV_PATH = Path("data/tokenized_dataset.csv")
CACHE_PATH = Path("data/document_embeddings_mpnet.npz")

df = load_corpus(CSV_PATH, n_rows=None)
print(f"Building MPNet cache for {len(df)} documents...")
engine = TransformerSemanticSearch(df, text_column="text", embeddings_cache=CACHE_PATH)
print(f"Cache saved to {CACHE_PATH}")
