#!/usr/bin/env python3
"""
Baseline TF-IDF search CLI.

Usage from repo root:
    python scripts/baseline.py --query "food delivery problem" --top-k 5
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.advanced_model import load_corpus, SearchHit
from src.text_preprocessing import preprocess_text, parse_cleaned_tokens

CSV_PATH = PROJECT_ROOT / "data" / "tokenized_dataset.csv"


class TFIDFSearch:
    """TF-IDF baseline: cosine similarity between query and corpus vectors."""

    def __init__(self, dataframe, text_column="text", cleaned_column="cleaned_text"):
        self.df = dataframe.reset_index(drop=True)
        self.text_column = text_column
        self.cleaned_column = cleaned_column
        corpus = []
        for cell in self.df[cleaned_column].tolist():
            tokens = parse_cleaned_tokens(cell)
            corpus.append(" ".join(tokens))
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query, top_k=5):
        tokens = preprocess_text(query)
        query_str = " ".join(tokens)
        q_vec = self.vectorizer.transform([query_str])
        scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_idx = np.argsort(-scores)[:top_k]
        return [
            SearchHit(rank=rank, index=int(j), score=float(scores[j]),
                      text=str(self.df.iloc[int(j)][self.text_column]))
            for rank, j in enumerate(top_idx, 1)
        ]


def main():
    parser = argparse.ArgumentParser(description="Baseline TF-IDF search")
    parser.add_argument("--query", type=str, default="food delivery problem")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    df = load_corpus(CSV_PATH, args.max_rows)
    engine = TFIDFSearch(df, text_column="text")

    # Optional: preprocess query the same way as cleaned_text
    # tokens = preprocess_text(args.query)
    # query_str = " ".join(tokens)
    query_str = args.query

    hits = engine.search(query_str, top_k=args.top_k)

    print(f"\nQuery: \"{args.query}\"")
    print(f"Top-{args.top_k} Results (TF-IDF Baseline):\n")
    for h in hits:
        snippet = (h.text[:120] + "...") if len(h.text) > 120 else h.text
        print(f"  [{h.rank}] idx={h.index:5d}  score={h.score:.4f}")
        print(f"       {snippet}\n")


if __name__ == "__main__":
    main()
