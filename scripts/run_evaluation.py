import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(".")
sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_model import GloVeSemanticSearch, TransformerSemanticSearch, load_corpus, SearchHit
from src.text_preprocessing import preprocess_text, parse_cleaned_tokens

CSV_PATH        = PROJECT_ROOT / "data" / "tokenized_dataset.csv"
GLOVE_PATH      = PROJECT_ROOT / "glove.6B.300d.txt"
TOP_K           = 5
MAX_ROWS        = None
USE_TRANSFORMER = True


QUERIES = [
    {"query": "food delivery problem",         "relevant_indices": [4941, 87989, 138526, 157525, 1950, 48572]},
    {"query": "late shipment issues",           "relevant_indices": [144173, 176638, 17785, 99118]},
    {"query": "machine learning applications", "relevant_indices": [115615, 173360]},
    {"query": "COVID vaccine rollout",         "relevant_indices": [3473, 2019, 2312, 2280]},
    {"query": "hurricane damage response",     "relevant_indices": [21617, 183605, 183105, 182491]},
    {"query": "gun control debate",            "relevant_indices": [59348, 23387, 97946, 12682, 93504]},
    {"query": "college education costs",       "relevant_indices": [155020, 107831, 131170]},
    {"query": "animal rescue stories",         "relevant_indices": [119229, 48284, 50833, 118539, 144663]},
]


class TFIDFSearch:
    def __init__(self, dataframe, text_column="text", cleaned_column="cleaned_text"):
        self.df = dataframe.reset_index(drop=True)
        self.text_column = text_column
        self.cleaned_column = cleaned_column
        # Build corpus from preprocessed tokens (same pipeline as cleaned_text)
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

    def all_scores(self, query):
        tokens = preprocess_text(query)
        query_str = " ".join(tokens)
        q_vec = self.vectorizer.transform([query_str])
        return cosine_similarity(q_vec, self.tfidf_matrix).flatten()





def precision_at_k(hits, relevant_indices, k):
    top_k_indices = [h.index for h in hits[:k]]
    relevant_set  = set(relevant_indices)
    return sum(1 for idx in top_k_indices if idx in relevant_set) / k


def evaluate(baseline_engine, advanced_engine):
    results = []
    for entry in QUERIES:
        q, rel_idx = entry["query"], entry["relevant_indices"]
        b_hits = baseline_engine.search(q, TOP_K)
        a_hits = advanced_engine.search(q, TOP_K)
        p_b = precision_at_k(b_hits, rel_idx, TOP_K)
        p_a = precision_at_k(a_hits, rel_idx, TOP_K)
        results.append({
            "Query"               : q,
            f"Baseline P@{TOP_K}" : round(p_b, 2),
            f"Advanced P@{TOP_K}" : round(p_a, 2),
            "Best Model"          : "Advanced" if p_a > p_b else "Baseline",
            "_b_hits": b_hits, "_a_hits": a_hits, "_relevant": set(rel_idx),
        })
    return results


def print_breakdown(results):
    for i, r in enumerate(results):
        rel = r["_relevant"]
        print(f"\n{'='*60}\n Query {i+1}: \"{r['Query']}\"\n{'='*60}")
        for label, hits in [("Baseline (TF-IDF)", r["_b_hits"]),
                             ("Advanced (Embeddings)", r["_a_hits"])]:
            print(f"\n  {label}:")
            for h in hits[:TOP_K]:
                tag = "Relevant" if h.index in rel else "Not Relevant"
                snippet = (h.text[:100] + "...") if len(h.text) > 100 else h.text
                print(f"   [{h.rank}] idx={h.index:5d}  score={h.score:.4f}  {tag}")
                print(f"         {snippet}")
            pk_key = f"Baseline P@{TOP_K}" if label.startswith("Baseline") else f"Advanced P@{TOP_K}"
            print(f"   Precision@{TOP_K} = {r[pk_key]:.2f}")


def plot_results(results):
    b_scores = [r[f"Baseline P@{TOP_K}"] for r in results]
    a_scores = [r[f"Advanced P@{TOP_K}"] for r in results]
    x = np.arange(len(results))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, b_scores, width, label="Baseline TF-IDF", color="steelblue")
    bars2 = ax.bar(x + width/2, a_scores, width, label="Advanced Embeddings", color="tomato")

    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}\n{r['Query'][:25]}" for i, r in enumerate(results)], fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel(f"Precision@{TOP_K}", fontsize=12)
    ax.set_title(f"Baseline vs Advanced — Precision@{TOP_K}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("results/evaluation_chart.png", dpi=150, bbox_inches="tight")
    plt.show()


def print_summary(results):
    avg_b = np.mean([r[f"Baseline P@{TOP_K}"] for r in results])
    avg_a = np.mean([r[f"Advanced P@{TOP_K}"] for r in results])
    best = "Advanced" if avg_a > avg_b else "Baseline"

    print(f"\nQueries Evaluated    : {len(results)}")
    print(f"k                    : {TOP_K}")
    print(f"Avg Baseline P@{TOP_K}  : {avg_b:.2f}")
    print(f"Avg Advanced P@{TOP_K}  : {avg_a:.2f}")
    print(f"Best Model           : {best}")

    cols = ["Query", f"Baseline P@{TOP_K}", f"Advanced P@{TOP_K}", "Best Model"]
    df_res = pd.DataFrame(results)[cols]
    avg_row = pd.DataFrame([{
        "Query": "Average",
        f"Baseline P@{TOP_K}": round(avg_b, 2),
        f"Advanced P@{TOP_K}": round(avg_a, 2),
        "Best Model": best
    }])
    print("\n" + pd.concat([df_res, avg_row], ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    df = load_corpus(CSV_PATH, MAX_ROWS)

    baseline_engine = TFIDFSearch(df, text_column="text")

    if USE_TRANSFORMER:
        advanced_engine = TransformerSemanticSearch(
            df, text_column="text",
            embeddings_cache=PROJECT_ROOT / "data" / "document_embeddings_mpnet.npz",
        )
    else:
        advanced_engine = GloVeSemanticSearch(
            df, glove_path=GLOVE_PATH,
            embeddings_cache=PROJECT_ROOT / "data" / "document_embeddings_glove.npz",
        )

    results = evaluate(baseline_engine, advanced_engine)

    print_summary(results)
    print_breakdown(results)
    plot_results(results)
