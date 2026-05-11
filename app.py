#!/usr/bin/env python3
"""
Streamlit GUI for the Intelligent Search Engine.

Run from repo root:
    streamlit run app.py
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.advanced_model import (
    GloVeSemanticSearch,
    TransformerSemanticSearch,
    load_corpus,
    SearchHit,
)
from src.text_preprocessing import preprocess_text, parse_cleaned_tokens

CSV_PATH = PROJECT_ROOT / "data" / "tokenized_dataset.csv"
GLOVE_PATH = PROJECT_ROOT / "glove.6B.300d.txt"
GLOVE_CACHE = PROJECT_ROOT / "data" / "document_embeddings_glove.npz"
TRANSFORMER_CACHE = PROJECT_ROOT / "data" / "document_embeddings_mpnet.npz"
CHART_PATH = PROJECT_ROOT / "results" / "evaluation_chart.png"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading corpus...")
def get_corpus():
    return load_corpus(CSV_PATH, n_rows=None)


@st.cache_resource(show_spinner="Loading transformer model...")
def get_transformer_engine(df):
    return TransformerSemanticSearch(
        df, text_column="text", embeddings_cache=TRANSFORMER_CACHE
    )


# ---------------------------------------------------------------------------
# Search engines
# ---------------------------------------------------------------------------

class TFIDFSearch:
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

    def all_scores(self, query):
        tokens = preprocess_text(query)
        query_str = " ".join(tokens)
        q_vec = self.vectorizer.transform([query_str])
        return cosine_similarity(q_vec, self.tfidf_matrix).flatten()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hits_to_df(hits):
    return pd.DataFrame([
        {"Rank": h.rank, "Index": h.index, "Score": round(h.score, 4),
         "Snippet": (h.text[:150] + "...") if len(h.text) > 150 else h.text}
        for h in hits
    ])


# ---------------------------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Intelligent Search Engine", layout="wide")
st.title("Intelligent Search Engine")
st.markdown("Baseline (TF-IDF) vs Advanced (MPNet)")

# Sidebar
with st.sidebar:
    st.header("Controls")
    query = st.text_input("Query", value="food delivery problem")
    top_k = st.slider("Top-k results", min_value=1, max_value=20, value=5)

    st.divider()
    use_glove = False
    # st.subheader("GloVe Availability")
    # glove_available = GLOVE_PATH.is_file()
    # if glove_available:
    #     st.success("glove.6B.300d.txt found")
    #     use_glove = st.toggle("Use GloVe instead of Transformer", value=False)
    # else:
    #     st.warning("glove.6B.300d.txt not found\n\nTransformer model will be used.")
    #     use_glove = False

    run_button = st.button("Run Search", type="primary")

    st.divider()
    if st.button("Re-run Evaluation"):
        with st.spinner("Running evaluation..."):
            proc = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "run_evaluation.py")],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT)
            )
        if proc.returncode == 0:
            st.success("Evaluation complete!")
            st.rerun()
        else:
            st.error(f"Evaluation failed:\n{proc.stderr}")

# Tabs
tab_search, tab_eval = st.tabs(["Search Results", "Evaluation Chart"])

# ---------------------------------------------------------------------------
# Tab 1 — Search Results
# ---------------------------------------------------------------------------

with tab_search:
    if run_button:
        df = get_corpus()

        # Baseline
        with st.spinner("Running TF-IDF baseline..."):
            baseline_engine = TFIDFSearch(df, text_column="text")
            baseline_hits = baseline_engine.search(query, top_k=top_k)

        # Advanced
        if use_glove and glove_available:
            with st.spinner("Loading GloVe model..."):
                advanced_engine = GloVeSemanticSearch(
                    df, glove_path=GLOVE_PATH, embeddings_cache=GLOVE_CACHE
                )
            model_label = "GloVe"
        else:
            with st.spinner("Loading Transformer model..."):
                advanced_engine = get_transformer_engine(df)
            model_label = "MPNet"

        with st.spinner("Running advanced model..."):
            advanced_hits = advanced_engine.search(query, top_k=top_k)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Baseline (TF-IDF)")
            st.dataframe(hits_to_df(baseline_hits), use_container_width=True)

        with col2:
            st.subheader(f"Advanced ({model_label})")
            st.dataframe(hits_to_df(advanced_hits), use_container_width=True)
    else:
        st.info("Enter a query and click **Run Search** to see results side-by-side.")

# ---------------------------------------------------------------------------
# Tab 2 — Evaluation Chart
# ---------------------------------------------------------------------------

with tab_eval:
    if CHART_PATH.exists():
        st.image(str(CHART_PATH), caption="Baseline vs Advanced — Precision@k")
    else:
        st.warning("No evaluation chart found. Click **Re-run Evaluation** in the sidebar.")
