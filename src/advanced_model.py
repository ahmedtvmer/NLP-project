"""
Task 4 — Advanced model: semantic search with dense embeddings.

Primary: mean-pooled GloVe (Common Crawl 42B / 840B / Wikipedia+Giga 6B — use 6B.300d
to match `feature_metadata.json`).

Optional: sentence-transformers (MiniLM etc.) when `sentence_transformers` is installed;
uses raw headline text for stronger paraphrase handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from .text_preprocessing import parse_cleaned_tokens, preprocess_text


@dataclass
class SearchHit:
    rank: int
    index: int
    score: float
    text: str


def _load_glove_txt(path: Path, dim: int) -> Dict[str, np.ndarray]:
    """Load GloVe *.txt into a dict word -> float32 vector."""
    glove: Dict[str, np.ndarray] = {}
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != dim + 1:
                continue
            word = parts[0]
            glove[word] = np.asarray(parts[1 : dim + 1], dtype=np.float32)
    return glove


def _mean_embedding(tokens: Sequence[str], glove: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    vecs = [glove[w] for w in tokens if w in glove]
    if not vecs:
        return None
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)


class GloVeSemanticSearch:
    """
    Semantic ranking: cosine similarity between L2-normalized mean GloVe vectors.
    Fits Task 4 (embedding-based retrieval). Use after Task 1 CSV is available.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        glove_path: Union[str, Path],
        *,
        cleaned_column: str = "cleaned_text",
        raw_text_column: str = "text",
        embedding_dim: int = 300,
        embeddings_cache: Optional[Union[str, Path]] = None,
        rebuild_cache: bool = False,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.cleaned_column = cleaned_column
        self.raw_text_column = raw_text_column
        self.embedding_dim = embedding_dim
        self.glove_path = Path(glove_path)
        cache_path = Path(embeddings_cache) if embeddings_cache else Path("data/document_embeddings_glove.npz")

        if cleaned_column not in self.df.columns:
            raise KeyError(f"Missing column `{cleaned_column}` in dataframe; got {list(self.df.columns)!r}")

        token_lists = [parse_cleaned_tokens(x) for x in self.df[cleaned_column].tolist()]

        if cache_path.exists() and not rebuild_cache:
            data = np.load(cache_path, allow_pickle=False)
            if "embeddings" not in data.files:
                raise KeyError(f"Cache {cache_path} must contain array key 'embeddings'")
            self.doc_embeddings = data["embeddings"].astype(np.float32)
            rows, cols = self.doc_embeddings.shape
            if rows != len(self.df):
                raise ValueError(
                    f"Cached embeddings rows ({rows}) do not match dataframe length ({len(self.df)})"
                )
            if cols != embedding_dim:
                raise ValueError(
                    f"Cached embedding dim ({cols}) != expected embedding_dim ({embedding_dim}); "
                    "use --rebuild-cache or delete the stale cache."
                )
        else:
            if not self.glove_path.is_file():
                raise FileNotFoundError(
                    f"GloVe file not found: {self.glove_path}. Download glove.6B.300d.txt from "
                    "https://nlp.stanford.edu/projects/glove/"
                )
            glove = _load_glove_txt(self.glove_path, embedding_dim)
            if not glove:
                raise ValueError(
                    f"No vectors loaded from {self.glove_path}. Check path and embedding_dim ({embedding_dim})."
                )
            self.doc_embeddings = self._compute_doc_matrix(token_lists, glove)
            np.savez_compressed(
                cache_path,
                embeddings=self.doc_embeddings.astype(np.float32),
            )

        # Cosine similarity via dot product on L2-normalized rows
        self._doc_normalized = normalize(self.doc_embeddings, norm="l2", axis=1, copy=True)

        self._glove: Optional[Dict[str, np.ndarray]] = None

    def _lazy_glove_for_queries(self) -> Dict[str, np.ndarray]:
        if self._glove is None:
            if not self.glove_path.is_file():
                raise FileNotFoundError(f"GloVe file missing for query encoding: {self.glove_path}")
            loaded = _load_glove_txt(self.glove_path, self.embedding_dim)
            if not loaded:
                raise ValueError(f"No GloVe vectors loaded from {self.glove_path}")
            self._glove = loaded
        return self._glove

    @staticmethod
    def _compute_doc_matrix(token_lists: List[List[str]], glove: Dict[str, np.ndarray]) -> np.ndarray:
        n = len(token_lists)
        dim = len(next(iter(glove.values())))
        out = np.zeros((n, dim), dtype=np.float32)
        for i, toks in enumerate(token_lists):
            m = _mean_embedding(toks, glove)
            if m is not None:
                out[i] = m
        return out

    def query_embedding(self, query: str) -> np.ndarray:
        tokens = preprocess_text(query)
        g = self._lazy_glove_for_queries()
        m = _mean_embedding(tokens, g)
        if m is None:
            return np.zeros((self.embedding_dim,), dtype=np.float32)
        v = normalize(m.reshape(1, -1), norm="l2", axis=1)
        return v.astype(np.float32).ravel()

    def all_scores(self, query: str) -> np.ndarray:
        q = self.query_embedding(query)
        return self._doc_normalized @ q

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        scores = self.all_scores(query)
        n_docs = len(self.df)
        if n_docs == 0:
            return []
        if top_k < 1:
            return []
        k = min(top_k, n_docs)
        idx_partial = np.argpartition(-scores, k - 1)[:k]
        idx_sorted = idx_partial[np.argsort(-scores[idx_partial])]

        hits: List[SearchHit] = []
        for rank, j in enumerate(idx_sorted, start=1):
            txt = ""
            if self.raw_text_column in self.df.columns:
                txt = str(self.df.iloc[int(j)][self.raw_text_column])
            hits.append(SearchHit(rank=rank, index=int(j), score=float(scores[j]), text=txt))
        return hits


class TransformerSemanticSearch:
    """
    Optional stronger semantic model (BERT-style sentence embeddings).
    Install: pip install sentence-transformers torch
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        *,
        text_column: str = "text",
        embeddings_cache: Optional[Union[str, Path]] = None,
        rebuild_cache: bool = False,
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "TransformerSemanticSearch requires `sentence_transformers`. "
                "Install with: pip install sentence-transformers torch"
            ) from e

        self.df = dataframe.reset_index(drop=True)
        self.text_column = text_column
        if text_column not in self.df.columns:
            raise KeyError(f"Missing column `{text_column}` in dataframe; got {list(self.df.columns)!r}")
        cache_path = Path(embeddings_cache) if embeddings_cache else Path("data/document_embeddings_minilm.npz")

        corpus = self.df[text_column].astype(str).tolist()

        if cache_path.exists() and not rebuild_cache:
            data = np.load(cache_path, allow_pickle=False)
            if "embeddings" not in data.files:
                raise KeyError(f"Cache {cache_path} must contain array key 'embeddings'")
            self.doc_embeddings = data["embeddings"].astype(np.float32)
            if self.doc_embeddings.shape[0] != len(self.df):
                raise ValueError("Cached transformer embeddings length mismatches dataframe.")
        else:
            model = SentenceTransformer(model_name, device=device)
            self.doc_embeddings = model.encode(
                corpus,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            np.savez_compressed(cache_path, embeddings=self.doc_embeddings)

        self._model_name = model_name
        self._device = device
        self._st_model: Any = None

    def _model(self):  # lazy load for encoding queries only
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self._model_name, device=self._device)
        return self._st_model

    def all_scores(self, query: str) -> np.ndarray:
        q = self._model().encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)[0]
        return self.doc_embeddings @ q

    def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        scores = self.all_scores(query)
        if len(self.df) == 0:
            return []
        if top_k < 1:
            return []
        k = min(top_k, len(self.df))
        idx_partial = np.argpartition(-scores, k - 1)[:k]
        idx_sorted = idx_partial[np.argsort(-scores[idx_partial])]

        hits: List[SearchHit] = []
        for rank, j in enumerate(idx_sorted, start=1):
            txt = str(self.df.iloc[int(j)][self.text_column])
            hits.append(SearchHit(rank=rank, index=int(j), score=float(scores[j]), text=txt))
        return hits


def load_corpus(csv_path: Union[str, Path], n_rows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if n_rows is not None:
        df = df.head(n_rows).copy()
    return df


def print_hits(hits: Sequence[SearchHit]) -> None:
    for h in hits:
        short = (h.text[:200] + "…") if len(h.text) > 200 else h.text
        print(f"{h.rank}. score={h.score:.4f}\n   {short}\n")


def main_cli() -> None:
    import argparse

    def positive_int(value: str) -> int:
        n = int(value)
        if n < 1:
            raise argparse.ArgumentTypeError(f"Expected positive integer; got {n}")
        return n

    parser = argparse.ArgumentParser(description="Task 4: GloVe semantic search")
    parser.add_argument("--csv", type=Path, default=Path("data/tokenized_dataset.csv"))
    parser.add_argument(
        "--glove",
        type=Path,
        default=None,
        help="Path to glove.6B.300d.txt (required unless --transformer)",
    )
    parser.add_argument("--query", type=str, default="food delivery problem")
    parser.add_argument("--top-k", type=positive_int, default=5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/document_embeddings_glove.npz"),
        help="Compressed .npz with array 'embeddings'",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--transformer",
        action="store_true",
        help="Use sentence-transformers instead of GloVe (requires extra packages)",
    )
    args = parser.parse_args()

    df = load_corpus(args.csv, args.max_rows)
    if args.transformer:
        engine: Any = TransformerSemanticSearch(
            df,
            embeddings_cache=args.cache.parent / "document_embeddings_minilm.npz",
            rebuild_cache=args.rebuild_cache,
        )
    else:
        if not args.glove:
            parser.error("--glove is required unless you pass --transformer")
        engine = GloVeSemanticSearch(
            df,
            args.glove,
            embeddings_cache=args.cache,
            rebuild_cache=args.rebuild_cache,
        )

    hits = engine.search(args.query, top_k=args.top_k)
    print_hits(hits)


if __name__ == "__main__":
    main_cli()
