"""Deterministic text embeddings for semantic search and content recommendation.

Default backend is TF-IDF (word + character n-grams) reduced with Truncated SVD
- an LSA embedder. It is deterministic, trains in seconds, needs no network and
no GPU, and is a genuinely strong lexical-semantic baseline.

If `sentence-transformers` is installed AND `EMBEDDING_BACKEND=sentence-transformers`
is set, that backend is used instead. The interface is identical, so nothing
downstream changes.
"""
from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "with", "your", "you", "this", "these", "those"])


def normalise_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall((text or "").lower()))


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall((text or "").lower())
    return [t for t in tokens if not drop_stopwords or t not in STOPWORDS]


@dataclass(slots=True)
class EmbeddingConfig:
    dim: int = 128
    min_df: int = 1
    max_features: int = 60000
    char_ngrams: tuple[int, int] = (3, 5)
    word_ngrams: tuple[int, int] = (1, 2)
    random_state: int = 42


class TfidfSvdEmbedder:
    """Word+char TF-IDF followed by SVD, L2-normalised. Fully deterministic."""

    backend_name = "tfidf-svd"

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self._word_vec = None
        self._char_vec = None
        self._svd = None
        self._fitted = False
        self.dim = self.config.dim

    def fit(self, texts: Sequence[str]) -> TfidfSvdEmbedder:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs = [normalise_text(t) for t in texts]
        if not docs:
            raise ValueError("cannot fit an embedder on an empty corpus")

        self._word_vec = TfidfVectorizer(
            ngram_range=self.config.word_ngrams,
            min_df=self.config.min_df,
            max_features=self.config.max_features,
            sublinear_tf=True,
            stop_words=list(STOPWORDS),
        )
        self._char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.config.char_ngrams,
            min_df=max(self.config.min_df, 2),
            max_features=self.config.max_features,
            sublinear_tf=True,
        )
        from scipy.sparse import hstack

        word_m = self._word_vec.fit_transform(docs)
        char_m = self._char_vec.fit_transform(docs)
        matrix = hstack([word_m, char_m]).tocsr()

        n_components = int(min(self.config.dim, max(2, min(matrix.shape) - 1)))
        self._svd = TruncatedSVD(n_components=n_components, random_state=self.config.random_state, algorithm="randomized")
        self._svd.fit(matrix)
        self.dim = n_components
        self._fitted = True
        return self

    def _raw_transform(self, texts: Sequence[str]) -> np.ndarray:
        from scipy.sparse import hstack

        docs = [normalise_text(t) for t in texts]
        word_m = self._word_vec.transform(docs)
        char_m = self._char_vec.transform(docs)
        return self._svd.transform(hstack([word_m, char_m]).tocsr())

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("embedder must be fitted before calling transform()")
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._raw_transform(texts).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1.0, norms)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    def encode_one(self, text: str) -> np.ndarray:
        return self.transform([text])[0]

    @property
    def explained_variance(self) -> float:
        if not self._fitted or self._svd is None:
            return 0.0
        return round(float(self._svd.explained_variance_ratio_.sum()), 5)


class SentenceTransformerEmbedder:  # pragma: no cover - optional heavy dependency
    """Optional drop-in backend when sentence-transformers is installed."""

    backend_name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self._fitted = True

    def fit(self, texts: Sequence[str]) -> SentenceTransformerEmbedder:
        return self  # pre-trained

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(
            self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False), dtype=np.float32
        )

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.transform(texts)

    def encode_one(self, text: str) -> np.ndarray:
        return self.transform([text])[0]

    @property
    def explained_variance(self) -> float:
        return 1.0


def build_embedder(dim: int = 128) -> TfidfSvdEmbedder | SentenceTransformerEmbedder:
    backend = os.getenv("EMBEDDING_BACKEND", "tfidf-svd").strip().lower()
    if backend in ("sentence-transformers", "st"):
        try:
            return SentenceTransformerEmbedder(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        except Exception:
            pass  # dependency missing -> deterministic fallback below
    return TfidfSvdEmbedder(EmbeddingConfig(dim=dim))


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    b = a if b is None else b
    return np.clip(a @ b.T, -1.0, 1.0)
