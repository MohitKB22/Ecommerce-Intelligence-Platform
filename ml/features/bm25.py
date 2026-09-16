"""Okapi BM25 lexical index.

Implemented directly so full-text ranking behaves identically on PostgreSQL and
SQLite, is unit-testable, and exposes per-term scores for search explainability.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ml.features.embeddings import tokenize


@dataclass(slots=True)
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    doc_ids: list[int] = field(default_factory=list)
    doc_len: np.ndarray = field(default_factory=lambda: np.zeros(0))
    avg_len: float = 0.0
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)  # term -> [(doc_pos, tf)]
    idf: dict[str, float] = field(default_factory=dict)
    vocabulary: set[str] = field(default_factory=set)

    @classmethod
    def build(cls, documents: Sequence[tuple[int, str]], k1: float = 1.5, b: float = 0.75) -> BM25Index:
        index = cls(k1=k1, b=b)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths: list[int] = []
        for pos, (doc_id, text) in enumerate(documents):
            tokens = tokenize(text)
            lengths.append(len(tokens))
            index.doc_ids.append(doc_id)
            for term, tf in Counter(tokens).items():
                postings[term].append((pos, tf))
        n = max(len(documents), 1)
        index.doc_len = np.asarray(lengths, dtype=np.float32)
        index.avg_len = float(index.doc_len.mean()) if len(index.doc_len) else 0.0
        index.postings = dict(postings)
        index.idf = {
            term: math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5)) for term, plist in postings.items()
        }
        index.vocabulary = set(postings)
        return index

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def score(self, query: str, limit: int | None = None) -> dict[int, float]:
        """Return {doc_id: bm25_score} for documents matching at least one term."""
        terms = tokenize(query)
        if not terms or self.size == 0:
            return {}
        scores = np.zeros(self.size, dtype=np.float32)
        denom_norm = self.k1 * (1 - self.b + self.b * (self.doc_len / (self.avg_len or 1.0)))
        for term, q_tf in Counter(terms).items():
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf.get(term, 0.0)
            positions = np.fromiter((p for p, _ in plist), dtype=np.int64, count=len(plist))
            tfs = np.fromiter((t for _, t in plist), dtype=np.float32, count=len(plist))
            contrib = idf * (tfs * (self.k1 + 1)) / (tfs + denom_norm[positions])
            scores[positions] += contrib * (1.0 + 0.15 * (q_tf - 1))
        nz = np.nonzero(scores)[0]
        if limit is not None and nz.size > limit:
            nz = nz[np.argsort(-scores[nz])[:limit]]
        return {self.doc_ids[int(i)]: float(scores[int(i)]) for i in nz}

    def term_exists(self, term: str) -> bool:
        return term in self.vocabulary

    def closest_terms(self, term: str, max_distance: int = 2, limit: int = 5) -> list[str]:
        """Damerau-Levenshtein neighbours - powers typo tolerance."""
        term = term.lower()
        if term in self.vocabulary:
            return [term]
        candidates: list[tuple[int, int, str]] = []
        for vocab_term in self.vocabulary:
            if abs(len(vocab_term) - len(term)) > max_distance:
                continue
            dist = _damerau_levenshtein(term, vocab_term, max_distance)
            if dist <= max_distance:
                candidates.append((dist, -len(self.postings.get(vocab_term, [])), vocab_term))
        candidates.sort()
        return [t for _, _, t in candidates[:limit]]


def _damerau_levenshtein(a: str, b: str, max_distance: int = 3) -> int:
    """Bounded optimal string alignment distance."""
    la, lb = len(a), len(b)
    if abs(la - lb) > max_distance:
        return max_distance + 1
    prev2: list[int] = []
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_min = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + cost)
            row_min = min(row_min, cur[j])
        if row_min > max_distance:
            return max_distance + 1
        prev2, prev = prev, cur
    return prev[lb]
