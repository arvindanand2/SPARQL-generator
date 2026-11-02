#!/usr/bin/env python3
"""
retrieval.py — lightweight semantic search over ontology classes & predicates.

Features
- Loads precomputed embeddings from your artifacts directory
- Reads the original JSONL records (paths stored in each manifest)
- Embeds queries with OpenAI (uses .env via python-dotenv if present)
- Returns top-k classes or predicates with cosine scores
- Schema-aware boost for predicates using domain/range hints

Usage
  from retrieval import Retrieval
  r = Retrieval(artifacts_dir="./artifacts", model="text-embedding-3-large")
  r.search_classes("sports and leisure industry", k=5)
  r.search_predicates("serving industry", subj_types=[".../Organization"], obj_types=[".../Industry"], k=5)
"""

import os, json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Optional .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from openai import OpenAI


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON Lines file into a list of dictionaries."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length to optimize cosine similarity."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms


def cosine_sim_matrix(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute cosine similarities between a query vector and matrix of embeddings."""
    # q: (D,), X: (N,D), both should be normalized (dot == cosine)
    return X @ q


class Retrieval:
    """Perform embedding-based search over ontological classes and predicates."""

    def __init__(self, artifacts_dir: str, model: str = "text-embedding-3-large", base_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        artifacts_dir must contain:
          - classes_manifest.json, predicates_manifest.json
          - classes_vectors.npy,  predicates_vectors.npy
        """
        self.art = Path(artifacts_dir)
        self.model = model
        # OpenAI client (uses env if not passed)
        kwargs = {}
        if api_key or os.getenv("OPENAI_API_KEY"):
            kwargs["api_key"] = api_key or os.getenv("OPENAI_API_KEY")
        if base_url or os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(**kwargs) if kwargs else OpenAI()

        # Load manifests & vectors
        self.classes_meta = json.load((self.art / "classes_manifest.json").open("r", encoding="utf-8"))
        self.pred_meta    = json.load((self.art / "predicates_manifest.json").open("r", encoding="utf-8"))

        self.class_vecs = np.load(self.classes_meta["vectors_path"])
        self.pred_vecs  = np.load(self.pred_meta["vectors_path"])

        # Assert normalized (best-effort fix if not)
        self.class_vecs = l2_normalize(self.class_vecs)
        self.pred_vecs  = l2_normalize(self.pred_vecs)

        # Load records (must be same order as vectors)
        self.class_records = read_jsonl(Path(self.classes_meta["records_path"]))
        self.pred_records  = read_jsonl(Path(self.pred_meta["records_path"]))

        if len(self.class_records) != self.class_vecs.shape[0]:
            raise ValueError("Classes: records count ≠ vectors count")
        if len(self.pred_records) != self.pred_vecs.shape[0]:
            raise ValueError("Predicates: records count ≠ vectors count")

    # --- Core embedding ---
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query string via OpenAI and return a normalized vector."""
        res = self.client.embeddings.create(model=self.model, input=[text])
        v = np.array(res.data[0].embedding, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    # --- Search helpers ---
    def _format_hit(self, idx: int, score: float, recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return a consistent hit dictionary for downstream consumers."""
        r = recs[idx]
        return {
            "iri": r.get("iri"),
            "label": r.get("label"),
            "score": float(score),
            "record": r
        }

    def search_classes(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Return top-k class candidates ranked by cosine similarity."""
        qv = self.embed_query(query)
        sims = cosine_sim_matrix(qv, self.class_vecs)
        top = np.argsort(sims)[::-1][:k]
        return [self._format_hit(i, sims[i], self.class_records) for i in top]

    def _schema_bonus(self, pred_rec: Dict[str, Any], subj_types: Optional[List[str]], obj_types: Optional[List[str]]) -> float:
        """Reward or penalize predicate hits based on domain/range hints."""
        subj_types = set(subj_types or [])
        obj_types  = set(obj_types or [])
        doms = set(pred_rec.get("domain_class_iris", []) or [])
        rngs = set(pred_rec.get("range_class_iris", []) or [])
        s_ok = (not doms) or bool(doms & subj_types)
        o_ok = (not rngs) or bool(rngs & obj_types)
        if s_ok and o_ok:
            return +0.12
        if doms or rngs:
            return -0.12
        return 0.0

    def search_predicates(self, query: str, k: int = 10, subj_types: Optional[List[str]] = None, obj_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Return top-k predicate candidates with optional schema re-ranking."""
        qv = self.embed_query(query)
        sims = cosine_sim_matrix(qv, self.pred_vecs)
        idxs = np.argsort(sims)[::-1][: max(k, 25)]  # take a wider pool, then re-rank

        scored: List[Tuple[float, int]] = []
        for i in idxs:
            base = float(sims[i])
            bonus = self._schema_bonus(self.pred_records[i], subj_types, obj_types)
            scored.append((base + bonus, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [i for _, i in scored[:k]]
        return [self._format_hit(i, sims[i], self.pred_records) for i in top]
