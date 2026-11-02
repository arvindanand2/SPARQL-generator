#!/usr/bin/env python3
"""
bm25_index.py
-------------
Lightweight BM25 store for classes & predicates.

- Loads records using the manifests in an artifacts directory
- Builds BM25Okapi indices over label + lexical_text + aliases_enriched
- Exposes get_scores(which, query) to return an array of BM25 scores (aligned to records order)
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON Lines artifact into memory."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _corpus_from_records(records: List[Dict[str,Any]]) -> List[List[str]]:
    """Derive a tokenized corpus from record labels, lexicals, and aliases."""
    corpus = []
    for r in records:
        txt = " ".join(filter(None, [
            r.get("label",""),
            r.get("lexical_text",""),
            " ".join(r.get("aliases_enriched", []))
        ])).lower()
        corpus.append(txt.split())
    return corpus

class BM25Store:
    """In-memory BM25 indexes backed by exported class and predicate manifests."""

    def __init__(self, artifacts_dir: str):
        """Load manifests and instantiate BM25 scorers.

        Parameters
        ----------
        artifacts_dir : str
            Directory containing ``classes_manifest.json`` and ``predicates_manifest.json``.
        """
        art = Path(artifacts_dir)
        self.classes_meta = json.load((art / "classes_manifest.json").open("r", encoding="utf-8"))
        self.pred_meta    = json.load((art / "predicates_manifest.json").open("r", encoding="utf-8"))
        self.class_records = _read_jsonl(Path(self.classes_meta["records_path"]))
        self.pred_records  = _read_jsonl(Path(self.pred_meta["records_path"]))
        self.bm25_classes = BM25Okapi(_corpus_from_records(self.class_records))
        self.bm25_preds   = BM25Okapi(_corpus_from_records(self.pred_records))

    def get_scores(self, which: str, query: str):
        """Retrieve BM25 relevance scores for the requested artifact type.

        Parameters
        ----------
        which : str
            Either ``"class"`` or ``"predicate"`` to select the underlying corpus.
        query : str
            User query or clause text to evaluate against the corpus.

        Returns
        -------
        list[float]
            BM25 scores ordered identically to the corresponding records manifest.
        """
        if which not in {"class","predicate"}:
            raise ValueError("which must be 'class' or 'predicate'")
        bm = self.bm25_classes if which=="class" else self.bm25_preds
        return bm.get_scores(query.lower().split())
