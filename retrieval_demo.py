
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Retrieval demo for Predicate/Class indices (hybrid: embeddings + BM25)

Usage (build vectors first):
  python retrieval_demo.py build --pred predicate_index.json --cls class_index.json --out ./store

Query predicates:
  python retrieval_demo.py query --type predicate --q "serves aerospace" --store ./store --k 5

Query classes:
  python retrieval_demo.py query --type class --q "process like anodizing" --store ./store --k 5

Interactive mode:
  python retrieval_demo.py repl --store ./store --type predicate
"""

import os
import json
import math
import time
import argparse
import pathlib
import hashlib
import random
from typing import List, Dict, Tuple, Any
import numpy as np

# ------------------ Utilities ------------------

def load_json_array(path: str) -> List[Dict[str, Any]]:
    """Load a JSON array file and validate its structure."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list), "Expected a JSON array"
    return data

def save_npz(path: str, **arrays):
    """Persist numpy arrays with compression."""
    import numpy as _np
    _np.savez_compressed(path, **arrays)

def load_npz(path: str):
    """Load a compressed npz archive allowing pickle contents."""
    return np.load(path, allow_pickle=True)

def simple_tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenizer used by the lightweight BM25."""
    return [t for t in ''.join(ch.lower() if ch.isalnum() else ' ' for ch in text).split() if t]

# ------------------ BM25 (light implementation) ------------------

class BM25:
    """Minimal BM25 implementation for quick lexical scoring."""

    def __init__(self, corpus_texts: List[str], k1: float = 1.5, b: float = 0.75):
        """Precompute term statistics for the provided corpus."""
        self.k1 = k1
        self.b = b
        self.docs = [simple_tokenize(t) for t in corpus_texts]
        self.n_docs = len(self.docs)
        self.doc_len = [len(d) for d in self.docs]
        self.avg_len = float(sum(self.doc_len)) / max(1, self.n_docs)
        # term frequencies per doc
        self.tf = []
        df = {}
        for d in self.docs:
            tf_map = {}
            for tok in d:
                tf_map[tok] = tf_map.get(tok, 0) + 1
            self.tf.append(tf_map)
            for tok in set(d):
                df[tok] = df.get(tok, 0) + 1
        self.df = df
        # idf with log with +0.5 smoothing
        self.idf = {tok: math.log((self.n_docs - dfv + 0.5) / (dfv + 0.5) + 1) for tok, dfv in df.items()}

    def score(self, q: str) -> np.ndarray:
        """Compute normalized BM25 scores for all documents."""
        if self.n_docs == 0:
            return np.zeros(0, dtype=float)
        q_tokens = simple_tokenize(q)
        scores = np.zeros(self.n_docs, dtype=float)
        for i, tf_map in enumerate(self.tf):
            dl = self.doc_len[i]
            K = self.k1 * (1 - self.b + self.b * (dl / max(1.0, self.avg_len)))
            s = 0.0
            for t in q_tokens:
                if t not in self.idf:
                    continue
                idf_t = self.idf[t]
                f = tf_map.get(t, 0)
                s += idf_t * ( (f * (self.k1 + 1)) / (f + K) )
            scores[i] = s
        # normalize to [0,1] for easier mixing
        if scores.max() > 0:
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        return scores

# ------------------ Embedding backends ------------------

def embed_openai(batch_texts: List[str], model: str = "text-embedding-3-large", api_base: str = "https://api.openai.com/v1") -> np.ndarray:
    """Call the OpenAI embeddings REST endpoint without the SDK."""
    # Minimal REST call, avoids extra SDK installs
    import requests
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = f"{api_base}/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": model, "input": batch_texts}
    r = requests.post(url, headers=headers, json=data, timeout=60)
    r.raise_for_status()
    js = r.json()
    vecs = [np.array(item["embedding"], dtype=np.float32) for item in js["data"]]
    return np.vstack(vecs)

def embed_sentence_transformers(batch_texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Embed using sentence-transformers (SBERT) models."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers not installed") from e
    m = SentenceTransformer(model_name)
    emb = m.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)
    return emb.astype(np.float32)

def embed_mock(batch_texts: List[str], dim: int = 384) -> np.ndarray:
    """Generate deterministic pseudo-embeddings for testing."""
    # Deterministic pseudo-embeddings using SHA-256 → PRNG, good for testing the pipeline
    vecs = []
    for t in batch_texts:
        seed = int(hashlib.sha256(t.encode("utf-8")).hexdigest(), 16) % (2**32 - 1)
        rnd = random.Random(seed)
        v = np.array([rnd.random() for _ in range(dim)], dtype=np.float32)
        # L2 normalize
        v = v / (np.linalg.norm(v) + 1e-12)
        vecs.append(v)
    return np.vstack(vecs)

def get_embedder(name: str):
    """Return an embedding function by name."""
    name = (name or "openai").lower()
    if name == "openai":
        return embed_openai
    if name in ("st", "sentence-transformers", "sbert"):
        return embed_sentence_transformers
    if name in ("mock", "dummy"):
        return embed_mock
    raise ValueError(f"Unknown embedder '{name}'")

def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

def cosine_sim(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query and matrix."""
    return (q @ X.T)

# ------------------ Store building ------------------

def build_store(pred_json_path: str, cls_json_path: str, out_dir: str, embedder: str = "openai", batch: int = 128):
    """Construct embedding and BM25 stores from predicate/class JSON inputs."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load docs
    preds = load_json_array(pred_json_path)
    clss  = load_json_array(cls_json_path)

    # Build BM25 corpora
    pred_corpus = [ (d.get("lexical_text") or d.get("label") or "") for d in preds ]
    cls_corpus  = [ (d.get("lexical_text") or d.get("label") or "") for d in clss ]

    bm25_pred = BM25(pred_corpus)
    bm25_cls  = BM25(cls_corpus)

    # Build embeddings
    emb_fn = get_embedder(embedder)

    def batch_embed(texts: List[str]) -> np.ndarray:
        """Embed texts in chunks according to the configured batch size."""
        vecs = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i+batch]
            vecs.append(emb_fn(chunk))
        return np.vstack(vecs)

    pred_texts = [d.get("embed_text") or d.get("label") or "" for d in preds]
    cls_texts  = [d.get("embed_text")  or d.get("label") or "" for d in clss]

    pred_vecs = l2_normalize(batch_embed(pred_texts))
    cls_vecs  = l2_normalize(batch_embed(cls_texts))

    # Save arrays + metadata
    save_npz(str(out / "pred_vectors.npz"),
             vectors=pred_vecs,
             meta=np.array(preds, dtype=object))
    save_npz(str(out / "cls_vectors.npz"),
             vectors=cls_vecs,
             meta=np.array(clss, dtype=object))

    # Persist BM25 corpora (we just recompute on load for simplicity)
    with open(out / "bm25_pred.json", "w", encoding="utf-8") as f:
        json.dump(pred_corpus, f, ensure_ascii=False)
    with open(out / "bm25_cls.json", "w", encoding="utf-8") as f:
        json.dump(cls_corpus, f, ensure_ascii=False)

    print(f"Built store in {out.resolve()}")
    print(f"Predicates: {len(preds)}  |  Classes: {len(clss)}")
    print(f"Embedder: {embedder}")

# ------------------ Querying ------------------

def load_store(store_dir: str):
    """Load previously built embedding and BM25 stores from disk."""
    store = pathlib.Path(store_dir)
    pred_pack = load_npz(str(store / "pred_vectors.npz"))
    cls_pack  = load_npz(str(store / "cls_vectors.npz"))

    with open(store / "bm25_pred.json", "r", encoding="utf-8") as f:
        pred_corpus = json.load(f)
    with open(store / "bm25_cls.json", "r", encoding="utf-8") as f:
        cls_corpus = json.load(f)

    return {
        "pred": {
            "vectors": pred_pack["vectors"],
            "meta":    list(pred_pack["meta"]),
            "bm25":    BM25(pred_corpus)
        },
        "class": {
            "vectors": cls_pack["vectors"],
            "meta":    list(cls_pack["meta"]),
            "bm25":    BM25(cls_corpus)
        }
    }

def query(store, q: str, target: str = "predicate", k: int = 5, embedder: str = "openai",
          w_vec: float = 0.70, w_bm25: float = 0.25, w_compat: float = 0.05,
          subject_hint: str = "", object_hint: str = ""):
    """Query the hybrid store combining embedding, BM25, and compatibility cues."""
    target = "pred" if target.startswith("pred") else "class"
    pack = store[target]
    meta = pack["meta"]
    X = pack["vectors"]
    bm25 = pack["bm25"]

    # Embed query (optionally enrich with hints)
    emb_fn = get_embedder(embedder)
    full_q = q
    hint_parts = []
    if subject_hint:
        hint_parts.append(f"subject: {subject_hint}")
    if object_hint:
        hint_parts.append(f"object: {object_hint}")
    if hint_parts:
        full_q = q + " | " + " | ".join(hint_parts)

    q_vec = l2_normalize(emb_fn([full_q]))[0]
    sim = cosine_sim(q_vec, X)
    bm = bm25.score(q)

    # Normalize cosine to [0,1]
    sim_min, sim_max = sim.min(), sim.max()
    if sim_max > sim_min:
        sim_norm = (sim - sim_min) / (sim_max - sim_min + 1e-12)
    else:
        sim_norm = sim

    # Compatibility prior (simple: if predicates and we have hints, boost domain/range matches)
    compat = np.zeros_like(sim_norm)
    if target == "pred":
        subj = subject_hint.lower().strip() if subject_hint else ""
        obj  = object_hint.lower().strip() if object_hint else ""
        for i, d in enumerate(meta):
            try:
                doms = [str(x).lower() for x in d.get("domain_class_iris",[])]
                rngs = [str(x).lower() for x in d.get("range_class_iris",[])]
                score = 0.0
                if subj and any(subj in z for z in doms):
                    score += 1.0
                if obj  and any(obj in z for z in rngs):
                    score += 1.0
                compat[i] = min(score, 1.0)
            except Exception:
                compat[i] = 0.0
    # Normalize compat to [0,1] (already is)
    final = w_vec * sim_norm + w_bm25 * bm + w_compat * compat

    top_idx = np.argsort(-final)[:k]
    results = []
    for i in top_idx:
        d = meta[i]
        results.append({
            "rank": len(results)+1,
            "score": float(final[i]),
            "id": d.get("id"),
            "iri": d.get("iri"),
            "label": d.get("label"),
            "aliases": d.get("aliases"),
            "domain_class_iris": d.get("domain_class_iris"),
            "range_class_iris": d.get("range_class_iris"),
            "comment": d.get("comment", ""),
            "embed_text": d.get("embed_text", ""),
            "lexical_text": d.get("lexical_text", "")
        })
    return results

# ------------------ CLI ------------------

def main():
    """Command-line interface for building, querying, or exploring the store."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build vector/BM25 stores from JSON indices")
    b.add_argument("--pred", required=True, help="predicate_index.json")
    b.add_argument("--cls", required=True, help="class_index.json")
    b.add_argument("--out", required=True, help="output directory for store")
    b.add_argument("--embedder", default="openai", choices=["openai","st","sbert","mock"], help="embedding backend")
    b.add_argument("--batch", type=int, default=128)

    q = sub.add_parser("query", help="Query the store")
    q.add_argument("--store", required=True, help="store directory")
    q.add_argument("--type", required=True, choices=["predicate","class"])
    q.add_argument("--q", required=True, help="query text")
    q.add_argument("--k", type=int, default=5)
    q.add_argument("--embedder", default="openai", choices=["openai","st","sbert","mock"])
    q.add_argument("--subject_hint", default="", help="optional subject class IRI substring (for predicates)")
    q.add_argument("--object_hint",  default="", help="optional object class IRI substring (for predicates)")

    r = sub.add_parser("repl", help="Interactive loop")
    r.add_argument("--store", required=True, help="store directory")
    r.add_argument("--type", required=True, choices=["predicate","class"])
    r.add_argument("--embedder", default="openai", choices=["openai","st","sbert","mock"])
    r.add_argument("--k", type=int, default=5)

    args = parser.parse_args()

    if args.cmd == "build":
        build_store(args.pred, args.cls, args.out, embedder=args.embedder, batch=args.batch)
        return

    store = load_store(args.store)

    if args.cmd == "query":
        res = query(store, args.q, target=args.type, k=args.k, embedder=args.embedder,
                    subject_hint=args.subject_hint, object_hint=args.object_hint)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    if args.cmd == "repl":
        print("Type your query (blank to exit).")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            res = query(store, line, target=args.type, k=args.k, embedder=args.embedder)
            for r in res:
                print(f"[{r['rank']:>2}] {r['label']}  (score={r['score']:.3f})  id={r['id']}")
        return

if __name__ == "__main__":
    main()
