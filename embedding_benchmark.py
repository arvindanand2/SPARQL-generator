#!/usr/bin/env python3

"""
Embedding Benchmark for Manufacturing Process Ontology

This script benchmarks embedding providers (OpenAI or Google Gemini) on a
taxonomy-matching task using your manufacturing process dataset.

It measures:
- Top-1 / Top-3 / Top-5 accuracy
- Mean Reciprocal Rank (MRR)
- nDCG@10
- Average cosine margin between correct label and strongest incorrect label

It uses your CSV with columns:
  - Label (required)
  - Also called (optional)
  - Hierarchy (optional)
  - Category (optional)
  - Definition (optional)

It builds evaluation queries from:
  - Label (exact match test)
  - Each synonym in "Also called" (synonym matching test)

USAGE
-----
# OpenAI (text-embedding-3-large) - with API key as argument
python embedding_benchmark.py \
  --csv your_data.csv \
  --provider openai \
  --model text-embedding-3-large \
  --api_key "your-openai-api-key"

# Gemini (text-embedding-004) - with API key as argument
python embedding_benchmark.py \
  --csv your_data.csv \
  --provider gemini \
  --model models/text-embedding-004 \
  --api_key "your-google-api-key"

# Alternative: Use environment variables (original method)
export OPENAI_API_KEY=your-key
python embedding_benchmark.py --csv your_data.csv --provider openai --model text-embedding-3-large

You can also restrict the number of classes or queries for testing:
  --max_labels 200 --max_queries 100

Outputs:
- A metrics JSON summary
- A CSV of per-query retrieval results
- (Optional) a .npz cache of embeddings to speed up repeat runs

Dependencies:
  pip install --upgrade openai google-generativeai pandas numpy tqdm
"""

import os
import re
import csv
import json
import time
import math
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Optional progress bars
try:
    from tqdm import tqdm
except:
    tqdm = lambda x, **k: x

load_dotenv()

# -------------------------
# Utility: text normalization
# -------------------------

def normalize_text(s: str) -> str:
    """Normalize whitespace for comparison-heavy text fields."""
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def split_synonyms(s: str) -> List[str]:
    """Split comma-separated synonyms while preserving order and uniqueness."""
    if not isinstance(s, str) or not s.strip():
        return []
    # split on commas; preserve abbreviations; strip
    parts = [p.strip() for p in s.split(",") if p.strip()]
    # dedupe while preserving order
    seen = set()
    out = []
    for p in parts:
        low = p.lower()
        if low not in seen:
            seen.add(low)
            out.append(p)
    return out

# -------------------------
# Embedding providers
# -------------------------

class OpenAIEmbedder:
    """Adapter around OpenAI's embeddings endpoint with batching."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        """Initialize the SDK client and store model metadata."""
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("Please install openai>=1.0.0: pip install --upgrade openai") from e
        
        # Use provided API key or fall back to environment variable
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()
        self.model = model

    def embed(self, texts: List[str], batch_size: int = 256, sleep: float = 0.0) -> np.ndarray:
        """Embed a list of texts, returning a stacked numpy array."""
        out = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding (OpenAI)"):
            batch = texts[i:i+batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
            out.extend(vecs)
            if sleep: time.sleep(sleep)
        return np.vstack(out)


class GeminiEmbedder:
    """Simple wrapper around Google Gemini's text embedding service."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        """Configure the client using an explicit or environment API key."""
        try:
            import google.generativeai as genai
        except Exception as e:
            raise RuntimeError("Please install google-generativeai: pip install --upgrade google-generativeai") from e
        self.genai = genai
        
        # Use provided API key or fall back to environment variable
        if api_key:
            self.genai.configure(api_key=api_key)
        else:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY is not set and no API key provided")
            self.genai.configure(api_key=api_key)
        self.model = model

    def embed(self, texts: List[str], batch_size: int = 128, sleep: float = 0.0) -> np.ndarray:
        """
        Gemini Embeddings API currently supports one text per call in many SDK versions.
        We'll batch by simple loop; if the SDK adds batch support, you can optimize here.
        """
        vecs = []
        for t in tqdm(texts, desc="Embedding (Gemini)"):
            resp = self.genai.embed_content(model=self.model, content=t)
            if isinstance(resp, dict):
                # New Gemini API: embedding is a direct list of floats
                if "embedding" in resp and isinstance(resp["embedding"], (list, tuple)):
                    v = np.array(resp["embedding"], dtype=np.float32)
                elif "embedding" in resp and "values" in resp["embedding"]:
                    v = np.array(resp["embedding"]["values"], dtype=np.float32)
                elif "data" in resp and len(resp["data"]) and "embedding" in resp["data"][0]:
                    v = np.array(resp["data"][0]["embedding"], dtype=np.float32)
                else:
                    raise RuntimeError(f"Unexpected Gemini embedding response: {resp}")
            else:
                try:
                    v = np.array(resp.embedding.values, dtype=np.float32)
                except Exception as e:
                    raise RuntimeError(f"Cannot parse Gemini response: {type(resp)}") from e
            vecs.append(v)
            if sleep: time.sleep(sleep)
        return np.vstack(vecs)


# -------------------------
# Data prep
# -------------------------

def build_corpus(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Construct canonical label descriptions and ids from the dataset."""
    label_texts = []
    label_ids = []
    for _, row in df.iterrows():
        label = normalize_text(row.get("Label", ""))
        syns = ", ".join(split_synonyms(row.get("Also called", "")))
        cat = normalize_text(row.get("Category", ""))
        hier = normalize_text(row.get("Hierarchy", ""))
        definition = normalize_text(row.get("Definition", ""))

        # Canonical label string (keep concise but informative)
        parts = [f"Label: {label}"]
        if definition:
            parts.append(f"Definition: {definition}")
        if syns:
            parts.append(f"Also called: {syns}")
        if cat:
            parts.append(f"Category: {cat}")
        if hier:
            parts.append(f"Hierarchy: {hier}")
        label_text = " | ".join(parts)
        label_texts.append(label_text)
        label_ids.append(label)
    return label_texts, label_ids


def build_queries(df: pd.DataFrame, max_queries_per_label: int = 5) -> List[Tuple[str, str, str]]:
    """
    Builds a list of (query_text, target_label, source) where source is 'label' or 'synonym'.
    Tests both label performance and synonym matching to correct labels.
    """
    queries = []
    for _, row in df.iterrows():
        label = normalize_text(row.get("Label", ""))
        syns = split_synonyms(row.get("Also called", ""))

        # 1) Use the label as a query
        if label:
            queries.append((label, label, "label"))

        # 2) Use synonyms as queries that should match back to the correct label
        for s in syns:
            queries.append((s, label, "synonym"))

    return queries


# -------------------------
# Metrics
# -------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two embedding sets."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T

def mrr(ranks: List[int]) -> float:
    """Compute mean reciprocal rank for 1-indexed rank positions."""
    return float(np.mean([1.0 / r for r in ranks])) if ranks else 0.0

def ndcg_at_k(ranked_true: List[int], k: int) -> float:
    """
    ranked_true: rank positions (1-indexed) of the true item for each query.
    For a single relevant item, DCG = 1/log2(1+rank), IDCG = 1.
    nDCG@k counts only if rank <= k.
    """
    vals = []
    for r in ranked_true:
        if r <= k:
            vals.append(1.0 / math.log2(1 + r))
        else:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0

# -------------------------
# Benchmark core
# -------------------------

def benchmark(df: pd.DataFrame,
              provider: str,
              model: str,
              api_key: Optional[str] = None,
              max_labels: Optional[int] = None,
              max_queries: Optional[int] = None,
              cache_path: Optional[str] = None) -> Dict:

    """Run the primary taxonomy-matching benchmark against a provider."""

    # Trim dataset (optional)
    if max_labels is not None:
        df = df.head(max_labels).copy()

    # Build label corpus and queries
    label_texts, label_ids = build_corpus(df)
    queries = build_queries(df)
    if max_queries is not None:
        queries = queries[:max_queries]

    # Select provider
    if provider.lower() == "openai":
        embedder = OpenAIEmbedder(model=model, api_key=api_key)
    elif provider.lower() == "gemini":
        embedder = GeminiEmbedder(model=model, api_key=api_key)
    else:
        raise ValueError("provider must be 'openai' or 'gemini'")

    # Cache setup
    cache = {}
    if cache_path and os.path.exists(cache_path):
        try:
            cache = dict(np.load(cache_path, allow_pickle=True))
        except Exception:
            cache = {}

    def get_cached(key: str):
        """Return cached array for the given key if available."""
        return cache.get(key, None)

    def set_cached(key: str, arr: np.ndarray):
        """Store an array in the in-memory embedding cache."""
        cache[key] = arr

    # Embed label corpus
    label_key = f"labels::{provider}::{model}::{len(label_texts)}"
    label_embs = get_cached(label_key)
    if label_embs is None:
        label_embs = embedder.embed(label_texts)
        set_cached(label_key, label_embs)

    # Embed queries
    query_texts = [q[0] for q in queries]
    query_key = f"queries::{provider}::{model}::{len(query_texts)}"
    query_embs = get_cached(query_key)
    if query_embs is None:
        query_embs = embedder.embed(query_texts)
        set_cached(query_key, query_embs)

    # Save cache if requested
    if cache_path:
        np.savez_compressed(cache_path, **cache)

    # Compute similarity & rankings
    S = cosine_sim(query_embs, label_embs)  # [Q, L]
    results = []
    ranks = []
    top1_correct = top3_correct = top5_correct = 0

    label_to_index = {lab: i for i, lab in enumerate(label_ids)}

    for i, (qtext, target, source) in enumerate(queries):
        sims = S[i]
        order = np.argsort(-sims)  # descending
        ranked_labels = [label_ids[j] for j in order]
        try:
            true_idx = label_to_index[target]
            rank_pos = int(np.where(order == true_idx)[0][0]) + 1  # 1-indexed
        except Exception:
            rank_pos = 999999  # not found (shouldn't happen)

        ranks.append(rank_pos)
        if rank_pos == 1: top1_correct += 1
        if rank_pos <= 3: top3_correct += 1
        if rank_pos <= 5: top5_correct += 1

        # Margin: true score minus best incorrect
        true_score = sims[true_idx] if true_idx < len(sims) else float("-inf")
        best_incorrect = max([sims[j] for j in order if j != true_idx][:1] or [float("-inf")])
        margin = float(true_score - best_incorrect) if np.isfinite(true_score) and np.isfinite(best_incorrect) else float("nan")

        results.append({
            "query": qtext,
            "source": source,
            "target_label": target,
            "rank": int(rank_pos),
            "top1_pred": ranked_labels[0],
            "top1_score": float(sims[order[0]]),
            "true_score": float(true_score),
            "margin_vs_next": margin
        })

    n = len(queries)
    metrics = {
        "provider": provider,
        "model": model,
        "num_labels": len(label_ids),
        "num_queries": n,
        "top1_acc": round(top1_correct / n, 4) if n else 0.0,
        "top3_acc": round(top3_correct / n, 4) if n else 0.0,
        "top5_acc": round(top5_correct / n, 4) if n else 0.0,
        "mrr": round(mrr(ranks), 4) if n else 0.0,
        "ndcg@10": round(ndcg_at_k(ranks, 10), 4) if n else 0.0,
        "avg_margin": round(float(np.nanmean([r["margin_vs_next"] for r in results])) , 4) if n else 0.0
    }

    return metrics, pd.DataFrame(results)


def query_to_label_benchmark(label_df, query_df, provider, model, api_key=None, cache_path=None):
    """Evaluate retrieval when queries map to expected labels via CSV inputs."""
    # Build label corpus and embeddings
    label_texts = [normalize_text(row.get("Label", "")) for _, row in label_df.iterrows()]
    label_ids = [normalize_text(row.get("Label", "")) for _, row in label_df.iterrows()]
    if provider.lower() == "openai":
        embedder = OpenAIEmbedder(model=model, api_key=api_key)
    elif provider.lower() == "gemini":
        embedder = GeminiEmbedder(model=model, api_key=api_key)
    else:
        raise ValueError("provider must be 'openai' or 'gemini'")
    # Embed labels
    label_embs = embedder.embed(label_texts)
    # Embed queries
    query_texts = [normalize_text(row.get("query", "")) for _, row in query_df.iterrows()]
    expected_labels = [normalize_text(row.get("expected_label", "")) for _, row in query_df.iterrows()]
    query_embs = embedder.embed(query_texts)
    # Compute similarity & rankings
    S = cosine_sim(query_embs, label_embs)  # [Q, L]
    results = []
    ranks = []
    top1_correct = top3_correct = top5_correct = 0
    label_to_index = {lab: i for i, lab in enumerate(label_ids)}
    for i, (qtext, expected) in enumerate(zip(query_texts, expected_labels)):
        sims = S[i]
        order = np.argsort(-sims)  # descending
        ranked_labels = [label_ids[j] for j in order]
        true_idx = label_to_index.get(expected, None)
        if true_idx is not None:
            rank_pos = int(np.where(order == true_idx)[0][0]) + 1  # 1-indexed
        else:
            rank_pos = 999999  # not found
        ranks.append(rank_pos)
        if rank_pos == 1: top1_correct += 1
        if rank_pos <= 3: top3_correct += 1
        if rank_pos <= 5: top5_correct += 1
        results.append({
            "query": qtext,
            "expected_label": expected,
            "rank": int(rank_pos),
            "top1_pred": ranked_labels[0],
            "top1_score": float(sims[order[0]]),
        })
    n = len(query_texts)
    metrics = {
        "provider": provider,
        "model": model,
        "num_labels": len(label_ids),
        "num_queries": n,
        "top1_acc": round(top1_correct / n, 4) if n else 0.0,
        "top3_acc": round(top3_correct / n, 4) if n else 0.0,
        "top5_acc": round(top5_correct / n, 4) if n else 0.0,
        "mrr": round(mrr(ranks), 4) if n else 0.0,
    }
    return metrics, pd.DataFrame(results)


def main():
    """Parse CLI args, execute benchmark, and persist metrics/results."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to the process CSV (labels)")
    ap.add_argument("--provider", required=True, choices=["openai", "gemini"])
    ap.add_argument("--model", required=True, help="Embedding model id")
    ap.add_argument("--api_key", default=None, help="API key for the provider (OpenAI or Google)")
    ap.add_argument("--max_labels", type=int, default=None)
    ap.add_argument("--max_queries", type=int, default=None)
    ap.add_argument("--cache", default=None, help="Optional path to .npz cache for embeddings")
    ap.add_argument("--out_prefix", default="benchmark_out", help="Prefix for output files")
    ap.add_argument("--query_csv", default=None, help="CSV with queries and expected labels for query-to-label benchmark")
    args = ap.parse_args()

    label_df = pd.read_csv(args.csv)
    label_df.columns = [c.strip() for c in label_df.columns]
    if args.query_csv:
        query_df = pd.read_csv(args.query_csv)
        query_df.columns = [c.strip() for c in query_df.columns]
        metrics, results_df = query_to_label_benchmark(
            label_df=label_df,
            query_df=query_df,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            cache_path=args.cache
        )
        meta_path = f"{args.out_prefix}.{args.provider}.query.json"
        results_path = f"{args.out_prefix}.{args.provider}.query.results.csv"
        with open(meta_path, "w") as f:
            json.dump(metrics, f, indent=2)
        results_df.to_csv(results_path, index=False)
        print("=== QUERY BENCHMARK METRICS ===")
        print(json.dumps(metrics, indent=2))
        print(f"\nSaved: {meta_path}")
        print(f"Saved: {results_path}")
    else:
        metrics, results_df = benchmark(
            df=label_df,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            max_labels=args.max_labels,
            max_queries=args.max_queries,
            cache_path=args.cache
        )
        meta_path = f"{args.out_prefix}.{args.provider}.json"
        results_path = f"{args.out_prefix}.{args.provider}.results.csv"
        with open(meta_path, "w") as f:
            json.dump(metrics, f, indent=2)
        results_df.to_csv(results_path, index=False)
        print("=== METRICS ===")
        print(json.dumps(metrics, indent=2))
        print(f"\nSaved: {meta_path}")
        print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
