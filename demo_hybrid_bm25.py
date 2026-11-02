#!/usr/bin/env python3
"""
demo_hybrid_bm25.py
-------------------
Run a few demo queries against the hybrid BM25+cosine detector and print the semantic sketches.

Usage:
  python demo_hybrid_bm25.py --artifacts ./artifacts --model text-embedding-3-large

Notes:
  - Requires: openai, python-dotenv, numpy, rank-bm25, spacy (and "python -m spacy download en_core_web_sm")
  - Your artifacts directory must contain the *_manifest.json + *_vectors.npy we generated earlier.
"""

import argparse, json, sys
from hybrid_detector_hybridbm25 import HybridDetector

DEFAULT_QUERIES = [
    # 1) Core manufacturer→industry relation
    "List manufacturers serving the sports and leisure industry by count",
    # 2) Geospatial relation (city/state)
    "Show manufacturers located in the city of Phoenix, Arizona",
    # 3) NAICS code-based filter
    "Find organizations with NAICS code 333 that supply the entertainment and media industry",
]

def run_queries(artifacts: str, model: str, queries):
    """Execute a set of natural language queries against the hybrid detector.

    Parameters
    ----------
    artifacts : str
        Filesystem path containing manifests and embedding vectors.
    model : str
        Embedding model identifier used for OpenAI calls.
    queries : Sequence[str]
        Natural language queries that will be evaluated.
    """
    hd = HybridDetector(artifacts_dir=artifacts, model=model,use_bm25=False)
    for q in queries:
        print("="*80)
        print("QUERY:", q)
        sketch = hd.detect_and_map(q)
        print(json.dumps(sketch, indent=2))

        # minimal sanity checks
        classes = sketch.get("classes", [])
        preds   = sketch.get("predicates", [])
        if not classes:
            print("[WARN] No classes detected for query:", q, file=sys.stderr)
        if not preds:
            print("[WARN] No predicates detected for query:", q, file=sys.stderr)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, default="./artifacts", help="Path to artifacts directory")
    ap.add_argument("--model", type=str, default="text-embedding-3-large", help="Embedding model")
    ap.add_argument("--queries", type=str, nargs="*", default=None, help="Override demo queries")
    args = ap.parse_args()

    queries = args.queries if args.queries else DEFAULT_QUERIES
    run_queries(args.artifacts, args.model, queries)
