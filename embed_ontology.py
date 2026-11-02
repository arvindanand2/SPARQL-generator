#!/usr/bin/env python3
"""
Embed ontology classes & predicates using OpenAI embeddings.

Requirements:
  pip install openai numpy tqdm

Environment:
  export OPENAI_API_KEY="..."
  # Optional:
  export OPENAI_BASE_URL="..."   # if using a compatible endpoint/proxy

Inputs (default paths):
  - classes_ready.jsonl (expects field 'embedding_text_enriched')
  - predicates_ready.jsonl (expects field 'embedding_text_enriched')

Outputs (default paths):
  - classes_vectors.npy
  - predicates_vectors.npy
  - classes_manifest.json
  - predicates_manifest.json

Usage:
  python embed_ontology.py \
    --classes /path/to/classes_ready.jsonl \
    --predicates /path/to/predicates_ready.jsonl \
    --out-dir ./artifacts \
    --model text-embedding-3-large \
    --batch-size 64

Notes:
  - We L2-normalize all vectors so dot == cosine similarity.
  - We include a small resume cache to avoid re-embedding unchanged records.
"""

import os, json, argparse, hashlib
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# OpenAI SDK (v1+)
from openai import OpenAI
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment. Please add it to your .env file.")

client = OpenAI(api_key=api_key)
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON Lines file into a list of dictionaries.

    Parameters
    ----------
    path : Path
        Path to the JSONL artifact to load.

    Returns
    -------
    list[dict]
        Parsed JSON objects for each non-empty line.
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def sha1_text(s: str) -> str:
    """Return a SHA-1 digest for the given text using UTF-8 encoding."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Normalize rows so cosine similarity equals dot product."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

def embed_texts_openai(texts: List[str], model: str, client: OpenAI, batch_size: int = 64) -> np.ndarray:
    """Batch through OpenAI embeddings API and return L2-normalized vectors.

    Parameters
    ----------
    texts : list[str]
        Input strings to embed.
    model : str
        Embedding model identifier.
    client : OpenAI
        Initialized OpenAI client.
    batch_size : int, default 64
        Number of texts to embed per request.

    Returns
    -------
    numpy.ndarray
        Array of normalized embeddings.
    """
    all_vecs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({model})", unit="batch"):
        chunk = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=chunk)
        vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
        all_vecs.extend(vecs)
    X = np.stack(all_vecs)
    return l2_normalize(X)

def ensure_outdir(p: Path):
    """Create the output directory hierarchy if it is absent."""
    p.mkdir(parents=True, exist_ok=True)

def build_cache_key(rec: Dict[str, Any], field: str) -> str:
    """Construct a stable hash from record identity and embedding text."""
    base = rec.get("iri") or rec.get("id") or rec.get("label","")
    text = rec.get(field, "") or ""
    return sha1_text(base + "||" + text)

def maybe_resume(existing_manifest: Path, records: List[Dict[str,Any]], field: str):
    """Inspect manifests to determine which records still need embedding.

    Returns indices, texts, cached vectors, and a cache lookup map for reuse.
    """
    cached = {}
    X_cached = None
    if existing_manifest.exists():
        manifest = json.load(existing_manifest.open("r", encoding="utf-8"))
        meta = manifest.get("meta", {})
        keys = manifest.get("cache_keys", [])
        # Try to load matching vectors if file exists
        vec_path = manifest.get("vectors_path")
        if vec_path and Path(vec_path).exists():
            X_cached = np.load(vec_path)
            # Map cache_key -> index in existing array
            for idx, key in enumerate(keys):
                cached[key] = idx

    pending_idx = []
    pending_texts = []
    for i, rec in enumerate(records):
        key = build_cache_key(rec, field)
        if key not in cached:
            pending_idx.append(i)
            pending_texts.append(rec.get(field, "") or "")
    return pending_idx, pending_texts, X_cached, cached

def write_manifest(path: Path, *, model: str, vectors_path: str, records_path: str, count: int, dim: int, cache_keys: List[str]):
    """Persist metadata describing the embedding vectors and cache keys."""
    data = {
        "model": model,
        "vectors_path": vectors_path,
        "records_path": records_path,
        "count": count,
        "dim": dim,
        "cache_keys": cache_keys,
        "meta": {}
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def process_split(client: OpenAI, records_path: Path, out_dir: Path, model: str, field: str, out_prefix: str, batch_size: int):
    """Embed a JSONL split, merging resume caches and writing manifests.

    Parameters
    ----------
    client : OpenAI
        Initialized SDK client for embedding calls.
    records_path : Path
        Path to the JSONL file to embed.
    out_dir : Path
        Directory where vectors and manifests should be written.
    model : str
        Embedding model identifier.
    field : str
        Record field containing the text to embed.
    out_prefix : str
        Prefix used for output filenames.
    batch_size : int
        Number of records to embed per API request.
    """
    records = read_jsonl(records_path)
    vectors_path = out_dir / f"{out_prefix}_vectors.npy"
    manifest_path = out_dir / f"{out_prefix}_manifest.json"

    # Resume support
    pending_idx, pending_texts, X_cached, cached_map = maybe_resume(manifest_path, records, field)

    # Build full list of cache_keys (current file state)
    cache_keys = [build_cache_key(r, field) for r in records]

    if len(pending_idx) == 0 and X_cached is not None and X_cached.shape[0] == len(records):
        print(f"[{out_prefix}] Nothing to do; vectors already up-to-date:", vectors_path)
        return

    # Embed pending texts
    if pending_texts:
        X_new = embed_texts_openai(pending_texts, model, client, batch_size=batch_size)
    else:
        X_new = np.zeros((0, X_cached.shape[1] if X_cached is not None else 0), dtype=np.float32)

    # Merge cached + new into final array in original order
    if X_cached is not None and X_cached.shape[0] >= 1:
        dim = X_cached.shape[1]
    else:
        dim = X_new.shape[1] if X_new.size else None

    if dim is None:
        raise RuntimeError(f"No vectors to write for {out_prefix}; check input texts.")

    X_final = np.zeros((len(records), dim), dtype=np.float32)
    # Fill from cache
    if X_cached is not None:
        for i, rec in enumerate(records):
            key = cache_keys[i]
            if key in cached_map:
                X_final[i] = X_cached[cached_map[key]]
    # Fill from new embeddings
    j = 0
    for i in range(len(records)):
        key = cache_keys[i]
        if key not in cached_map:
            X_final[i] = X_new[j]
            j += 1

    # Save outputs
    np.save(vectors_path, X_final)
    write_manifest(manifest_path, model=model, vectors_path=str(vectors_path), records_path=str(records_path),
                   count=X_final.shape[0], dim=X_final.shape[1], cache_keys=cache_keys)
    print(f"[{out_prefix}] Wrote {X_final.shape} to {vectors_path}")
    print(f"[{out_prefix}] Manifest: {manifest_path}")

def main():
    """CLI entrypoint to embed class and predicate JSONL datasets."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=str, default="classes_ready.jsonl", help="Path to classes JSONL")
    ap.add_argument("--predicates", type=str, default="predicates_ready.jsonl", help="Path to predicates JSONL")
    ap.add_argument("--out-dir", type=str, default="./artifacts", help="Output directory")
    ap.add_argument("--model", type=str, default="text-embedding-3-large", help="OpenAI embedding model")
    ap.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Create OpenAI client (respects OPENAI_API_KEY and optional OPENAI_BASE_URL)
    client = OpenAI()

    # Process classes
    process_split(client, Path(args.classes), out_dir, args.model, field="embedding_text_enriched", out_prefix="classes", batch_size=args.batch_size)

    # Process predicates
    process_split(client, Path(args.predicates), out_dir, args.model, field="embedding_text_enriched", out_prefix="predicates", batch_size=args.batch_size)

if __name__ == "__main__":
    main()
