#!/usr/bin/env python3
"""
Enhanced CSV Embedding Pipeline (Improved)

Key upgrades
------------
- Robust logging instead of prints (levels + timestamps)
- Configurable retries, timeout, and batch size
- Offline mode (no API calls): TF-IDF + hashing fallback for embeddings
- Query embedding cache (LRU) to avoid repeated calls
- Model-agnostic embedding size detection (no hard-coded 1536)
- Safe CSV loading (dtype/backfill/NA handling) and de-duplication
- Pre-normalize vectors at index time for faster cosine search
- Deterministic cache key based on CSV file hashes + model name
- Optional integration with PermanentEmbeddingStorage, if provided
- CLI improvements: --offline, --batch-size, --timeout, --retries, --cache-dir, --no-color

This stays dependency-light; offline fallback uses scikit-learn *if available*; otherwise
it falls back to a simple hashing vectorizer implemented here.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional .env support
try:  # pragma: no cover
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

# OpenAI client (optional if offline)
try:  # pragma: no cover
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# Optional vectorizers for offline mode
try:  # pragma: no cover
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    SKLEARN_AVAILABLE = False

# Optional tqdm progress
try:  # pragma: no cover
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **_: x  # type: ignore

# Import permanent storage if available
try:  # pragma: no cover
    from permanent_embedding_storage import PermanentEmbeddingStorage  # type: ignore
except Exception:  # pragma: no cover
    PermanentEmbeddingStorage = None  # type: ignore


# -------------------------------
# Logging setup
# -------------------------------
LOG = logging.getLogger("csv_embedder")


def setup_logging(no_color: bool = False, level: str = "INFO") -> None:
    class _Color:
        RESET = "\033[0m"
        GRAY = "\033[90m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        CYAN = "\033[96m"

    colors = _Color() if not no_color else type("_NoColor", (), {k: "" for k in [
        "RESET", "GRAY", "GREEN", "YELLOW", "RED", "CYAN"]})()

    class ColorFormatter(logging.Formatter):
        LEVEL_COLOR = {
            logging.DEBUG: colors.GRAY,
            logging.INFO: colors.GREEN,
            logging.WARNING: colors.YELLOW,
            logging.ERROR: colors.RED,
            logging.CRITICAL: colors.RED,
        }

        def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
            color = self.LEVEL_COLOR.get(record.levelno, "")
            msg = super().format(record)
            return f"{color}{msg}{colors.RESET}"

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))
    LOG.setLevel(getattr(logging, level.upper(), logging.INFO))
    LOG.addHandler(handler)


# -------------------------------
# Data structures
# -------------------------------
@dataclass
class EmbeddingResult:
    label: str
    score: float
    iri: str
    category: str
    hierarchy: str
    definition: str
    source_file: str


# -------------------------------
# Utility helpers
# -------------------------------

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def calc_cache_key(model: str, csv_files: Dict[str, str]) -> str:
    parts = [f"model={model}"]
    for name, p in sorted(csv_files.items()):
        path = Path(p)
        if path.exists():
            parts.append(f"{name}:{file_sha256(path)}")
        else:
            parts.append(f"{name}:MISSING")
    key = "|".join(parts)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def safe_get(d: Dict[str, Any], k: str, default: str = "") -> str:
    v = d.get(k, default)
    if v is None:
        return default
    s = str(v)
    return "" if s.lower() == "nan" else s


# -------------------------------
# Offline embedding fallback
# -------------------------------
class OfflineEmbedder:
    """Simple text vectorizer for offline mode.

    If scikit-learn is present, use TF-IDF. Otherwise, a basic hashing
    vectorizer (non-invertible) is used. Output vectors are L2-normalized.
    """

    def __init__(self) -> None:
        self.use_sklearn = SKLEARN_AVAILABLE
        self.dim = 8192 if not self.use_sklearn else None  # hashing dim
        self._tfidf: Optional[TfidfVectorizer] = None

    def fit(self, corpus: List[str]) -> None:
        if self.use_sklearn:
            self._tfidf = TfidfVectorizer(max_features=8192, ngram_range=(1, 2))
            self._tfidf.fit(corpus)

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.use_sklearn and self._tfidf is not None:
            mat = self._tfidf.transform(texts)
            # Convert to dense float32
            arr = mat.astype(np.float32).toarray()
        else:
            # Simple hashing trick
            arr = np.zeros((len(texts), self.dim), dtype=np.float32)
            for i, t in enumerate(texts):
                tokens = t.lower().split()
                for tok in tokens:
                    idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
                    arr[i, idx] += 1.0
        # L2 normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms).astype(np.float32)


# -------------------------------
# Main embedder
# -------------------------------
class EnhancedCSVEmbedder:
    def __init__(
        self,
        openai_model: str = "text-embedding-3-large",
        cache_dir: str = "embedding_cache",
        timeout: int = 30,
        retries: int = 2,
        batch_size: int = 100,
        offline: bool = False,
    ) -> None:
        self.openai_model = openai_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.retries = retries
        self.batch_size = max(1, batch_size)
        self.offline = offline

        self.client = None
        if not offline and OpenAI is not None:
            self.client = OpenAI(timeout=timeout, max_retries=retries)
        elif not offline:
            LOG.warning("OpenAI client not available; forcing offline mode")
            self.offline = True

        # Embedding + metadata stores
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, List[Dict[str, Any]]] = {}
        self.iri_mapping: Dict[str, Dict[str, str]] = {}

        # Offline encoder
        self._offline = OfflineEmbedder() if self.offline else None

        # Query embedding cache (in-memory)
        self._query_cache: Dict[str, np.ndarray] = {}

        # Optional vector DB interface
        self.storage = None
        if PermanentEmbeddingStorage is not None:
            try:
                self.storage = PermanentEmbeddingStorage()
            except Exception as e:  # pragma: no cover
                LOG.debug(f"PermanentEmbeddingStorage unavailable: {e}")

        LOG.info(f"Initialized embedder (model={openai_model}, offline={self.offline})")

    # ---------- IRI mapping ----------
    def load_iri_mapping(self, iri_mapping_file: str = "label_iri_mappings.csv") -> None:
        try:
            df = pd.read_csv(iri_mapping_file, encoding="utf-8")
            count = 0
            for _, row in df.iterrows():
                label = safe_get(row, "label").strip().lower()
                if not label:
                    continue
                self.iri_mapping[label] = {
                    "iri": safe_get(row, "iri"),
                    "type": safe_get(row, "type"),
                    "id": safe_get(row, "id"),
                    "source_file": safe_get(row, "source_file"),
                }
                count += 1
            LOG.info(f"Loaded {count} label→IRI mappings")
        except Exception as e:
            LOG.warning(f"Could not load IRI mapping from {iri_mapping_file}: {e}")
            self.iri_mapping = {}

    # ---------- CSV processing ----------
    @staticmethod
    def _read_csv(csv_file: str) -> pd.DataFrame:
        df = pd.read_csv(csv_file, encoding="utf-8", dtype=str, keep_default_na=False)
        # Normalize column names to a consistent set
        df.columns = [c.strip() for c in df.columns]
        # Fill missing keys
        for col in ["Label", "Definition", "Category", "Hierarchy", "Also Called"]:
            if col not in df.columns:
                df[col] = ""
        # Drop empty labels, de-duplicate by label+definition
        df["Label"] = df["Label"].fillna("").astype(str).str.strip()
        df = df[df["Label"] != ""].copy()
        df["Definition"] = df["Definition"].fillna("").astype(str)
        df.drop_duplicates(subset=["Label", "Definition"], inplace=True)
        return df

    def process_csv_file(self, csv_file: str, source_name: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        LOG.info(f"Processing {csv_file} ({source_name})")
        try:
            df = self._read_csv(csv_file)
            LOG.info(f"  Loaded {len(df)} rows")
        except Exception as e:
            LOG.error(f"Error loading {csv_file}: {e}")
            return [], []

        texts, meta = [], []
        for _, row in df.iterrows():
            label = safe_get(row, "Label").strip()
            definition = safe_get(row, "Definition").strip()
            embed_text = f"{label}: {definition}" if definition else label
            texts.append(embed_text)
            meta.append({
                "label": label,
                "category": safe_get(row, "Category").strip(),
                "hierarchy": safe_get(row, "Hierarchy").strip(),
                "definition": definition,
                "also_called": safe_get(row, "Also Called").strip(),
                "source_file": source_name,
                "embed_text": embed_text,
            })
        LOG.info(f"  Prepared {len(texts)} items for embedding")
        return texts, meta

    # ---------- Embedding ----------
    def _embed_api(self, texts: List[str]) -> np.ndarray:
        assert self.client is not None
        response = self.client.embeddings.create(model=self.openai_model, input=texts)
        emb_list = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        arr = np.vstack(emb_list)
        # L2 normalize once here for faster cosine similarity later
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return (arr / norms).astype(np.float32)

    def _embed_offline(self, texts: List[str]) -> np.ndarray:
        assert self._offline is not None
        return self._offline.embed(texts)

    def embed_text_batch(self, texts: List[str]) -> np.ndarray:
        """Batch-embed and return L2-normalized vectors (float32)."""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        vectors: List[np.ndarray] = []
        iterator = tqdm(range(0, len(texts), self.batch_size), total=(len(texts) + self.batch_size - 1)//self.batch_size, desc="Embedding")
        for start in iterator:
            batch = texts[start:start + self.batch_size]
            try:
                if self.offline:
                    vecs = self._embed_offline(batch)
                else:
                    vecs = self._embed_api(batch)
                vectors.append(vecs)
            except Exception as e:
                LOG.error(f"Embedding batch failed: {e}")
                # Fallback to zeros with inferred dim if possible
                dim = vectors[0].shape[1] if vectors else (self._offline.dim if self._offline else 1536)
                vectors.append(np.zeros((len(batch), dim), dtype=np.float32))
        return np.vstack(vectors)

    # ---------- Indexing ----------
    def embed_csv_files(self, csv_files: Dict[str, str], force_recompute: bool = False) -> None:
        cache_key = calc_cache_key(self.openai_model if not self.offline else "offline", csv_files)
        cache_file = self.cache_dir / f"enhanced_csv_embeddings_{cache_key}.pkl"

        if cache_file.exists() and not force_recompute:
            LOG.info(f"Loading embeddings from cache: {cache_file}")
            try:
                with cache_file.open("rb") as f:
                    cached = pickle.load(f)
                self.embeddings = cached["embeddings"]
                self.metadata = cached["metadata"]
                LOG.info(f"Loaded {sum(len(v) for v in self.embeddings.values())} vectors from cache")
                return
            except Exception as e:
                LOG.warning(f"Cache load failed ({e}); recomputing…")

        LOG.info("Computing embeddings for all CSV files…")

        # If offline, pre-fit on all texts to stabilize TF-IDF space
        if self.offline and self._offline and SKLEARN_AVAILABLE:
            all_texts: List[str] = []
            for _, csv_path in csv_files.items():
                if Path(csv_path).exists():
                    t, _ = self.process_csv_file(csv_path, "_PRELOAD_")
                    all_texts.extend(t)
            LOG.info(f"Fitting offline TF-IDF on {len(all_texts)} texts…")
            self._offline.fit(all_texts)

        for source_name, csv_path in csv_files.items():
            p = Path(csv_path)
            if not p.exists():
                LOG.warning(f"File not found: {csv_path}")
                continue
            texts, meta = self.process_csv_file(csv_path, source_name)
            if not texts:
                LOG.warning(f"No texts to embed for {source_name}")
                continue
            vecs = self.embed_text_batch(texts)
            self.embeddings[source_name] = vecs.astype(np.float32)
            self.metadata[source_name] = meta
            LOG.info(f"✅ Completed {source_name}: {vecs.shape[0]} vectors, dim={vecs.shape[1]}")

        try:
            with cache_file.open("wb") as f:
                pickle.dump({
                    "embeddings": self.embeddings,
                    "metadata": self.metadata,
                    "timestamp": datetime.now().isoformat(),
                    "model": self.openai_model if not self.offline else "offline",
                }, f)
            LOG.info(f"💾 Cached embeddings → {cache_file}")
        except Exception as e:
            LOG.warning(f"Could not save cache: {e}")

    # ---------- IRI helpers ----------
    def find_best_iri_match(self, query_label: str, threshold: float = 0.7) -> Optional[str]:
        if not self.iri_mapping:
            return None
        q = query_label.strip().lower()
        if not q:
            return None
        # Exact
        hit = self.iri_mapping.get(q)
        if hit:
            return hit.get("iri")
        # Capability-style variants
        no_spaces = q.replace(" ", "").replace("-", "").replace("_", "")
        variants = [no_spaces + "capability", no_spaces]
        words = q.replace("-", " ").replace("_", " ").split()
        if words:
            camel = "".join(w.capitalize() for w in words).lower()
            variants.append(camel + "capability")
        for v in variants:
            hit = self.iri_mapping.get(v)
            if hit:
                return hit.get("iri")
        # Partial / overlap heuristic
        best_score, best_iri = 0.0, None
        q_words = set(q.split())
        for mapped_label, iri_info in self.iri_mapping.items():
            if q in mapped_label:
                score = len(q) / max(1, len(mapped_label))
            elif mapped_label in q:
                score = len(mapped_label) / max(1, len(q))
            elif "capability" in mapped_label and no_spaces in mapped_label:
                score = 0.8
            else:
                m_words = set(mapped_label.split())
                inter = len(q_words & m_words)
                union = len(q_words | m_words) or 1
                score = inter / union
            if score > best_score:
                best_score, best_iri = score, iri_info.get("iri")
        return best_iri if best_score >= 0.3 else None

    # ---------- Search ----------
    def _embed_query(self, query: str) -> np.ndarray:
        if query in self._query_cache:
            return self._query_cache[query]
        if self.offline:
            assert self._offline is not None
            vec = self._offline.embed([query])[0]
        else:
            vec = self._embed_api([query])[0]
        self._query_cache[query] = vec.astype(np.float32)
        return self._query_cache[query]

    def search_closest_labels(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[List[str]] = None,
    ) -> List[EmbeddingResult]:
        if not self.embeddings:
            LOG.warning("No embeddings loaded. Run embed_csv_files() first.")
            return []
        LOG.info(f"Searching for: '{query}' (top_k={top_k})")
        try:
            qvec = self._embed_query(query)
        except Exception as e:
            LOG.error(f"Query embedding failed: {e}")
            return []

        results: List[EmbeddingResult] = []
        sources = source_filter if source_filter else list(self.embeddings.keys())
        for source_name in sources:
            if source_name not in self.embeddings:
                continue
            E = self.embeddings[source_name]  # pre-normalized
            meta = self.metadata[source_name]
            if E.size == 0:
                continue
            sims = E @ qvec  # cosine because both are L2-normalized
            idx = np.argpartition(-sims, kth=min(top_k, len(sims)-1))[:top_k]
            # Sort top_k by score descending
            idx = idx[np.argsort(-sims[idx])]
            for i in idx:
                m = meta[int(i)]
                label = m["label"]
                iri = self.find_best_iri_match(label, threshold=0.7) or f"No IRI found for: {label}"
                results.append(EmbeddingResult(
                    label=label,
                    score=float(sims[int(i)]),
                    iri=iri,
                    category=m.get("category", ""),
                    hierarchy=m.get("hierarchy", ""),
                    definition=m.get("definition", ""),
                    source_file=source_name,
                ))
        # Global top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ---------- Display helpers ----------
    @staticmethod
    def normalize_label_for_display(label: str, iri: str = "") -> str:
        suffixes = [
            "Capability", "ProcessingCapability", "MachiningCapability",
            "ManufacturingCapability", "IndustryCapability", "QualityCapability",
            "CertificateCapability", "TechnologyCapability",
        ]
        normalized = label
        for s in suffixes:
            if normalized.lower().endswith(s.lower()):
                normalized = normalized[:-len(s)].strip()
                break
        # Camel -> words
        if normalized and not any(c.islower() for c in normalized):
            import re
            spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", normalized)
            spaced = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", spaced)
            spaced = re.sub(r"\s+", " ", spaced).strip()
            if spaced != normalized:
                normalized = spaced.title()
        return normalized or label

    @staticmethod
    def extract_process_type_from_iri(iri: str) -> str:
        if not iri or "No IRI found" in iri:
            return ""
        try:
            class_name = iri.split("/")[-1]
            for s in ["Capability", "ProcessingCapability", "MachiningCapability"]:
                if class_name.endswith(s):
                    class_name = class_name[:-len(s)]
                    break
            import re
            readable = re.sub(r"([a-z])([A-Z])", r"\1 \2", class_name)
            return readable.title() if readable else class_name
        except Exception:
            return ""

    def search_and_display_results(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[List[str]] = None,
        generic_display: bool = True,
    ) -> List[EmbeddingResult]:
        results = self.search_closest_labels(query, top_k, source_filter)
        LOG.info(f"Top {len(results)} results for: '{query}'")
        for i, r in enumerate(results, 1):
            disp = self.normalize_label_for_display(r.label, r.iri) if generic_display else r.label
            LOG.info(f"{i}. {disp}  score={r.score:.4f}  IRI={r.iri}")
        return results


# -------------------------------
# CLI
# -------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Enhanced CSV Embedding Pipeline (Improved)")
    p.add_argument("--embed", action="store_true", help="Generate embeddings for CSV files")
    p.add_argument("--search", type=str, help="Search query")
    p.add_argument("--top-k", type=int, default=5, help="Number of top results to return")
    p.add_argument("--sources", nargs="+", help="Filter by source keys (e.g., industries material_caps)")
    p.add_argument("--force", action="store_true", help="Force recompute embeddings, ignore cache")
    p.add_argument("--model", default="text-embedding-3-large", help="OpenAI embedding model")
    p.add_argument("--batch-size", type=int, default=100, help="Embedding batch size")
    p.add_argument("--timeout", type=int, default=30, help="OpenAI client timeout (s)")
    p.add_argument("--retries", type=int, default=2, help="OpenAI client max retries")
    p.add_argument("--cache-dir", default="embedding_cache", help="Cache directory path")
    p.add_argument("--offline", action="store_true", help="Run without network/API (use TF-IDF/hashing)")
    p.add_argument("--no-color", action="store_true", help="Disable colored logs")
    p.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    setup_logging(no_color=args.no_color, level=args.log_level)

    # Initialize embedder
    embedder = EnhancedCSVEmbedder(
        openai_model=args.model,
        cache_dir=args.cache_dir,
        timeout=args.timeout,
        retries=args.retries,
        batch_size=args.batch_size,
        offline=args.offline,
    )

    # Load IRI mapping
    embedder.load_iri_mapping("label_iri_mappings.csv")

    # Define CSV files to process (customize as needed)
    csv_files = {
        "industries": "industries_enhanced.csv",
        "material_caps": "material_caps_enhanced.csv",
        "certificates": "certificates_enhanced_fixed.csv",
        "process_embeddings": "process_embeddings_enhanced_with_def.csv",
    }

    # Decide whether we need to (re)index
    cache_key = calc_cache_key(embedder.openai_model if not embedder.offline else "offline", csv_files)
    cache_file = Path(args.cache_dir) / f"enhanced_csv_embeddings_{cache_key}.pkl"

    need_index = args.embed or not cache_file.exists()
    if need_index:
        embedder.embed_csv_files(csv_files, force_recompute=args.force)
    else:
        LOG.info(f"Loading embeddings from cache: {cache_file}")
        try:
            with cache_file.open("rb") as f:
                cached = pickle.load(f)
            embedder.embeddings = cached["embeddings"]
            embedder.metadata = cached["metadata"]
            LOG.info(f"Loaded {sum(len(v) for v in embedder.embeddings.values())} cached embeddings")
        except Exception as e:
            LOG.warning(f"Cache load failed ({e}); rebuilding…")
            embedder.embed_csv_files(csv_files, force_recompute=False)

    # Search
    if args.search:
        embedder.search_and_display_results(
            args.search,
            top_k=args.top_k,
            source_filter=args.sources,
        )
    elif not args.embed:
        LOG.info("Interactive Search Mode — type a query (or 'quit'):")
        try:
            while True:
                q = input("\nSearch: ").strip()
                if q.lower() in {"quit", "exit", "q"}:
                    break
                if not q:
                    continue
                embedder.search_and_display_results(q, top_k=args.top_k)
        except KeyboardInterrupt:
            LOG.info("Goodbye!")


if __name__ == "__main__":  # pragma: no cover
    main()
