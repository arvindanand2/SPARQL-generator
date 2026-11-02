"""Embed ontology predicates with descriptive text prompts."""

import orjson as json
from pathlib import Path
from urllib.parse import urlparse
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment first so OPENAI_API_KEY is available
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment. Please add it to your .env file.")

client = OpenAI(api_key=api_key)

def short_name(iri: str) -> str:
    """Extract the fragment or terminal path segment from an IRI."""
    if not iri:
        return ""
    p = urlparse(iri)
    if p.fragment:
        return p.fragment
    return (p.path or "").rstrip("/").rsplit("/", 1)[-1] or iri

def to_embedding_text(p):
    """Generate a predicate-centric embedding prompt string.

    Parameters
    ----------
    p : dict
        Predicate metadata dictionary including label, aliases, and domain/range.

    Returns
    -------
    str
        Multi-line textual description used for embedding.
    """
    label   = p.get("label") or short_name(p["iri"])
    aliases = ", ".join(p.get("aliases") or []) or "none"
    iri     = p["iri"]
    domains = [short_name(x) for x in p.get("domain_class_iris") or []] or ["unspecified"]
    ranges  = [short_name(x) for x in p.get("range_class_iris")  or []] or ["unspecified"]
    comment = (p.get("comment") or "").strip() or "no definition"
    natural = f"A(n) {', '.join(domains)} {label} a(n) {', '.join(ranges)}."
    return (
        f"{label} (predicate)\n"
        f"Aliases: {aliases}.\n"
        f"IRI: {iri}.\n"
        f"Domain: {', '.join(domains)}.\n"
        f"Range: {', '.join(ranges)}.\n"
        f"Definition: {comment}\n"
        f"Natural phrasing: “{natural}”"
    )

def load_jsonl(path: Path):
    """Yield parsed JSON objects from a JSONL file.

    Parameters
    ----------
    path : Path
        Path to the JSON Lines file.

    Yields
    ------
    dict
        Parsed object for each non-empty line.
    """
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# --- main ---
src = Path("predicates.jsonl")
if not src.exists():
    raise FileNotFoundError(f"Input file not found: {src}")

preds = list(load_jsonl(src))

for p in preds:
    p["embedding_text"] = to_embedding_text(p)

# now embed them
texts = [p["embedding_text"] for p in preds]
resp = client.embeddings.create(model="text-embedding-3-large", input=texts)

for p, e in zip(preds, resp.data):
    p["embedding"] = e.embedding

# save enriched file
out = Path("predicates_with_embeddings.jsonl")
with out.open("wb") as f:
    for p in preds:
        # orjson.dumps returns bytes, standard json.dumps returns str
        data = json.dumps(p)
        if isinstance(data, bytes):
            f.write(data + b"\n")
        else:
            f.write(data.encode("utf-8") + b"\n")

print(f"Saved {len(preds)} predicates with embeddings → {out}")
