"""Embed ontology classes using OpenAI embeddings with textual summaries."""

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
    """Extract the fragment or last path segment from an IRI."""
    if not iri:
        return ""
    p = urlparse(iri)
    if p.fragment:
        return p.fragment
    return (p.path or "").rstrip("/").rsplit("/", 1)[-1] or iri


def to_embedding_text(c):
    """Build a concise text description for a class embedding payload.

    Parameters
    ----------
    c : dict
        Class metadata including label, aliases, and property hints.

    Returns
    -------
    str
        Multi-line string used as input to the embedding model.
    """
    # Build a short, informative text for embedding a class
    label = c.get("label") or short_name(c.get("iri", ""))
    aliases = ", ".join(c.get("aliases") or []) or "none"
    iri = c.get("iri", "")
    supers = [short_name(x) for x in c.get("superClassOf") or []] if c.get("superClassOf") else [short_name(x) for x in c.get("super_class_iris") or []] if c.get("super_class_iris") else []
    supers = supers or ["unspecified"]
    comment = (c.get("comment") or c.get("description") or "").strip() or "no definition"
    props = c.get("properties") or c.get("predicates") or []
    prop_names = []
    for p in props:
        if isinstance(p, dict):
            prop_names.append(p.get("label") or short_name(p.get("iri", "")))
        else:
            prop_names.append(short_name(p))
    prop_str = ", ".join(prop_names) or "none"
    natural = f"Class {label}; superclasses: {', '.join(supers)}; properties: {prop_str}."
    return (
        f"{label} (class)\n"
        f"Aliases: {aliases}.\n"
        f"IRI: {iri}.\n"
        f"Superclasses: {', '.join(supers)}.\n"
        f"Properties: {prop_str}.\n"
        f"Definition: {comment}\n"
        f"Natural phrasing: “{natural}”"
    )


def load_jsonl(path: Path):
    """Yield parsed JSON objects from a JSONL file.

    Parameters
    ----------
    path : Path
        Path to the JSON Lines file to stream.

    Yields
    ------
    dict
        Parsed JSON objects for each non-empty line.
    """
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# --- main ---
# Accept either classes.jsonl or classes_embedding.jsonl as input
candidates = [Path("classes.jsonl"), Path("classes_embedding.jsonl"), Path("classes.jsonl.gz")]
src = None
for cand in candidates:
    if cand.exists():
        src = cand
        break
if src is None:
    raise FileNotFoundError(f"Input file not found. Create 'classes.jsonl' or 'classes_embedding.jsonl' in the project root.")

classes = list(load_jsonl(src))

for c in classes:
    c["embedding_text"] = to_embedding_text(c)

# now embed them
texts = [c["embedding_text"] for c in classes]
try:
    resp = client.embeddings.create(model="text-embedding-3-large", input=texts)
except Exception as e:
    raise RuntimeError(f"OpenAI embedding call failed: {e}")

for c, e in zip(classes, resp.data):
    c["embedding"] = e.embedding

# save enriched file
out = Path("classes_with_embeddings.jsonl")
with out.open("wb") as f:
    for c in classes:
        data = json.dumps(c)
        if isinstance(data, bytes):
            f.write(data + b"\n")
        else:
            f.write(data.encode("utf-8") + b"\n")

print(f"Saved {len(classes)} classes with embeddings → {out}")
