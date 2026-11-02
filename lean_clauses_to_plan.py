#!/usr/bin/env python3
"""
lean_clauses_to_plan.py
-----------------------
Accepts a JSON file with "clauses" (SPO-style) and resolves subject/predicate/object
to KG IRIs using the hybrid detector, but outputs a **lean** JSON that contains only
what you need to build SPARQL (no raw blobs, no giant candidates).

Usage:
  python lean_clauses_to_plan.py --clauses clauses.json \
      --artifacts ./artifacts --model text-embedding-3-large [--no-bm25] \
      [--min-conf 0.0] [--include-alts 0]

Output schema (per clause):
{
  "surface": "...",
  "type": "spo",
  "connective": "AND",
  "subject":   {"label": "...", "iri": "...", "conf": 0.81},
  "predicate": {"label": "...", "iri": "...", "dir": "s->o", "conf": 0.77},
  "object":    {"label": "...", "iri": "...", "conf": 0.79},
  "filter": null,
  "modifier": null
}
"""

import argparse, json, sys
from typing import Dict, Any, List, Optional

# Try both module names to avoid import errors
try:
    from hybrid_detector_hybridbm25_toggle import HybridDetector
except ImportError:
    from hybrid_detector_hybridbm25 import HybridDetector  # fallback

def _pack_slot(mapped: Dict[str, Any], role: str, min_conf: float, include_alts: int):
    """Convert detector slot to lean form."""
    if not mapped:
        return None
    top = mapped.get("top")
    if not top:
        return None
    conf = float(top.get("score", 0.0))
    if conf < min_conf:
        return None
    out = {"label": top.get("label"), "iri": top.get("iri"), "conf": round(conf, 3)}
    if role == "predicate":
        out["dir"] = mapped.get("direction") or "s->o"
    if include_alts > 0:
        alts = [{"label": a.get("label"), "iri": a.get("iri")} for a in mapped.get("alternates", [])[:include_alts]]
        if alts:
            out["alts"] = alts
    return out

def _resolve(hd: HybridDetector, span: str, kind: str, ctx: str, subj_types=None, obj_types=None):
    """Resolve an individual hint using the hybrid detector."""
    mention = {"span": span or "", "kind": kind, "ctx": ctx or span or "", "source": "clause"}
    return hd.map_mention(mention, subj_types=subj_types, obj_types=obj_types, k=8)

def run(clauses_json: Dict[str, Any], artifacts: str, model: str, use_bm25: bool, min_conf: float, include_alts: int) -> Dict[str, Any]:
    """Resolve clause hints and return a lean plan structure."""
    hd = HybridDetector(artifacts_dir=artifacts, model=model, use_bm25=use_bm25)
    lean = {"bm25_enabled": use_bm25, "clauses": [], "warnings": clauses_json.get("warnings", [])}

    for clause in clauses_json.get("clauses", []):
        if clause.get("type") != "spo":
            # Pass non-spo through (filters-only etc.), unchanged but lean.
            lean["clauses"].append({
                "surface": clause.get("surface"),
                "type": clause.get("type"),
                "connective": clause.get("connective", "AND"),
                "filter": clause.get("filter"),
                "modifier": clause.get("modifier")
            })
            continue

        surface = clause.get("surface", "")
        subj_hint = clause.get("subject_hint") or ""
        pred_hint = clause.get("relation_hint") or ""
        obj_hint  = clause.get("object_hint") or ""

        sub_map = _resolve(hd, subj_hint, "class", surface)
        # Use subject/object IRIs as weak schema hints for predicate
        stypes = [sub_map["top"]["iri"]] if sub_map.get("top") and sub_map["top"].get("iri") else None
        obj_map = _resolve(hd, obj_hint, "class", surface) if obj_hint else {"top": None}
        otypes = [obj_map["top"]["iri"]] if obj_map.get("top") and obj_map["top"].get("iri") else None
        pred_map = _resolve(hd, pred_hint, "predicate", surface, subj_types=stypes, obj_types=otypes)

        entry = {
            "surface": surface,
            "type": "spo",
            "connective": clause.get("connective", "AND"),
            "subject": _pack_slot(sub_map, "class", min_conf, include_alts),
            "predicate": _pack_slot(pred_map, "predicate", min_conf, include_alts),
            "object": _pack_slot(obj_map, "class", min_conf, include_alts),
            "filter": clause.get("filter"),
            "modifier": clause.get("modifier"),
        }
        lean["clauses"].append(entry)

    return lean

def main():
    """CLI front-end for producing lean SPARQL planning inputs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--clauses", type=str, required=True, help="Path to clauses JSON")
    ap.add_argument("--artifacts", type=str, default="./artifacts", help="Path to artifacts directory")
    ap.add_argument("--model", type=str, default="text-embedding-3-large", help="Embedding model name")
    ap.add_argument("--no-bm25", action="store_true", help="Disable BM25 (cosine-only)")
    ap.add_argument("--min-conf", type=float, default=0.0, help="Drop slots whose confidence < this")
    ap.add_argument("--include-alts", type=int, default=0, help="Include up to N alternates per slot (0 to omit)")
    args = ap.parse_args()

    try:
        data = json.loads(open(args.clauses, "r", encoding="utf-8").read())
    except Exception as e:
        print(f"[ERROR] Failed to read clauses JSON: {e}", file=sys.stderr)
        sys.exit(1)

    lean = run(data, artifacts=args.artifacts, model=args.model, use_bm25=(not args.no_bm25),
               min_conf=args.min_conf, include_alts=args.include_alts)
    print(json.dumps(lean, indent=2))

if __name__ == "__main__":
    main()
