#!/usr/bin/env python3
"""
demo_clauses_to_plan.py
-----------------------
Takes structured clauses JSON (user input) and resolves subject/predicate/object hints
to ontology IRIs using the HybridDetector. Optionally emits a SPARQL skeleton.

Usage:
  python demo_clauses_to_plan.py --clauses clauses.json --artifacts ./artifacts --model text-embedding-3-large
  python demo_clauses_to_plan.py --clauses clauses.json --artifacts ./artifacts --model text-embedding-3-large --no-bm25 --emit-sparql
"""

import argparse, json, sys
from pathlib import Path

from demo_hybrid_bm25 import HybridDetector

def resolve_clause(hd: HybridDetector, clause: dict, subj_types=None, obj_types=None):
    """Use the detector to map clause hints onto ontology resources.

    Parameters
    ----------
    hd : HybridDetector
        Configured detector providing embedding and BM25 lookups.
    clause : dict
        Clause dictionary as produced by clause extraction.
    subj_types : list | None
        Optional subject type hints that refine predicate mapping.
    obj_types : list | None
        Optional object type hints that refine predicate mapping.

    Returns
    -------
    dict
        Copy of the clause with ``subject``, ``object``, and ``predicate`` enrichments.
    """
    out = dict(clause)

    subj_hint = clause.get("subject_hint")
    obj_hint  = clause.get("object_hint")
    rel_hint  = clause.get("relation_hint")

    if subj_hint:
        out["subject"] = hd.map_mention({"span": subj_hint, "kind":"class","ctx":clause.get("surface","")})
    if obj_hint:
        out["object"]  = hd.map_mention({"span": obj_hint, "kind":"class","ctx":clause.get("surface","")})
    if rel_hint:
        out["predicate"] = hd.map_mention({"span": rel_hint, "kind":"predicate","ctx":clause.get("surface","")}, subj_types=subj_types, obj_types=obj_types)

    return out

def emit_sparql(clauses, out=sys.stdout):
    """Emit a minimal SPARQL skeleton illustrating resolved predicates.

    Parameters
    ----------
    clauses : list[dict]
        Resolved clause dictionaries including ``predicate`` metadata.
    out : TextIO, optional
        Writable stream for the emitted SPARQL sketch; defaults to stdout.
    """
    print("SELECT ?s ?o WHERE {", file=out)
    for c in clauses:
        if "predicate" in c and c["predicate"].get("top"):
            p = c["predicate"]["top"]["iri"]
            direction = c["predicate"].get("direction","s->o")
            s="?s"; o="?o"
            if direction == "o->s":
                s,o=o,s
            print(f"  {s} <{p}> {o} .", file=out)
        if c.get("filter"):
            print(f"  # FILTER {json.dumps(c['filter'])}", file=out)
    print("}", file=out)

def main():
    """Resolve clause hints to IRIs and optionally generate a SPARQL sketch."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--clauses", type=str, required=True, help="Path to clauses JSON file")
    ap.add_argument("--artifacts", type=str, default="./artifacts", help="Path to artifacts directory")
    ap.add_argument("--model", type=str, default="text-embedding-3-large", help="Embedding model")
    ap.add_argument("--no-bm25", action="store_true", help="Disable BM25 scoring")
    ap.add_argument("--emit-sparql", action="store_true", help="Emit a SPARQL skeleton")
    ap.add_argument("--output", type=str, default=None, help="Path to output file (JSON, and SPARQL if requested)")
    args = ap.parse_args()

    hd = HybridDetector(artifacts_dir=args.artifacts, model=args.model, use_bm25=not args.no_bm25)

    clauses = json.load(open(args.clauses,"r",encoding="utf-8")).get("clauses",[])

    resolved = []
    for c in clauses:
        subj_types=[]; obj_types=[]
        rc = resolve_clause(hd, c, subj_types=subj_types, obj_types=obj_types)
        resolved.append(rc)

    output_obj = {"resolved_clauses": resolved, "bm25_enabled": not args.no_bm25}

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_obj, f, indent=2)
            f.write("\n")
            if args.emit_sparql:
                f.write("\n# SPARQL sketch:\n")
                import io
                sparql_buf = io.StringIO()
                emit_sparql(resolved, out=sparql_buf)
                f.write(sparql_buf.getvalue())
    else:
        print(json.dumps(output_obj, indent=2))
        if args.emit_sparql:
            print("\n# SPARQL sketch:")
            emit_sparql(resolved)

if __name__ == "__main__":
    main()
