#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust SHACL → LLM-optimized schema (compact text + JSON) with target inference.

Usage:
  python shacl_to_llm_improved.py \
    --in shapes.ttl \
    --out-text compact.txt \
    --out-json compact.json \
    --infer-targets --data data.ttl --ontology ontology.ttl \
    --omit-optional --top-props 12 --max-enum 15

You can also import and call convert_shacl(...).
"""

import argparse, json, hashlib, re
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

from rdflib import Graph, URIRef, BNode, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD
from rdflib.collection import Collection
from rdflib.term import Identifier

SH = Namespace("http://www.w3.org/ns/shacl#")

# ---------- heuristics ----------
def importance_score(p: Dict[str, Any]) -> float:
    """Heuristic scoring function prioritizing significant SHACL properties."""
    score = 0.0
    if int(p.get("minCount", 0)) >= 1: score += 5.0
    if p.get("datatype") or p.get("class") or p.get("nodeKind"): score += 2.5
    if any(k in p for k in ("minInclusive","maxInclusive","minExclusive","maxExclusive")): score += 1.5
    if p.get("in"): score += 1.5
    if p.get("pattern"): score += 1.0
    if p.get("qualifiedValueShape"): score += 1.0
    if p.get("maxCount") is not None: score += 0.3
    return score

# ---------- helpers ----------
def to_curie(node: Identifier, g: Graph) -> str:
    """Render an RDFLib node as a CURIE or string fallback."""
    if isinstance(node, Literal): return str(node)
    try:
        return g.namespace_manager.normalizeUri(node)
    except Exception:
        return str(node)

def first_obj(g: Graph, s: Identifier, p: URIRef) -> Optional[Identifier]:
    """Return the first object for a subject/predicate pair, if any."""
    return next(g.objects(s, p), None)

def objs(g: Graph, s: Identifier, p: URIRef) -> List[Identifier]:
    """Return all objects for a subject/predicate pair."""
    return list(g.objects(s, p))

def get_list(g: Graph, head: Identifier) -> List[Identifier]:
    """Safely read RDF list structures into Python lists."""
    try:
        return list(Collection(g, head))
    except Exception:
        vals, cur, nil, seen = [], head, RDF.nil, set()
        while cur and cur != nil and cur not in seen:
            seen.add(cur)
            v = first_obj(g, cur, RDF.first)
            if v is not None: vals.append(v)
            cur = first_obj(g, cur, RDF.rest)
        return vals

def literal_to_py(l: Literal) -> Any:
    """Convert RDF literals to native Python types where feasible."""
    if not isinstance(l, Literal): return str(l)
    try:
        if l.datatype in (XSD.integer, XSD.int, XSD.long, XSD.short, XSD.byte,
                          XSD.nonNegativeInteger, XSD.nonPositiveInteger,
                          XSD.positiveInteger, XSD.negativeInteger):
            return int(str(l))
        if l.datatype in (XSD.decimal, XSD.double, XSD.float):
            return float(str(l))
    except Exception:
        pass
    return str(l)

# SHACL path renderer: inverse/alt/sequence/* + ? + +
def render_path(g: Graph, node: Identifier) -> str:
    """Render SHACL property paths (inverse/sequence/quantifiers) to readable text."""
    if isinstance(node, URIRef): return to_curie(node, g)
    inv = first_obj(g, node, SH.inversePath)
    if inv is not None: return f"^{render_path(g, inv)}"
    alt = first_obj(g, node, SH.alternativePath)
    if alt is not None:
        return "(" + " | ".join(render_path(g, v) for v in get_list(g, alt)) + ")"
    zmore = first_obj(g, node, SH.zeroOrMorePath)
    if zmore is not None: return f"{render_path(g, zmore)}*"
    omore = first_obj(g, node, SH.oneOrMorePath)
    if omore is not None: return f"{render_path(g, omore)}+"
    zone = first_obj(g, node, SH.zeroOrOnePath)
    if zone is not None: return f"{render_path(g, zone)}?"
    # try sequence (rdf:List)
    try:
        seq = get_list(g, node)
        if seq:
            return "/".join(render_path(g, v) for v in seq)
    except Exception:
        pass
    return to_curie(node, g)

# ---------- extraction ----------
def extract_property_shape(g: Graph, ps: Identifier, max_enum: int) -> Dict[str, Any]:
    """Summarize a SHACL property shape into a compact dictionary."""
    p: Dict[str, Any] = {}
    path = first_obj(g, ps, SH.path)
    if path is not None: p["path"] = render_path(g, path)

    dt = first_obj(g, ps, SH.datatype)
    if dt is not None: p["datatype"] = to_curie(dt, g)
    cl = first_obj(g, ps, SH["class"])
    if cl is not None: p["class"] = to_curie(cl, g)
    nk = first_obj(g, ps, SH.nodeKind)
    if nk is not None: p["nodeKind"] = to_curie(nk, g)

    mc = first_obj(g, ps, SH.minCount)
    if mc is not None: p["minCount"] = int(str(mc))
    mcx = first_obj(g, ps, SH.maxCount)
    if mcx is not None: p["maxCount"] = int(str(mcx))

    for k in (SH.minInclusive, SH.maxInclusive, SH.minExclusive, SH.maxExclusive):
        v = first_obj(g, ps, k)
        if v is not None:
            p[to_curie(k, g).split(":")[-1]] = literal_to_py(v)

    mlen = first_obj(g, ps, SH.minLength)
    if mlen is not None: p["minLength"] = int(str(mlen))
    mlen2 = first_obj(g, ps, SH.maxLength)
    if mlen2 is not None: p["maxLength"] = int(str(mlen2))
    patt = first_obj(g, ps, SH.pattern)
    if patt is not None: p["pattern"] = str(patt)
    flags = first_obj(g, ps, SH.flags)
    if flags is not None: p["flags"] = str(flags)

    langIn = first_obj(g, ps, SH.languageIn)
    if langIn is not None:
        p["languageIn"] = [str(v) if isinstance(v, Literal) else to_curie(v, g) for v in get_list(g, langIn)]
    uniq = first_obj(g, ps, SH.uniqueLang)
    if uniq is not None: p["uniqueLang"] = str(uniq).lower() in ("true","1")

    in_list = first_obj(g, ps, SH["in"])
    if in_list is not None:
        vals = [str(v) if isinstance(v, Literal) else to_curie(v, g) for v in get_list(g, in_list)]
        if vals: p["in"] = vals[:max_enum] + (["…"] if len(vals) > max_enum else [])

    hv = first_obj(g, ps, SH.hasValue)
    if hv is not None: p["hasValue"] = str(hv) if isinstance(hv, Literal) else to_curie(hv, g)

    node = first_obj(g, ps, SH.node)
    if node is not None: p["node"] = to_curie(node, g)

    qvs = first_obj(g, ps, SH.qualifiedValueShape)
    if qvs is not None:
        q: Dict[str, Any] = {"shape": to_curie(qvs, g)}
        qmin = first_obj(g, ps, SH.qualifiedMinCount)
        qmax = first_obj(g, ps, SH.qualifiedMaxCount)
        qdis = first_obj(g, ps, SH.qualifiedValueShapesDisjoint)
        if qmin is not None: q["min"] = int(str(qmin))
        if qmax is not None: q["max"] = int(str(qmax))
        if qdis is not None: q["disjoint"] = str(qdis).lower() in ("true","1")
        p["qualifiedValueShape"] = q

    for pred, key in ((SH.equals,"equals"), (SH.disjoint,"disjoint"),
                      (SH.lessThan,"lessThan"), (SH.lessThanOrEquals,"lessThanOrEquals")):
        v = first_obj(g, ps, pred)
        if v is not None: p[key] = to_curie(v, g)

    msg = first_obj(g, ps, SH.message)
    if msg is not None: p["message"] = str(msg)
    sev = first_obj(g, ps, SH.severity)
    if sev is not None: p["severity"] = to_curie(sev, g)

    return p

def extract_logical(g: Graph, s: Identifier, pred: URIRef) -> Optional[List[str]]:
    """Extract list-based logical constraints (AND/OR/NOT) as CURIE strings."""
    head = first_obj(g, s, pred)
    if head is None: return None
    vals = [to_curie(n, g) for n in get_list(g, head)]
    return vals or None

def extract_node_shape(g: Graph, s: Identifier, max_enum: int) -> Dict[str, Any]:
    """Convert SHACL node shapes into a friendly schema dictionary."""
    sh: Dict[str, Any] = {
        "id": to_curie(s, g),
        "label": next((str(x) for x in g.objects(s, RDFS.label)), None),
        "targets": {},
        "properties": [],
        "constraints": {}
    }
    for t in objs(g, s, SH.targetClass):
        sh["targets"].setdefault("targetClass", []).append(to_curie(t, g))
    for t in objs(g, s, SH.targetNode):
        sh["targets"].setdefault("targetNode", []).append(to_curie(t, g))
    for t in objs(g, s, SH.targetSubjectsOf):
        sh["targets"].setdefault("targetSubjectsOf", []).append(to_curie(t, g))
    for t in objs(g, s, SH.targetObjectsOf):
        sh["targets"].setdefault("targetObjectsOf", []).append(to_curie(t, g))

    cl = first_obj(g, s, SH.closed)
    if cl is not None:
        sh["constraints"]["closed"] = str(cl).lower() in ("true","1")
    ign = first_obj(g, s, SH.ignoredProperties)
    if ign is not None:
        sh["constraints"]["ignoredProperties"] = [to_curie(v, g) for v in get_list(g, ign)]

    for key, pred in (("and", SH["and"]), ("or", SH["or"]), ("xone", SH["xone"])):
        vals = extract_logical(g, s, pred)
        if vals: sh["constraints"][key] = vals
    not_v = first_obj(g, s, SH["not"])
    if not_v is not None: sh["constraints"]["not"] = to_curie(not_v, g)

    node = first_obj(g, s, SH.node)
    if node is not None: sh["constraints"]["node"] = to_curie(node, g)

    for ps in objs(g, s, SH.property):
        sh["properties"].append(extract_property_shape(g, ps, max_enum))
    msg = first_obj(g, s, SH.message)
    if msg is not None: sh["message"] = str(msg)
    sev = first_obj(g, s, SH.severity)
    if sev is not None: sh["severity"] = to_curie(sev, g)
    return sh

def extract_all(shapes_graph: Graph, max_enum: int) -> Dict[str, Any]:
    """Extract all SHACL node/property shapes into a structured dictionary."""
    out: Dict[str, Any] = {
        "prefixes": {p: str(u) for p, u in shapes_graph.namespace_manager.namespaces()},
        "nodeShapes": [],
        "propertyShapes": []
    }
    for s in shapes_graph.subjects(RDF.type, SH.NodeShape):
        out["nodeShapes"].append(extract_node_shape(shapes_graph, s, max_enum))
    for s in shapes_graph.subjects(RDF.type, SH.PropertyShape):
        out["propertyShapes"].append(extract_property_shape(shapes_graph, s, max_enum))
    payload = json.dumps(out, sort_keys=True).encode("utf-8")
    out["version"] = {"hash": "sha1:" + hashlib.sha1(payload).hexdigest()}
    return out

# ---------- target inference ----------
def iri_local(name: str) -> str:
    """Strip conventional shape suffixes to derive a local name hint."""
    return re.sub(r'(Shape|NodeShape)$', '', name)

def compute_qname_iri(curie: str, g: Graph) -> Optional[URIRef]:
    """Resolve a CURIE into a full IRI using the graph namespace manager."""
    try:
        return URIRef(g.namespace_manager.compute_qname(curie)[2])
    except Exception:
        return None

def majority_vote(counters: List[Counter]) -> Optional[str]:
    """Return the dominant candidate across vote counters with thresholds."""
    total = Counter()
    for c in counters: total.update(c)
    if not total: return None
    (top, n1), rest = total.most_common(1)[0], total.most_common(2)
    n2 = rest[1][1] if len(rest) > 1 else 0
    # require ≥50% or margin ≥2
    return top if (n1 >= sum(total.values()) * 0.5 or (n1 - n2) >= 2) else None

def infer_target_for_shape(shape: Dict[str, Any], shapes_graph: Graph,
                           ontology: Optional[Graph], data: Optional[Graph]) -> Optional[str]:
    """Infer a missing target class using ontology, data, and naming hints."""
    votes: List[Counter] = []
    ns = shapes_graph.namespace_manager

    # 1) ontology domain votes
    if ontology:
        c = Counter()
        for p in shape.get("properties", []):
            path = p.get("path")
            if not path or any(ch in path for ch in "()*+?|/^"):  # skip complex
                continue
            prop_iri = compute_qname_iri(path, shapes_graph)
            if not prop_iri: continue
            for dom in ontology.objects(prop_iri, RDFS.domain):
                c[to_curie(dom, shapes_graph)] += 1
        if c: votes.append(c)

    # 2) data-driven subject types
    if data:
        c = Counter()
        for p in shape.get("properties", []):
            path = p.get("path")
            if not path or any(ch in path for ch in "()*+?|/^"): continue
            prop_iri = compute_qname_iri(path, shapes_graph)
            if not prop_iri: continue
            for s in data.subjects(prop_iri, None):
                for t in data.objects(s, RDF.type):
                    c[to_curie(t, shapes_graph)] += 1
        if c: votes.append(c)

    # 3) shape name/label hint
    c = Counter()
    hint = None
    if shape.get("label"):
        hint = iri_local(shape["label"])
    elif shape.get("id"):
        sid = shape["id"]
        if ":" in sid:
            hint = iri_local(sid.split(":")[-1])
    if hint:
        for pfx, iri in ns.namespaces():
            if pfx in ("rdf","rdfs","xsd","sh"): continue
            cand_curie = f"{pfx}:{hint}"
            cand_iri = compute_qname_iri(cand_curie, shapes_graph)
            if not cand_iri: continue
            # +2 if ontology mentions class IRI anywhere; else +1
            exists = ontology and ((cand_iri, None, None) in ontology or (None, None, cand_iri) in ontology)
            c[cand_curie] += 2 if exists else 1
    if c: votes.append(c)

    # 4) inverse path back-prop via ontology
    if ontology:
        c = Counter()
        for p in shape.get("properties", []):
            path = (p.get("path") or "")
            if path.startswith("^") and ":" in path[1:]:
                prop_iri = compute_qname_iri(path[1:], shapes_graph)
                if not prop_iri: continue
                for dom in ontology.objects(prop_iri, RDFS.domain):
                    c[to_curie(dom, shapes_graph)] += 1
        if c: votes.append(c)

    return majority_vote(votes)

def infer_missing_targets(full: Dict[str, Any], shapes_graph: Graph,
                          ontology: Optional[Graph], data: Optional[Graph]) -> None:
    """Populate targetClass entries when absent by invoking inference heuristics."""
    for sh in full.get("nodeShapes", []):
        tg = sh.get("targets", {})
        has_target = any(tg.get(k) for k in ("targetClass","targetNode","targetSubjectsOf","targetObjectsOf"))
        if not has_target:
            cand = infer_target_for_shape(sh, shapes_graph, ontology, data)
            if cand:
                sh.setdefault("targets", {})["targetClass"] = [cand]

# ---------- optimization ----------
def optimize(full: Dict[str, Any], top_props: int, omit_optional: bool) -> Dict[str, Any]:
    """Trim and rank properties for a compact LLM-friendly representation."""
    res = {"prefixes": full["prefixes"], "nodeShapes": [], "propertyShapes": full["propertyShapes"], "version": full["version"]}
    for sh in full["nodeShapes"]:
        props = sh.get("properties", [])
        if omit_optional:
            props = [p for p in props if int(p.get("minCount", 0)) >= 1 or p.get("datatype") or p.get("class")]
        props.sort(key=importance_score, reverse=True)
        if top_props and len(props) > top_props:
            props = props[:top_props] + [{"path": "…"}]
        res["nodeShapes"].append({**{k:v for k,v in sh.items() if k!="properties"}, "properties": props})
    return res

# ---------- render ----------
def render_text(compact: Dict[str, Any]) -> str:
    """Render the compact schema summary as a Markdown-like text block."""
    lines: List[str] = []
    lines.append("# Compact SHACL (LLM-optimized)")
    lines.append("")
    for k, v in compact.get("prefixes", {}).items():
        lines.append(f"@prefix {k}: <{v}>")
    lines.append("")

    for sh in compact.get("nodeShapes", []):
        tg = sh.get("targets", {})
        tgt_bits = []
        for tk in ("targetClass","targetNode","targetSubjectsOf","targetObjectsOf"):
            if tk in tg: tgt_bits.append(f"{tk} " + " | ".join(tg[tk]))
        title = (sh.get("label") + " ") if sh.get("label") else ""
        title += "(" + " ; ".join(tgt_bits) + ")" if tgt_bits else "(NodeShape)"
        lines.append(title)
        for p in sh.get("properties", []):
            if p.get("path") == "…": lines.append("  - …"); continue
            rhs = []
            if "datatype" in p: rhs.append(p["datatype"])
            if "class" in p: rhs.append("@"+p["class"])
            if "nodeKind" in p: rhs.append(p["nodeKind"])
            rhs_str = " ".join(rhs) if rhs else "IRI|Literal"
            req = "[required]" if int(p.get("minCount", 0)) >= 1 else "[optional]"
            tail = []
            for k in ("minInclusive","maxInclusive","minExclusive","maxExclusive"):
                if k in p: tail.append(f"{k}={p[k]}")
            if "minLength" in p: tail.append(f"minLen={p['minLength']}")
            if "maxLength" in p: tail.append(f"maxLen={p['maxLength']}")
            if "pattern" in p: tail.append(f"/{p['pattern']}/")
            if "languageIn" in p: tail.append("lang∈{"+",".join(p["languageIn"])+"}")
            if p.get("uniqueLang"): tail.append("uniqueLang")
            if "in" in p: tail.append("{"+ " | ".join(map(str, p["in"])) + "}")
            if "hasValue" in p: tail.append(f"= {p['hasValue']}")
            for k in ("equals","disjoint","lessThan","lessThanOrEquals"):
                if k in p: tail.append(f"{k}:{p[k]}")
            if "qualifiedValueShape" in p:
                q = p["qualifiedValueShape"]
                qbits = [q.get("shape")]
                if "min" in q: qbits.append(f"min={q['min']}")
                if "max" in q: qbits.append(f"max={q['max']}")
                if q.get("disjoint"): qbits.append("disjoint")
                tail.append("QVS(" + ", ".join(qbits) + ")")
            lines.append(f"  - {p.get('path','?')} : {rhs_str} {req}" + ("" if not tail else " " + " ".join(tail)))
        c = sh.get("constraints", {})
        extras = []
        if c.get("closed"): extras.append("closed")
        if c.get("ignoredProperties"): extras.append("ignored{" + ", ".join(c["ignoredProperties"]) + "}")
        if c.get("and"): extras.append("AND(" + " , ".join(c["and"]) + ")")
        if c.get("or"): extras.append("OR(" + " , ".join(c["or"]) + ")")
        if c.get("xone"): extras.append("XONE(" + " , ".join(c["xone"]) + ")")
        if c.get("not"): extras.append("NOT(" + c["not"] + ")")
        if c.get("node"): extras.append("NODE(" + c["node"] + ")")
        if extras: lines.append("  # " + " ; ".join(extras))
        lines.append("")
    if compact.get("propertyShapes"):
        lines.append("# Standalone PropertyShapes")
        for ps in compact["propertyShapes"]:
            head = ps.get("path", ps.get("id", "(PropertyShape)"))
            rhs = []
            if "datatype" in ps: rhs.append(ps["datatype"])
            if "class" in ps: rhs.append("@"+ps["class"])
            if "nodeKind" in ps: rhs.append(ps["nodeKind"])
            req = "[required]" if int(ps.get("minCount", 0)) >= 1 else "[optional]"
            lines.append(f"- {head} : {' '.join(rhs) if rhs else 'IRI|Literal'} {req}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"

# ---------- conversion API ----------
def convert_shacl(
    shacl_graph: Graph,
    *,
    out_text_path: Optional[str] = None,
    out_json_path: Optional[str] = None,
    top_props: int = 16,
    max_enum: int = 20,
    omit_optional: bool = False,
    infer_targets: bool = False,
    ontology_graph: Optional[Graph] = None,
    data_graph: Optional[Graph] = None,
) -> Dict[str, Any]:
    """Convert SHACL shapes into compact text/JSON artifacts."""
    full = extract_all(shacl_graph, max_enum=max_enum)
    if infer_targets:
        infer_missing_targets(full, shacl_graph, ontology_graph, data_graph)
    compact = optimize(full, top_props=top_props, omit_optional=omit_optional)
    if out_json_path:
        with open(out_json_path, "w", encoding="utf-8") as jf:
            json.dump(compact, jf, indent=2)
    if out_text_path:
        with open(out_text_path, "w", encoding="utf-8") as tf:
            tf.write(render_text(compact))
    return compact

# ---------- CLI ----------
def main():
    """CLI harness for converting SHACL files into LLM-optimized summaries."""
    ap = argparse.ArgumentParser(description="Improved SHACL→LLM converter with target inference.")
    ap.add_argument("--in", dest="inp", required=True, help="Input SHACL (.ttl/.rdf/.jsonld/...)")
    ap.add_argument("--out-text", dest="out_text", required=True, help="Output compact text")
    ap.add_argument("--out-json", dest="out_json", required=True, help="Output compact JSON")
    ap.add_argument("--top-props", type=int, default=16, help="Top-K properties per NodeShape (0=all)")
    ap.add_argument("--max-enum", type=int, default=20, help="Cap sh:in values to N")
    ap.add_argument("--omit-optional", action="store_true", help="Drop optional properties (minCount==0)")
    ap.add_argument("--infer-targets", action="store_true", help="Infer missing targets using ontology/data/name hints")
    ap.add_argument("--ontology", dest="ont", help="Optional ontology (for rdfs:domain/range)")
    ap.add_argument("--data", dest="data", help="Optional data graph (your TTL output) for evidence")
    args = ap.parse_args()

    shacl_g = Graph(); shacl_g.parse(args.inp)
    ont_g = Graph(); ont_loaded = False
    if args.ont:
        ont_g.parse(args.ont); ont_loaded = True
    data_g = Graph(); data_loaded = False
    if args.data:
        data_g.parse(args.data); data_loaded = True

    convert_shacl(
        shacl_g,
        out_text_path=args.out_text,
        out_json_path=args.out_json,
        top_props=args.top_props,
        max_enum=args.max_enum,
        omit_optional=args.omit_optional,
        infer_targets=args.infer_targets,
        ontology_graph=(ont_g if ont_loaded else None),
        data_graph=(data_g if data_loaded else None),
    )
    print(f"✓ Wrote {args.out_text}")
    print(f"✓ Wrote {args.out_json}")

if __name__ == "__main__":
    main()
