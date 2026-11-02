# compact_chunk.py
"""Helpers to compress clause extraction payloads down to meaningful fields."""

import json
from typing import Any, Dict, List

KEEP_KEYS = {
    "type",
    "subject_hint",
    "relation_hint",
    "object_hint",
    "enum_values",
    "filter",
    "modifier",
    "connective",
}
MEANINGFUL_FILTER_OPS = {"lt", "lte", "eq", "gte", "gt", "between", "before", "after", "on", "near"}

def _is_empty(v: Any) -> bool:
    """Return ``True`` when the value carries no actionable information."""
    return v is None or v == "" or v == [] or v == {}

def _is_meaningful_filter(f: Dict[str, Any]) -> bool:
    """Determine whether a filter clause should be retained in the compact form."""
    if not isinstance(f, dict):
        return False
    op = f.get("op")
    if op not in MEANINGFUL_FILTER_OPS:
        return False
    # Keep if any of these carry information
    if f.get("value") not in (None, ""):
        return True
    if f.get("unit"):
        return True
    if f.get("min") is not None or f.get("max") is not None:
        return True
    # Keep op=near even if value present as empty check; op itself may imply semantics
    if op == "near" and f.get("value"):
        return True
    return False

def _is_meaningful_modifier(m: Dict[str, Any], clause_type: str) -> bool:
    """Mark modifiers that introduce groupings or adjustments worth retaining."""
    if not isinstance(m, dict):
        return False
    if clause_type == "modifier":
        return True
    # throw away the common no-op group stub
    if m.get("kind") == "group" and m.get("k") in (None, 0) and (m.get("by") in (None, "")):
        return False
    # keep if it has actionable knobs
    return bool(m.get("k") not in (None,) or m.get("by"))

def compact_clause(c: Dict[str, Any]) -> Dict[str, Any]:
    """Strip redundant keys from a clause while preserving semantics."""
    # base keep set
    out = {k: c.get(k) for k in KEEP_KEYS if k in c}

    # trim enum_values if empty
    if "enum_values" in out and _is_empty(out["enum_values"]):
        out.pop("enum_values")

    # filter
    f = out.get("filter")
    if f is not None:
        if not _is_meaningful_filter(f):
            out.pop("filter", None)

    # modifier
    m = out.get("modifier")
    if m is not None:
        if not _is_meaningful_modifier(m, c.get("type", "")):
            out.pop("modifier", None)

    # drop any empty/null leftovers
    for k in list(out.keys()):
        if _is_empty(out[k]):
            out.pop(k, None)

    return out

def compact_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Apply ``compact_clause`` to each clause and retain warnings mapping."""
    clauses = chunk.get("clauses", [])
    compacted = [cc for c in clauses if (cc := compact_clause(c))]
    # Keep warnings as-is (or drop if you prefer)
    warnings = chunk.get("warnings", [])
    return {"clauses": compacted, "mapping": warnings}

if __name__ == "__main__":
    import sys
    raw = json.load(sys.stdin)
    result = compact_chunk(raw)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
