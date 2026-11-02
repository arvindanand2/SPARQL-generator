"""
Test to verify that both batch pipelines produce identical results.
"""

import json
import os
import sys
from pathlib import Path

def compare_extractions(original_dir, resumable_dir):
    """Compare extracted.json from both pipelines."""
    orig_path = Path(original_dir) / "extracted.json"
    resu_path = Path(resumable_dir) / "extracted.json"

    if not orig_path.exists() or not resu_path.exists():
        print("❌ Missing extraction files")
        return False

    with open(orig_path) as f:
        orig = json.load(f)
    with open(resu_path) as f:
        resu = json.load(f)

    if orig == resu:
        print("✅ Extractions are identical")
        return True
    else:
        print("❌ Extractions differ")
        print("Original:", json.dumps(orig, indent=2)[:500])
        print("Resumable:", json.dumps(resu, indent=2)[:500])
        return False

def compare_sparql(original_dir, resumable_dir):
    """Compare SPARQL queries from both pipelines."""
    orig_path = Path(original_dir) / "sparql_query.txt"
    resu_path = Path(resumable_dir) / "sparql_query.txt"

    if not orig_path.exists() or not resu_path.exists():
        print("❌ Missing SPARQL files")
        return False

    orig = orig_path.read_text(encoding='utf-8').strip()
    resu = resu_path.read_text(encoding='utf-8').strip()

    if orig == resu:
        print("✅ SPARQL queries are identical")
        return True
    else:
        print("❌ SPARQL queries differ")
        print("Original:", orig[:300])
        print("Resumable:", resu[:300])
        return False

def compare_results(original_dir, resumable_dir):
    """Compare GraphDB results from both pipelines."""
    orig_path = Path(original_dir) / "graphdb_result.txt"
    resu_path = Path(resumable_dir) / "graphdb_result.txt"

    if not orig_path.exists() or not resu_path.exists():
        print("❌ Missing result files")
        return False

    orig = orig_path.read_text(encoding='utf-8').strip()
    resu = resu_path.read_text(encoding='utf-8').strip()

    if orig == resu:
        print("✅ GraphDB results are identical")
        return True
    else:
        print("❌ GraphDB results differ")
        print("Original:", orig[:300])
        print("Resumable:", resu[:300])
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python test_pipeline_equivalence.py <original_run_dir> <resumable_run_dir>")
        sys.exit(1)

    original_dir = sys.argv[1]
    resumable_dir = sys.argv[2]

    print("Comparing pipeline outputs...")
    print("="*80)

    results = []
    results.append(compare_extractions(original_dir, resumable_dir))
    results.append(compare_sparql(original_dir, resumable_dir))
    results.append(compare_results(original_dir, resumable_dir))

    print("="*80)
    if all(results):
        print("✅ ALL OUTPUTS ARE IDENTICAL")
        sys.exit(0)
    else:
        print("❌ OUTPUTS DIFFER")
        sys.exit(1)

if __name__ == "__main__":
    main()

