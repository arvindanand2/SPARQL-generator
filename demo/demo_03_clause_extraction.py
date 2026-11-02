"""
Demo 3: Clause Extraction and Chunking
Demonstrates how user queries are processed by extract_chunk.py to extract structured clauses.
"""

import sys
import os
import subprocess
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from compress_chunk import compact_chunk

def demo_clause_extraction():
    """Show how clause extraction runs and how the results can be compressed."""
    print("=" * 60)
    print("DEMO 3: Clause Extraction and Chunking")
    print("=" * 60)

    # Target query and variations for testing
    target_query = "Get the number of manufacturers in the KG serving each type of Industry. Display the Industry name and corresponding count."

    test_queries = [
        target_query,
        "Count manufacturers by industry type",
        "Show industry breakdown with manufacturer counts",
        "List all industries and their manufacturer numbers"
    ]

    print("\n1. Testing clause extraction for industry count queries:")
    print("-" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: '{query[:60]}{'...' if len(query) > 60 else ''}'")
        print(f"   Full query: {query}")

        try:
            # Run extract_chunk.py
            result = subprocess.run([
                sys.executable, "../extract_chunk.py", query
            ], capture_output=True, text=True, timeout=600, cwd=os.path.dirname(__file__))

            if result.returncode == 0:
                try:
                    extracted = json.loads(result.stdout)

                    print(f"   ✅ Extraction successful!")
                    print(f"   📊 Found {len(extracted.get('clauses', []))} clauses")

                    # Show clause details
                    for j, clause in enumerate(extracted.get('clauses', [])[:3], 1):  # Show first 3
                        print(f"   Clause {j}:")
                        print(f"     Type: {clause.get('type', 'N/A')}")
                        print(f"     Surface: '{clause.get('surface', 'N/A')}'")
                        print(f"     Subject: {clause.get('subject_hint', 'N/A')}")
                        print(f"     Relation: {clause.get('relation_hint', 'N/A')}")
                        print(f"     Object: {clause.get('object_hint', 'N/A')}")
                        print(f"     Confidence: {clause.get('confidence', 'N/A')}")

                    if len(extracted.get('clauses', [])) > 3:
                        print(f"   ... and {len(extracted.get('clauses', [])) - 3} more clauses")

                    # Test compression
                    print(f"\n   📦 Testing compression:")
                    original_size = len(json.dumps(extracted))
                    compressed = compact_chunk(extracted)
                    compressed_size = len(json.dumps(compressed))

                    print(f"     Original size: {original_size} chars")
                    print(f"     Compressed size: {compressed_size} chars")
                    print(f"     Compression ratio: {compressed_size/original_size:.2f}")

                    # Show warnings if any
                    if 'warnings' in extracted:
                        print(f"   ⚠️  Warnings: {len(extracted['warnings'])}")
                        for warning in extracted['warnings'][:2]:
                            print(f"     - {warning}")

                    # Analyze query intent
                    has_count = any('count' in clause.get('surface', '').lower() or 'number' in clause.get('surface', '').lower()
                                  for clause in extracted.get('clauses', []))
                    has_industry = any('industry' in clause.get('surface', '').lower()
                                     for clause in extracted.get('clauses', []))
                    has_manufacturer = any('manufacturer' in clause.get('surface', '').lower()
                                         for clause in extracted.get('clauses', []))

                    print(f"   🎯 Query intent analysis:")
                    print(f"     Contains counting: {'✓' if has_count else '✗'}")
                    print(f"     References industry: {'✓' if has_industry else '✗'}")
                    print(f"     References manufacturers: {'✓' if has_manufacturer else '✗'}")

                except json.JSONDecodeError as e:
                    print(f"   ❌ JSON parse error: {e}")
                    print(f"   Raw output: {result.stdout[:200]}...")
            else:
                print(f"   ❌ Extraction failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            print(f"   ⏱️  Extraction timed out")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print(f"\n2. Expected clause patterns for counting queries:")
    print("-" * 50)

    expected_patterns = [
        {"type": "aggregation", "operation": "count", "subject": "manufacturers"},
        {"type": "grouping", "group_by": "industry", "measure": "manufacturer_count"},
        {"type": "display", "fields": ["industry_name", "count"]},
        {"type": "spo", "subject": "manufacturer", "relation": "serves", "object": "industry"}
    ]

    print("   Expected extraction patterns:")
    for i, pattern in enumerate(expected_patterns, 1):
        print(f"   {i}. {pattern}")

    print("\n✅ Clause extraction and chunking demo complete!")

if __name__ == "__main__":
    demo_clause_extraction()
