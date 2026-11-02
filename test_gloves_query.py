#!/usr/bin/env python3
"""
Test passing only query_text directly to embedding search
"""

import json
from enhanced_csv_embedder import EnhancedCSVEmbedder

def test_direct_query():
    """Test passing query_text directly to embedding search"""

    # Your natural language query
    query_text = "Where can I find gloves which are made for medical grade in Savannah Georgia which has more than 250 employees. I want the company to be iso 9000 certified and also manufacture chips?"

    print("🚀 TESTING DIRECT QUERY EMBEDDING")
    print("=" * 80)
    print(f"Query: {query_text}")
    print("=" * 80)

    # Initialize the embedder
    embedder = EnhancedCSVEmbedder()
    embedder.load_iri_mapping("label_iri_mappings.csv")

    # Load embeddings from cache
    print("\n📥 Loading embeddings...")
    csv_files = {
        "industries": "industries_enhanced.csv",
        "material_caps": "material_caps_enhanced.csv",
        "certificates": "certificates_enhanced_fixed.csv",
        "process_embeddings": "process_embeddings_enhanced_with_def.csv"
    }
    embedder.embed_csv_files(csv_files, force_recompute=False)

    # Direct search with query_text only
    print("\n" + "=" * 80)
    print("🔍 DIRECT EMBEDDING SEARCH")
    print("=" * 80)

    results = embedder.search_and_display_results(query_text, top_k=10)

    # Show just the essential results
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)

    print(f"✅ Query: {query_text}")
    print(f"🔍 Found {len(results)} relevant results")

    print(f"\n🏆 TOP 5 MATCHES:")
    for i, result in enumerate(results[:5], 1):
        print(f"   {i}. {result.label} ({result.score:.3f}) - {result.iri}")

    return results

if __name__ == "__main__":
    results = test_direct_query()
    print(f"\n🎉 Direct query test completed!")
