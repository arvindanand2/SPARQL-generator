"""
Demo 1: Embedder Initialization and Setup
Demonstrates how the enhanced CSV embedder is initialized and configured.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from enhanced_csv_embedder import EnhancedCSVEmbedder
import json

def demo_embedder_initialization():
    """Walk through initialization, CSV loading, and a sample retrieval query."""
    print("=" * 60)
    print("DEMO 1: Enhanced CSV Embedder Initialization")
    print("=" * 60)

    print("\n1. Creating embedder instance...")
    embedder = EnhancedCSVEmbedder()

    print("\n2. Loading IRI mapping...")
    embedder.load_iri_mapping("label_iri_mappings.csv")
    print(f"   Loaded {len(embedder.label_to_iri)} label-to-IRI mappings")

    print("\n3. Sample mappings (first 5):")
    for i, (label, iri) in enumerate(list(embedder.label_to_iri.items())[:5]):
        print(f"   {i+1}. '{label}' -> {iri}")

    print("\n4. Loading CSV embeddings...")
    csv_files = {
        "industries": "industries_enhanced.csv",
        "material_caps": "material_caps_enhanced.csv",
        "certificates": "certificates_enhanced_fixed.csv",
        "process_embeddings": "process_embeddings_enhanced_with_def.csv"
    }

    embedder.embed_csv_files(csv_files, force_recompute=False)

    print("\n5. Embedder statistics:")
    for category, data in embedder.embeddings.items():
        print(f"   {category}: {len(data['labels'])} items")

    print("\n6. Sample labels from each category:")
    for category, data in embedder.embeddings.items():
        sample_labels = data['labels'][:3]
        print(f"   {category}: {sample_labels}")

    print("\n7. Testing with target query:")
    target_query = "Get the number of manufacturers in the KG serving each type of Industry. Display the Industry name and corresponding count."
    print(f"   Target query: '{target_query}'")

    # Test embedding search for this query
    results = embedder.search_closest_labels(target_query, top_k=5)
    print(f"   Top matches for this query:")
    for i, result in enumerate(results[:3], 1):
        print(f"   {i}. {result.label} -> {result.iri}")

    print("\n✅ Embedder initialization complete!")
    return embedder

if __name__ == "__main__":
    demo_embedder_initialization()
