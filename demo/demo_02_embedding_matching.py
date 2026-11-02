"""
Demo 2: Embedding Matching and IRI Extraction
Demonstrates how user queries are matched to embeddings and IRIs are extracted.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from enhanced_csv_embedder import EnhancedCSVEmbedder
import json

def demo_embedding_matching():
    """Showcase multi-query embedding retrieval and IRI extraction workflows."""
    print("=" * 60)
    print("DEMO 2: Embedding Matching and IRI Extraction")
    print("=" * 60)

    # Initialize embedder
    print("\n1. Initializing embedder...")
    embedder = EnhancedCSVEmbedder()
    embedder.load_iri_mapping("label_iri_mappings.csv")

    csv_files = {
        "industries": "industries_enhanced.csv",
        "material_caps": "material_caps_enhanced.csv",
        "certificates": "certificates_enhanced_fixed.csv",
        "process_embeddings": "process_embeddings_enhanced_with_def.csv"
    }
    embedder.embed_csv_files(csv_files, force_recompute=False)

    # Target query for demonstration
    target_query = "Get the number of manufacturers in the KG serving each type of Industry. Display the Industry name and corresponding count."

    # Additional test variations to show matching capabilities
    test_queries = [
        target_query,
        "Count of manufacturers by industry type",
        "How many companies serve each industry sector",
        "Industry breakdown of manufacturing companies",
        "Manufacturer distribution across industries"
    ]

    print("\n2. Testing embedding matching for industry-related queries:")
    print("-" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: '{query}'")

        # Get closest matches
        results = embedder.search_closest_labels(query, top_k=5)

        print(f"   Top {len(results)} matches:")
        for j, result in enumerate(results, 1):
            similarity = f"{result.similarity:.3f}" if hasattr(result, 'similarity') else "N/A"
            iri_status = "✓" if result.iri and "No IRI found" not in result.iri else "✗"
            print(f"   {j}. {result.label} (sim: {similarity}) [{iri_status}]")
            if result.iri and "No IRI found" not in result.iri:
                print(f"      IRI: {result.iri}")

        # Extract IRIs for this query
        iris_info = []
        for result in results:
            if result.iri and "No IRI found" not in result.iri:
                iris_info.append({
                    "label": result.label,
                    "iri": result.iri,
                    "similarity": getattr(result, 'similarity', 0.0)
                })

        print(f"   📋 Extracted {len(iris_info)} valid IRIs from this query")

    print(f"\n3. Focus on target query analysis:")
    print(f"   Target: '{target_query}'")

    # Deep analysis of the main query
    results = embedder.search_closest_labels(target_query, top_k=10)

    # Categorize results
    industry_related = []
    manufacturer_related = []
    count_related = []

    for result in results:
        label_lower = result.label.lower()
        if any(word in label_lower for word in ['industry', 'industrial', 'sector']):
            industry_related.append(result)
        elif any(word in label_lower for word in ['manufacturer', 'manufacturing', 'company']):
            manufacturer_related.append(result)
        elif any(word in label_lower for word in ['count', 'number', 'quantity']):
            count_related.append(result)

    print(f"   Industry-related matches: {len(industry_related)}")
    for result in industry_related[:3]:
        print(f"     - {result.label}")

    print(f"   Manufacturer-related matches: {len(manufacturer_related)}")
    for result in manufacturer_related[:3]:
        print(f"     - {result.label}")

    print(f"   Count-related matches: {len(count_related)}")
    for result in count_related[:3]:
        print(f"     - {result.label}")

    print("\n✅ Embedding matching and IRI extraction complete!")

if __name__ == "__main__":
    demo_embedding_matching()
