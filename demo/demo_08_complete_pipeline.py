"""
Demo 8: Complete Pipeline Integration
Demonstrates the full end-to-end pipeline execution with all components working together.
"""

import sys
import os
import subprocess
import json
import datetime
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from enhanced_csv_embedder import EnhancedCSVEmbedder
from compress_chunk import compact_chunk

def demo_complete_pipeline():
    """Tie the pipeline stages together for an end-to-end aggregation scenario."""
    print("=" * 60)
    print("DEMO 8: Complete Pipeline Integration")
    print("=" * 60)

    # Target query and variations for end-to-end demonstration
    target_query = "Get the number of manufacturers in the KG serving each type of Industry. Display the Industry name and corresponding count."

    test_queries = [
        target_query,
        "Count manufacturers by industry type",
        "Show industry breakdown with manufacturer counts"
    ]

    print(f"\n1. Pipeline Overview for Industry Count Queries:")
    print("-" * 30)
    pipeline_steps = [
        "1. Initialize embedder and load industry/manufacturer mappings",
        "2. Extract IRIs from industry count query",
        "3. Extract and compress aggregation clauses",
        "4. Build LLM prompt with COUNT/GROUP BY guidance",
        "5. Generate SPARQL aggregation query via LLM",
        "6. Execute COUNT query against GraphDB",
        "7. Handle aggregation errors and retry if needed",
        "8. Save all outputs including count results"
    ]

    for step in pipeline_steps:
        print(f"   {step}")

    print(f"\n2. Running complete pipeline demonstrations:")
    print("=" * 50)

    for query_idx, user_query in enumerate(test_queries, 1):
        print(f"\n🚀 PIPELINE RUN {query_idx}")
        print(f"Query: '{user_query}'")
        print("-" * 50)

        run_data = {"user_query": user_query}
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Step 1: Initialize embedder
        print(f"\n   Step 1: Embedder Initialization")
        try:
            embedder = EnhancedCSVEmbedder()
            embedder.load_iri_mapping("../label_iri_mappings.csv")

            csv_files = {
                "industries": "../industries_enhanced.csv",
                "material_caps": "../material_caps_enhanced.csv",
                "certificates": "../certificates_enhanced_fixed.csv",
                "process_embeddings": "../process_embeddings_enhanced_with_def.csv"
            }
            embedder.embed_csv_files(csv_files, force_recompute=False)
            print(f"   ✅ Embedder initialized with {len(embedder.embeddings)} categories")

        except Exception as e:
            print(f"   ❌ Embedder initialization failed: {e}")
            continue

        # Step 2: Extract IRIs
        print(f"\n   Step 2: IRI Extraction for Industry Count Query")
        try:
            results = embedder.search_closest_labels(user_query, top_k=5)
            iris_info = []
            industry_iris = []
            manufacturer_iris = []

            for result in results:
                if result.iri and "No IRI found" not in result.iri:
                    iris_info.append({
                        "label": result.label,
                        "iri": result.iri
                    })

                    # Categorize IRIs
                    if any(word in result.label.lower() for word in ['industry', 'industrial', 'sector']):
                        industry_iris.append(result.label)
                    elif any(word in result.label.lower() for word in ['manufacturer', 'company', 'manufacturing']):
                        manufacturer_iris.append(result.label)

            print(f"   ✅ Found {len(iris_info)} relevant IRIs")
            print(f"   📊 Industry-related: {len(industry_iris)} IRIs")
            print(f"   🏭 Manufacturer-related: {len(manufacturer_iris)} IRIs")
            for iri_data in iris_info[:3]:  # Show top 3
                print(f"     • {iri_data['label']}")

            run_data["extracted_iris"] = iris_info

        except Exception as e:
            print(f"   ❌ IRI extraction failed: {e}")
            run_data["extracted_iris"] = []

        # Step 3: Extract clauses
        print(f"\n   Step 3: Clause Extraction for Aggregation Query")
        try:
            result = subprocess.run([
                sys.executable, "../extract_chunk.py", user_query
            ], capture_output=True, text=True, timeout=600, cwd=os.path.dirname(__file__))

            if result.returncode == 0:
                extracted = json.loads(result.stdout)
                compressed = compact_chunk(extracted)

                # Add IRIs to compressed chunk
                if iris_info:
                    compressed["relevant_iris"] = iris_info

                # Analyze clause types for counting query
                aggregation_clauses = [c for c in compressed.get('clauses', []) if c.get('type') == 'aggregation']
                grouping_clauses = [c for c in compressed.get('clauses', []) if c.get('type') == 'grouping']
                display_clauses = [c for c in compressed.get('clauses', []) if c.get('type') == 'display']

                print(f"   ✅ Extracted {len(compressed.get('clauses', []))} clauses")
                print(f"   📊 Aggregation clauses: {len(aggregation_clauses)}")
                print(f"   📋 Grouping clauses: {len(grouping_clauses)}")
                print(f"   🖥️  Display clauses: {len(display_clauses)}")
                print(f"   📦 Compression ratio: {len(json.dumps(compressed))/len(json.dumps(extracted)):.2f}")

                run_data["extracted"] = compressed

            else:
                print(f"   ❌ Clause extraction failed: {result.stderr}")
                continue

        except Exception as e:
            print(f"   ❌ Clause extraction error: {e}")
            continue

        # Step 4: Build prompt
        print(f"\n   Step 4: Prompt Building for COUNT Query")
        try:
            # Read template
            with open("../sparql_prompt_template.txt", "r", encoding="utf-8") as f:
                template = f.read()

            # Read reference turtle data
            with open("../reference_turtle.txt", "r", encoding="utf-8") as f:
                reference_turtle_data = f.read()

            # Build prompt
            chunk_str = json.dumps(compressed, separators=(',', ':'))
            prompt = f"User Query: {user_query}\n" + template.replace(
                "{EXTRACTED_CHUNK}", chunk_str
            ).replace(
                "{REFERENCE_TURTLE_DATA}", reference_turtle_data
            )

            # Analyze prompt for aggregation context
            prompt_lower = prompt.lower()
            has_count_context = any(word in prompt_lower for word in ['count', 'number', 'aggregate'])
            has_group_context = any(word in prompt_lower for word in ['group', 'each', 'by'])
            has_industry_context = 'industry' in prompt_lower

            print(f"   ✅ Prompt built: {len(prompt)} characters")
            print(f"   📊 Contains count context: {'✓' if has_count_context else '✗'}")
            print(f"   📋 Contains grouping context: {'✓' if has_group_context else '✗'}")
            print(f"   🏭 Contains industry context: {'✓' if has_industry_context else '✗'}")

            run_data["llm_prompt"] = prompt

        except Exception as e:
            print(f"   ❌ Prompt building failed: {e}")
            continue

        # Step 5: Mock SPARQL generation for counting query
        print(f"\n   Step 5: SPARQL Generation for Industry Count (Mocked)")

        # Generate mock SPARQL based on counting query pattern
        if query_idx == 1:  # Main query
            mock_sparql = """PREFIX sudokn: <http://asu.edu/semantics/SUDOKN/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?industryName (COUNT(?manufacturer) as ?count)
WHERE {
  ?manufacturer a sudokn:Company ;
                sudokn:serves ?industry .
  ?industry a sudokn:Industry ;
            rdfs:label ?industryName .
}
GROUP BY ?industryName ?industry
ORDER BY DESC(?count)"""
        elif query_idx == 2:  # Variation with distinct
            mock_sparql = """PREFIX sudokn: <http://asu.edu/semantics/SUDOKN/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?industryType (COUNT(DISTINCT ?manufacturer) as ?manufacturerCount)
WHERE {
  ?manufacturer a sudokn:Company ;
                sudokn:serves ?industry .
  ?industry a sudokn:Industry ;
            sudokn:industryType ?industryType .
}
GROUP BY ?industryType
ORDER BY DESC(?manufacturerCount)"""
        else:  # Simple breakdown
            mock_sparql = """PREFIX sudokn: <http://asu.edu/semantics/SUDOKN/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?industryName (COUNT(?manufacturer) as ?count)
WHERE {
  ?manufacturer a sudokn:Company ;
                sudokn:serves ?industry .
  ?industry a sudokn:Industry ;
            rdfs:label ?industryName .
}
GROUP BY ?industryName ?industry
ORDER BY DESC(?count)
LIMIT 20"""

        # Analyze generated SPARQL for aggregation patterns
        sparql_analysis = {
            "has_count": "COUNT" in mock_sparql,
            "has_group_by": "GROUP BY" in mock_sparql,
            "has_order_by": "ORDER BY" in mock_sparql,
            "has_limit": "LIMIT" in mock_sparql,
            "variable_count": len([word for word in mock_sparql.split() if word.startswith('?')])
        }

        print(f"   ✅ SPARQL generated: {len(mock_sparql)} characters")
        print(f"   📊 Aggregation analysis:")
        for feature, present in sparql_analysis.items():
            status = "✓" if present else "✗"
            print(f"     {feature}: {status}")

        run_data["sparql_query"] = mock_sparql

        # Step 6: Mock GraphDB execution with count results
        print(f"\n   Step 6: GraphDB Execution for COUNT Query (Mocked)")

        # Simulate different outcomes for count queries
        if query_idx == 1:
            mock_result = """industryName,count
"Automotive Industry",23
"Electronics Manufacturing",18
"Chemical Processing",12
"Pharmaceutical Manufacturing",9
"Aerospace Industry",7
"Food Processing",5"""
            print(f"   ✅ Count query executed successfully")
            print(f"   📊 Returned 6 industry categories with manufacturer counts")
            print(f"   🏆 Top industry: Automotive (23 manufacturers)")

        elif query_idx == 2:
            mock_result = """industryType,manufacturerCount
"Heavy Manufacturing",45
"Light Manufacturing",28
"Specialized Manufacturing",15"""
            print(f"   ✅ Aggregation query executed successfully")
            print(f"   📊 Returned 3 industry types with distinct counts")

        else:
            mock_result = """industryName,count
"Automotive Industry",23
"Electronics Manufacturing",18"""
            print(f"   ✅ Limited count query executed successfully")
            print(f"   📊 Returned top 2 industries (LIMIT applied)")

        run_data["graphdb_result"] = mock_result

        # Step 7: Save outputs
        print(f"\n   Step 7: Output Saving for Industry Count Analysis")

        demo_dir = f"pipeline_run_{query_idx}_{timestamp}"
        os.makedirs(demo_dir, exist_ok=True)

        # Save all components
        files_saved = []

        # Standard files
        with open(os.path.join(demo_dir, "user_query.txt"), "w", encoding="utf-8") as f:
            f.write(user_query)
        files_saved.append("user_query.txt")

        with open(os.path.join(demo_dir, "extracted_iris.json"), "w", encoding="utf-8") as f:
            json.dump(run_data.get("extracted_iris", []), f, indent=2)
        files_saved.append("extracted_iris.json")

        with open(os.path.join(demo_dir, "extracted.json"), "w", encoding="utf-8") as f:
            json.dump(run_data.get("extracted", {}), f, indent=2)
        files_saved.append("extracted.json")

        with open(os.path.join(demo_dir, "llm_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(run_data.get("llm_prompt", ""))
        files_saved.append("llm_prompt.txt")

        with open(os.path.join(demo_dir, "sparql_query.sparql"), "w", encoding="utf-8") as f:
            f.write(f"-- Industry Count Query for: {user_query}\n\n")
            f.write(run_data.get("sparql_query", ""))
        files_saved.append("sparql_query.sparql")

        with open(os.path.join(demo_dir, "graphdb_result.csv"), "w", encoding="utf-8") as f:
            f.write(run_data.get("graphdb_result", ""))
        files_saved.append("graphdb_result.csv")

        with open(os.path.join(demo_dir, "model.txt"), "w", encoding="utf-8") as f:
            f.write("gpt-4.1 (mocked for demo)")
        files_saved.append("model.txt")

        # Additional analysis file for counting queries
        with open(os.path.join(demo_dir, "count_analysis.txt"), "w", encoding="utf-8") as f:
            f.write(f"Industry Count Query Analysis\n")
            f.write(f"============================\n\n")
            f.write(f"Query: {user_query}\n\n")
            f.write(f"Analysis Results:\n")
            f.write(f"- IRIs extracted: {len(run_data.get('extracted_iris', []))}\n")
            f.write(f"- Clauses identified: {len(run_data.get('extracted', {}).get('clauses', []))}\n")
            f.write(f"- Aggregation patterns: {'COUNT, GROUP BY' if 'COUNT' in run_data.get('sparql_query', '') else 'None'}\n")
            f.write(f"- Expected result: Industry names with manufacturer counts\n")
        files_saved.append("count_analysis.txt")

        print(f"   ✅ Saved {len(files_saved)} files to {demo_dir}/")

        # Step 8: Summary
        print(f"\n   🎯 Pipeline Summary for Query {query_idx} (Industry Count):")
        print(f"     - IRIs found: {len(run_data.get('extracted_iris', []))}")
        print(f"     - Clauses extracted: {len(run_data.get('extracted', {}).get('clauses', []))}")
        print(f"     - Prompt size: {len(run_data.get('llm_prompt', ''))} chars")
        print(f"     - SPARQL type: {'Aggregation' if 'COUNT' in run_data.get('sparql_query', '') else 'Simple'}")
        print(f"     - Result format: {'CSV with counts' if ',' in run_data.get('graphdb_result', '') else 'Text'}")
        print(f"     - Files saved: {len(files_saved)}")

    print(f"\n3. Pipeline Performance Summary for Industry Counting:")
    print("=" * 40)
    print(f"   Total queries processed: {len(test_queries)}")
    print(f"   Focus: Aggregation and counting operations")
    print(f"   Query type: Industry manufacturer analysis")
    print(f"   Expected SPARQL pattern: SELECT with COUNT and GROUP BY")
    print(f"   Output files per run: 8 files (including count_analysis.txt)")

    print(f"\n4. Industry Count Query Integration Points:")
    print("-" * 40)

    integration_points = {
        "Embedder → IRI Extraction": "Industry/manufacturer labels matched to ontology IRIs",
        "IRI Extraction → Clause Extraction": "Relevant IRIs enhance aggregation clause detection",
        "Clause Extraction → Prompt Building": "COUNT/GROUP BY clauses inserted into templates",
        "Prompt Building → LLM": "Aggregation-aware prompt sent for SPARQL generation",
        "LLM → GraphDB": "COUNT queries with proper GROUP BY executed against KG",
        "GraphDB → Output": "Industry count results and analysis files saved",
        "Error Handling → Retry": "Aggregation errors trigger specialized correction prompts"
    }

    for integration, description in integration_points.items():
        print(f"   {integration}: {description}")

    print(f"\n5. Expected Industry Count Results Pattern:")
    print("-" * 40)
    print(f"   Expected output format: CSV with industryName,count columns")
    print(f"   Expected sorting: Descending by manufacturer count")
    print(f"   Expected insights: Which industries have most manufacturers")
    print(f"   Expected use cases: Market analysis, supplier distribution")

    print("\n✅ Complete pipeline integration demo finished!")
    print(f"📁 Check the pipeline_run_* directories for detailed outputs")
    print(f"📊 Focus: Industry manufacturer counting and aggregation analysis")

if __name__ == "__main__":
    demo_complete_pipeline()
