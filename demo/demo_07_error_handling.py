"""
Demo 7: Error Handling and Retry Mechanisms
Demonstrates how the pipeline handles errors and implements retry logic with error correction prompts.
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def demo_error_handling():
    """Walk through error taxonomies and iterative recovery strategies."""
    print("=" * 60)
    print("DEMO 7: Error Handling and Retry Mechanisms")
    print("=" * 60)

    # Sample error scenarios specific to the industry count query
    error_scenarios = {
        "missing_group_by": {
            "error": "Aggregate function COUNT used without GROUP BY clause",
            "original_query": """PREFIX sudokn: <http://asu.edu/semantics/SUDOKN/>
SELECT ?industryName (COUNT(?manufacturer) as ?count)
WHERE {
  ?manufacturer a sudokn:Company ;
                sudokn:serves ?industry .
  ?industry a sudokn:Industry ;
            rdfs:label ?industryName .
}""",
            "correction": "Add GROUP BY clause for proper aggregation"
        },

        "undefined_property": {
            "error": "Property 'servesIndustry' not found in ontology",
            "original_query": """PREFIX sudokn: <http://asu.edu/semantics/SUDOKN/>
SELECT ?industryName (COUNT(?manufacturer) as ?count)
WHERE {
  ?manufacturer a sudokn:Company ;
                sudokn:servesIndustry ?industry .
  ?industry rdfs:label ?industryName .
}
GROUP BY ?industryName""",
            "correction": "Use correct property 'sudokn:serves' from ontology"
        },

        "syntax_error": {
            "error": "Syntax error at line 4: Expected '.' after triple pattern",
            "original_query": """PREFIX sudokn: <http://asu.edu/semantics/SUDOKN/>
SELECT ?industryName (COUNT(?manufacturer) as ?count)
WHERE {
  ?manufacturer a sudokn:Company
  ?manufacturer sudokn:serves ?industry .
  ?industry rdfs:label ?industryName .
}
GROUP BY ?industryName""",
            "correction": "Add missing dot after first triple pattern"
        },

        "performance_timeout": {
            "error": "Query execution timeout after 60 seconds",
            "original_query": """PREFIX sudokn: <http://asu.edu/semantics/SUDOKN/>
SELECT ?industryName (COUNT(?manufacturer) as ?count)
WHERE {
  ?manufacturer a sudokn:Company ;
                sudokn:serves ?industry ;
                ?anyProperty ?anyValue .
  ?industry a sudokn:Industry ;
            rdfs:label ?industryName ;
            ?anyProperty2 ?anyValue2 .
}
GROUP BY ?industryName""",
            "correction": "Remove unnecessary broad pattern matching and add LIMIT clause"
        }
    }

    print(f"\n1. Error Detection and Classification for Industry Count Query:")
    print("-" * 40)

    target_query = "Get the number of manufacturers in the KG serving each type of Industry. Display the Industry name and corresponding count."
    print(f"   Target query: {target_query}")

    for scenario_name, scenario in error_scenarios.items():
        print(f"\n🚨 Scenario: {scenario_name}")
        print(f"   Error message: {scenario['error']}")
        print(f"   Suggested fix: {scenario['correction']}")

        # Classify error type
        error_msg = scenario['error'].lower()
        if "group by" in error_msg or "aggregate" in error_msg:
            error_type = "Aggregation Error"
            severity = "High"
        elif "property" in error_msg and "not found" in error_msg:
            error_type = "Ontology Error"
            severity = "High"
        elif "syntax" in error_msg:
            error_type = "Syntax Error"
            severity = "Medium"
        elif "timeout" in error_msg:
            error_type = "Performance Error"
            severity = "Low"
        else:
            error_type = "Unknown Error"
            severity = "Unknown"

        print(f"   Classification: {error_type} (Severity: {severity})")

    print(f"\n2. Error Correction Prompt Building for Counting Queries:")
    print("-" * 40)

    # Demonstrate error correction prompt template
    sample_error = error_scenarios["missing_group_by"]

    print(f"\n   Building error correction prompt for aggregation error...")

    # Mock template content specific to counting queries
    error_template = """You are an expert SPARQL query debugger specializing in aggregation queries.

ORIGINAL USER QUERY: Get the number of manufacturers in the KG serving each type of Industry. Display the Industry name and corresponding count.

ERROR MESSAGE:
{ERROR_MESSAGE}

PREVIOUS SPARQL QUERY:
{PREVIOUS_SPARQL}

EXTRACTED CLAUSES:
{EXTRACTED_CHUNK}

The query should perform COUNT aggregation and GROUP BY operations. Please analyze the error and provide a corrected SPARQL query. Focus on:
1. Proper aggregation syntax (COUNT with GROUP BY)
2. Correct ontology properties (sudokn:serves, not servesIndustry)
3. Valid class names (sudokn:Company, sudokn:Industry)
4. Proper variable binding and grouping

Output only the corrected SPARQL query, no explanation."""

    sample_extracted = {
        "clauses": [
            {"type": "aggregation", "subject_hint": "manufacturer", "relation_hint": "count", "object_hint": "number"},
            {"type": "grouping", "subject_hint": "manufacturer", "relation_hint": "serves", "object_hint": "industry"},
            {"type": "display", "subject_hint": "industry", "relation_hint": "display", "object_hint": "name, count"}
        ]
    }

    error_prompt = error_template.replace(
        "{ERROR_MESSAGE}", sample_error["error"]
    ).replace(
        "{PREVIOUS_SPARQL}", sample_error["original_query"]
    ).replace(
        "{EXTRACTED_CHUNK}", json.dumps(sample_extracted, separators=(',', ':'))
    )

    print(f"   ✅ Error correction prompt built")
    print(f"   Prompt length: {len(error_prompt)} characters")
    print(f"   Contains error: {'✓' if sample_error['error'] in error_prompt else '✗'}")
    print(f"   Contains query: {'✓' if sample_error['original_query'] in error_prompt else '✗'}")
    print(f"   Contains aggregation guidance: {'✓' if 'COUNT' in error_prompt and 'GROUP BY' in error_prompt else '✗'}")

    print(f"\n3. Retry Logic Simulation for Industry Count Query:")
    print("-" * 40)

    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        print(f"\n   Attempt {attempt}:")

        # Simulate different outcomes for each attempt with counting query errors
        if attempt == 1:
            print(f"     📤 Initial aggregation query execution")
            print(f"     ❌ Failed: {error_scenarios['missing_group_by']['error']}")
            print(f"     🔄 Generating GROUP BY correction prompt...")

        elif attempt == 2:
            print(f"     📤 Retry with GROUP BY clause added")
            print(f"     ❌ Failed: {error_scenarios['undefined_property']['error']}")
            print(f"     🔄 Generating ontology property correction...")

        elif attempt == 3:
            print(f"     📤 Second retry with correct properties")
            print(f"     ❌ Failed: {error_scenarios['syntax_error']['error']}")
            print(f"     🔄 Generating syntax correction...")

        elif attempt == 4:
            print(f"     📤 Third retry with syntax fixes")
            print(f"     ✅ Success: Query executed successfully")
            print(f"     📊 Returned industry counts: Automotive (15), Electronics (12), Chemical (8)")
            break
        else:
            print(f"     📤 Final attempt")
            print(f"     ❌ Failed: Maximum retries exceeded")

    print(f"\n4. Error Recovery Strategies for Counting Queries:")
    print("-" * 40)

    recovery_strategies = {
        "Aggregation Errors": [
            "Add proper GROUP BY clause with aggregated variables",
            "Use DISTINCT in COUNT if duplicate counting is an issue",
            "Ensure all non-aggregated variables are in GROUP BY",
            "Add ORDER BY for meaningful result ranking"
        ],
        "Ontology Errors": [
            "Use correct property names from reference turtle data",
            "Verify class names match ontology definitions",
            "Check namespace prefixes are properly defined",
            "Validate relationships exist in the knowledge graph"
        ],
        "Performance Errors": [
            "Add LIMIT clauses to reduce result size",
            "Remove broad pattern matching (?p ?o)",
            "Use more specific filters and constraints",
            "Consider using DISTINCT to reduce duplicates"
        ]
    }

    for error_type, strategies in recovery_strategies.items():
        print(f"\n   {error_type}:")
        for i, strategy in enumerate(strategies, 1):
            print(f"     {i}. {strategy}")

    print(f"\n5. Expected corrected query for industry count:")
    corrected_query = """PREFIX sudokn: <http://asu.edu/semantics/SUDOKN/>
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

    print(f"   Expected final query structure:")
    for i, line in enumerate(corrected_query.split('\n')[:8], 1):
        print(f"     {i}. {line}")
    print(f"     ... (complete query shows proper aggregation pattern)")

    print(f"\n6. Saving error correction examples:")
    for scenario_name, scenario in error_scenarios.items():
        # Save original query
        with open(f"demo_error_{scenario_name}_original.sparql", "w", encoding="utf-8") as f:
            f.write(f"-- Original Query with Error: {scenario_name}\n")
            f.write(f"-- Target: Get manufacturer count by industry\n")
            f.write(f"-- Error: {scenario['error']}\n\n")
            f.write(scenario["original_query"])

        # Save error message
        with open(f"demo_error_{scenario_name}_message.txt", "w", encoding="utf-8") as f:
            f.write(f"Error Type: {scenario_name}\n")
            f.write(f"Error Message: {scenario['error']}\n")
            f.write(f"Suggested Fix: {scenario['correction']}\n")
            f.write(f"Query Context: Industry manufacturer counting\n")

        print(f"   ✅ Saved error scenario: {scenario_name}")

    # Save error correction prompt template
    with open("demo_error_correction_prompt.txt", "w", encoding="utf-8") as f:
        f.write(error_prompt)
    print(f"   ✅ Saved error correction prompt template")

    print("\n✅ Error handling and retry mechanisms demo complete!")

if __name__ == "__main__":
    demo_error_handling()
