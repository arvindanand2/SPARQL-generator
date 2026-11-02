"""
Demo 5: LLM Interaction and SPARQL Generation
Demonstrates how prompts are sent to the LLM and SPARQL queries are generated.
Note: This demo uses a mock LLM response to avoid API calls.
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def demo_llm_interaction():
    """Illustrate prompt analysis and SPARQL response handling using mock data."""
    print("=" * 60)
    print("DEMO 5: LLM Interaction and SPARQL Generation")
    print("=" * 60)

    # Sample prompt for the new query (shortened for demo)
    sample_prompt = """User Query: Get the number of manufacturers in the KG serving each type of Industry. Display the Industry name and corresponding count.

# SPARQL Query Generation Prompt

You are an expert in semantic web and SPARQL query generation.

Instructions:
- Use the extracted clause data provided below to determine the logic and constraints for the SPARQL query.
- Use the Turtle data only for reference to predicate and class names. Do not invent new predicates or classes.
- Strictly follow the predicate and class naming conventions as shown in the Turtle data.
- Output only a valid SPARQL query. Do not include any explanation or commentary.

Extracted Clauses:
{"clauses":[{"type":"aggregation","surface":"get the number of manufacturers","subject_hint":"manufacturer","relation_hint":"count","object_hint":"number","confidence":0.85},{"type":"grouping","surface":"serving each type of Industry","subject_hint":"manufacturer","relation_hint":"serves","object_hint":"industry","confidence":0.82}]}

Reference Turtle Data:
@prefix sudokn: <http://asu.edu/semantics/SUDOKN/> .
sudokn:Company a owl:Class .
sudokn:Industry a owl:Class .
sudokn:serves a owl:ObjectProperty .
"""

    print(f"\n1. Input prompt analysis:")
    print(f"   Prompt length: {len(sample_prompt)} characters")
    print(f"   Lines: {len(sample_prompt.split('\\n'))}")
    print(f"   Contains instructions: {'✓' if 'Instructions:' in sample_prompt else '✗'}")
    print(f"   Contains extracted data: {'✓' if 'clauses' in sample_prompt else '✗'}")
    print(f"   Contains turtle data: {'✓' if '@prefix' in sample_prompt else '✗'}")

    print(f"\n2. System prompt configuration:")
    system_prompt = "You are an expert in semantic web and SPARQL query generation. Strictly follow the Turtle predicate and class naming conventions. Output only a valid SPARQL query, no explanation."
    print(f"   System prompt: {system_prompt[:80]}...")
    print(f"   Length: {len(system_prompt)} characters")

    print(f"\n3. Simulating LLM interaction:")
    print(f"   Model: gpt-4.1")
    print(f"   Response format: text")
    print(f"   Messages structure:")
    print(f"   - System message: {len(system_prompt)} chars")
    print(f"   - User message: {len(sample_prompt)} chars")

    # Mock LLM responses for the industry count query
    mock_responses = {
        "successful": """PREFIX sudokn: <http://asu.edu/semantics/SUDOKN/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?industryName (COUNT(?manufacturer) as ?count)
WHERE {
  ?manufacturer a sudokn:Company ;
                sudokn:serves ?industry .
  ?industry a sudokn:Industry ;
            rdfs:label ?industryName .
}
GROUP BY ?industryName ?industry
ORDER BY DESC(?count)""",
        "with_error": """SELECT ?industryName ?count
WHERE {
  ?manufacturer serves ?industry .
  ?industry hasName ?industryName .
}
GROUP BY ?industryName""",
        "complex": """PREFIX sudokn: <http://asu.edu/semantics/SUDOKN/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?industryName ?industryType (COUNT(DISTINCT ?manufacturer) as ?manufacturerCount)
WHERE {
  ?manufacturer a sudokn:Company ;
                sudokn:serves ?industry ;
                sudokn:hasCapability ?capability .
  ?industry a sudokn:Industry ;
            rdfs:label ?industryName ;
            sudokn:industryType ?industryType .
  FILTER(?industryType != "")
}
GROUP BY ?industryName ?industryType ?industry
ORDER BY ?industryType DESC(?manufacturerCount)"""
    }

    print(f"\n4. Sample LLM responses for industry counting query:")
    for scenario, response in mock_responses.items():
        print(f"\n   Scenario: {scenario}")
        lines = response.strip().split('\n')
        print(f"   Response length: {len(response)} characters, {len(lines)} lines")
        print(f"   Preview:")
        for i, line in enumerate(lines[:3]):
            print(f"     {i+1}: {line}")
        if len(lines) > 3:
            print(f"     ... and {len(lines) - 3} more lines")

        # Analyze response for counting query specifics
        has_prefix = "PREFIX" in response
        has_select = "SELECT" in response
        has_where = "WHERE" in response
        has_count = "COUNT" in response
        has_group_by = "GROUP BY" in response
        has_order = "ORDER BY" in response

        print(f"   Analysis:")
        print(f"     PREFIX({'✓' if has_prefix else '✗'}) SELECT({'✓' if has_select else '✗'}) WHERE({'✓' if has_where else '✗'})")
        print(f"     COUNT({'✓' if has_count else '✗'}) GROUP BY({'✓' if has_group_by else '✗'}) ORDER BY({'✓' if has_order else '✗'})")

        # Check for proper aggregation patterns
        if "COUNT" in response and "GROUP BY" in response:
            print(f"     ✅ Proper aggregation query structure")
        else:
            print(f"     ❌ Missing aggregation components")

    print(f"\n5. Response processing:")
    print(f"   - Strip whitespace from response")
    print(f"   - Validate SPARQL syntax (basic checks)")
    print(f"   - Extract query type (SELECT, CONSTRUCT, ASK, etc.)")
    print(f"   - Count prefixes and variables")

    sample_response = mock_responses["successful"]
    processed = sample_response.strip()

    print(f"\n6. Processed response analysis for counting query:")
    print(f"   Final length: {len(processed)} characters")
    print(f"   Prefixes found: {processed.count('PREFIX')}")
    print(f"   Variables found: {len([word for word in processed.split() if word.startswith('?')])}")
    print(f"   Query type: {'SELECT' if 'SELECT' in processed else 'Other'}")
    print(f"   Aggregation functions: {processed.count('COUNT')}")
    print(f"   Grouping clauses: {processed.count('GROUP BY')}")
    print(f"   Ordering clauses: {processed.count('ORDER BY')}")

    # Analyze expected results
    print(f"\n7. Expected query behavior analysis:")
    print(f"   Should return: Industry names with manufacturer counts")
    print(f"   Result format: Two columns (industryName, count)")
    print(f"   Grouping: By industry")
    print(f"   Sorting: By count (descending)")
    print(f"   Data source: Companies serving industries")

    # Save sample responses
    print(f"\n8. Saving sample responses...")
    for scenario, response in mock_responses.items():
        filename = f"demo_sparql_response_{scenario}.sparql"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"-- Industry Count Query Response: {scenario}\n")
            f.write(f"-- Query: Get manufacturer count by industry\n\n")
            f.write(response)
        print(f"   ✅ Saved {filename}")

    print("\n✅ LLM interaction and SPARQL generation demo complete!")

if __name__ == "__main__":
    demo_llm_interaction()
