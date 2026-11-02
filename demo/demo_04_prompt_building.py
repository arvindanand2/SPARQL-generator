"""
Demo 4: Prompt Building and Template Processing
Demonstrates how the LLM prompt is built using templates and reference data.
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def demo_prompt_building():
    """Demonstrate prompt templating and sanity checks for SPARQL generation."""
    print("=" * 60)
    print("DEMO 4: Prompt Building and Template Processing")
    print("=" * 60)

    # Sample extracted chunk for the new query (simplified)
    sample_extracted = {
        "clauses": [
            {
                "type": "aggregation",
                "surface": "get the number of manufacturers",
                "subject_hint": "manufacturer",
                "relation_hint": "count",
                "object_hint": "number",
                "confidence": 0.85
            },
            {
                "type": "grouping",
                "surface": "serving each type of Industry",
                "subject_hint": "manufacturer",
                "relation_hint": "serves",
                "object_hint": "industry",
                "confidence": 0.82
            },
            {
                "type": "display",
                "surface": "Display the Industry name and corresponding count",
                "subject_hint": "industry",
                "relation_hint": "display",
                "object_hint": "name, count",
                "confidence": 0.78
            }
        ],
        "relevant_iris": [
            {
                "label": "Manufacturing Industry",
                "iri": "http://asu.edu/semantics/SUDOKN/ManufacturingIndustry"
            },
            {
                "label": "Company",
                "iri": "http://asu.edu/semantics/SUDOKN/Company"
            }
        ]
    }

    user_query = "Get the number of manufacturers in the KG serving each type of Industry. Display the Industry name and corresponding count."

    print(f"\n1. Input components:")
    print(f"   User query: '{user_query}'")
    print(f"   Extracted clauses: {len(sample_extracted['clauses'])} clauses")
    print(f"   Relevant IRIs: {len(sample_extracted.get('relevant_iris', []))} IRIs")

    print(f"\n2. Reading template file...")
    try:
        with open("../sparql_prompt_template.txt", "r", encoding="utf-8") as f:
            template = f.read()
        print(f"   ✅ Template loaded ({len(template)} characters)")
        print(f"   Template preview:")
        print(f"   {repr(template[:200])}...")

        print(f"\n   Placeholders found:")
        placeholders = ["{EXTRACTED_CHUNK}", "{REFERENCE_TURTLE_DATA}"]
        for placeholder in placeholders:
            if placeholder in template:
                print(f"   ✓ {placeholder}")
            else:
                print(f"   ✗ {placeholder} (missing)")

    except FileNotFoundError:
        print(f"   ❌ Template file not found")
        return

    print(f"\n3. Reading reference turtle data...")
    try:
        with open("../reference_turtle.txt", "r", encoding="utf-8") as f:
            reference_turtle_data = f.read()
        print(f"   ✅ Reference turtle loaded ({len(reference_turtle_data)} characters)")
        print(f"   Turtle preview:")
        lines = reference_turtle_data.split('\n')[:5]
        for line in lines:
            print(f"   {line}")
        if len(reference_turtle_data.split('\n')) > 5:
            print(f"   ... and {len(reference_turtle_data.split('\n')) - 5} more lines")

    except FileNotFoundError:
        print(f"   ❌ Reference turtle file not found")
        return

    print(f"\n4. Building final prompt...")

    # Convert extracted chunk to compact JSON
    chunk_str = json.dumps(sample_extracted, separators=(',', ':'))

    # Build prompt
    prompt = f"User Query: {user_query}\n" + template.replace(
        "{EXTRACTED_CHUNK}", chunk_str
    ).replace(
        "{REFERENCE_TURTLE_DATA}", reference_turtle_data
    )

    print(f"   ✅ Prompt built successfully!")
    print(f"   Total prompt length: {len(prompt)} characters")

    print(f"\n5. Prompt structure analysis:")
    prompt_sections = prompt.split('\n\n')
    for i, section in enumerate(prompt_sections[:3], 1):
        section_preview = section.replace('\n', ' ')[:100]
        print(f"   Section {i}: {section_preview}...")

    if len(prompt_sections) > 3:
        print(f"   ... and {len(prompt_sections) - 3} more sections")

    print(f"\n6. Query-specific analysis:")
    # Analyze how well the prompt handles the counting query
    prompt_lower = prompt.lower()
    has_aggregation = any(word in prompt_lower for word in ['count', 'number', 'sum', 'group by'])
    has_industry = 'industry' in prompt_lower
    has_manufacturer = 'manufacturer' in prompt_lower
    has_display = any(word in prompt_lower for word in ['display', 'show', 'list'])

    print(f"   Contains aggregation concepts: {'✓' if has_aggregation else '✗'}")
    print(f"   References industries: {'✓' if has_industry else '✗'}")
    print(f"   References manufacturers: {'✓' if has_manufacturer else '✗'}")
    print(f"   Has display instructions: {'✓' if has_display else '✗'}")

    print(f"\n7. Saving demo prompt to file...")
    demo_prompt_path = "demo_prompt_output.txt"
    with open(demo_prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"   ✅ Demo prompt saved to {demo_prompt_path}")

    print(f"\n8. Prompt statistics:")
    print(f"   Lines: {len(prompt.split('\\n'))}")
    print(f"   Words: {len(prompt.split())}")
    print(f"   Contains user query: {'✓' if user_query in prompt else '✗'}")
    print(f"   Contains extracted data: {'✓' if chunk_str in prompt else '✗'}")
    print(f"   Contains turtle data: {'✓' if reference_turtle_data[:50] in prompt else '✗'}")

    print(f"\n9. Expected SPARQL query pattern:")
    expected_sparql_pattern = """
    Expected output should be a SPARQL query like:
    PREFIX sudokn: <http://asu.edu/semantics/SUDOKN/>
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
    """
    print(f"   {expected_sparql_pattern}")

    print("\n✅ Prompt building demo complete!")

if __name__ == "__main__":
    demo_prompt_building()
