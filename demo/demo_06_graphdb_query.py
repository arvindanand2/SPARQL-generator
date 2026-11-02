"""
Demo 6: GraphDB Query Execution and Results
Demonstrates how SPARQL queries are executed against GraphDB and results are processed.
"""

import sys
import os
import subprocess
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def demo_graphdb_query():
    """Demonstrate execution, analysis, and retry logic for SPARQL queries."""
    print("=" * 60)
    print("DEMO 6: GraphDB Query Execution and Results")
    print("=" * 60)

    # Sample SPARQL queries for the industry count testing
    test_queries = {
        "industry_count_basic": """PREFIX sudokn: <http://asu.edu/semantics/SUDOKN/>
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

        "industry_count_with_filter": """PREFIX sudokn: <http://asu.edu/semantics/SUDOKN/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?industryName (COUNT(DISTINCT ?manufacturer) as ?manufacturerCount)
WHERE {
  ?manufacturer a sudokn:Company ;
                sudokn:serves ?industry .
  ?industry a sudokn:Industry ;
            rdfs:label ?industryName .
  FILTER(STRLEN(?industryName) > 0)
}
GROUP BY ?industryName ?industry
ORDER BY DESC(?manufacturerCount)
LIMIT 10""",

        "simple_industry_list": """PREFIX sudokn: <http://asu.edu/semantics/SUDOKN/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?industry ?industryName
WHERE {
  ?industry a sudokn:Industry ;
            rdfs:label ?industryName .
}
ORDER BY ?industryName
LIMIT 5""",

        "invalid_aggregation": """SELECT ?industryName ?count
WHERE {
  ?manufacturer serves ?industry .
  ?industry hasName ?industryName .
  ?count = COUNT(?manufacturer) .
}"""
    }

    print(f"\n1. Testing GraphDB connectivity with industry count queries:")
    print("-" * 50)

    for query_name, sparql_query in test_queries.items():
        print(f"\n🔍 Testing: {query_name}")
        query_preview = sparql_query.split('WHERE')[0].strip().replace('\n', ' ')
        print(f"   Query preview: {query_preview[:80]}...")

        try:
            # Execute query via query_Graph.py
            result = subprocess.run([
                sys.executable, "../query_Graph.py", sparql_query
            ], capture_output=True, text=True, timeout=60, cwd=os.path.dirname(__file__))

            if result.returncode == 0 and not result.stderr:
                print(f"   ✅ Query executed successfully!")
                output = result.stdout.strip()

                # Analyze results for counting queries
                if output:
                    lines = output.split('\n')
                    print(f"   📊 Result: {len(lines)} lines returned")

                    # Show sample results
                    print(f"   Sample output:")
                    for i, line in enumerate(lines[:3]):
                        print(f"     {i+1}: {line}")
                    if len(lines) > 3:
                        print(f"     ... and {len(lines) - 3} more lines")

                    # Analyze result format for counting queries
                    if "count" in output.lower() or "COUNT" in output:
                        print(f"   📈 Contains count data - good for aggregation query")

                    # Try to detect CSV/tabular format
                    if ',' in output or '\t' in output:
                        print(f"   📋 Format: Tabular data detected")
                    elif output.startswith('{') or output.startswith('['):
                        try:
                            parsed = json.loads(output)
                            print(f"   📋 Format: JSON with {len(parsed)} items")
                        except:
                            print(f"   📋 Format: Text (possibly malformed JSON)")
                    else:
                        print(f"   📋 Format: Plain text")

                    # Check for expected industry count structure
                    if any(word in output.lower() for word in ['industry', 'manufacturing', 'automotive']):
                        print(f"   🏭 Contains industry-related data")
                else:
                    print(f"   📊 Result: Empty result set")
                    if "count" in query_name.lower():
                        print(f"   ⚠️  Empty result for count query - may indicate no data or query issue")

            else:
                error_msg = result.stderr if result.stderr else result.stdout
                print(f"   ❌ Query failed: {error_msg[:100]}...")

                # Analyze error type for counting queries
                error_lower = error_msg.lower()
                if "group by" in error_lower or "aggregate" in error_lower:
                    print(f"   🔍 Error type: Aggregation syntax error")
                elif "syntax" in error_lower:
                    print(f"   🔍 Error type: SPARQL syntax error")
                elif "timeout" in error_lower:
                    print(f"   🔍 Error type: Query timeout (complex aggregation?)")
                elif "connection" in error_lower:
                    print(f"   🔍 Error type: Connection error")
                else:
                    print(f"   🔍 Error type: Unknown")

        except subprocess.TimeoutExpired:
            print(f"   ⏱️  Query timed out (>60 seconds) - aggregation queries can be slow")
        except Exception as e:
            print(f"   ❌ Execution error: {e}")

    print(f"\n2. Retry mechanism simulation:")
    print("-" * 30)

    # Simulate retry logic
    sample_query = test_queries["simple_select"]
    max_retries = 3

    print(f"   Testing retry mechanism with {max_retries} attempts...")
    for attempt in range(1, max_retries + 1):
        print(f"   Attempt {attempt}:")

        # In real scenario, this would be actual query execution
        # For demo, we'll simulate different outcomes
        if attempt == 1:
            print(f"     ❌ Simulated failure: Connection timeout")
        elif attempt == 2:
            print(f"     ❌ Simulated failure: Syntax error")
        else:
            print(f"     ✅ Simulated success: Query executed")
            break

    print(f"\n3. Result processing patterns for counting queries:")
    print("-" * 30)

    # Show different result processing scenarios specific to counting
    sample_results = {
        "successful_count": """industryName,count
"Automotive Industry",15
"Electronics Manufacturing",12
"Chemical Processing",8
"Pharmaceutical",5""",
        "json_count": '[{"industryName": "Automotive Industry", "count": 15}, {"industryName": "Electronics Manufacturing", "count": 12}]',
        "empty_count": "",
        "aggregation_error": "ERROR: Aggregate function COUNT requires GROUP BY clause"
    }

    for result_type, sample_data in sample_results.items():
        print(f"\n   Result type: {result_type}")
        print(f"   Data length: {len(sample_data)} characters")
        print(f"   Processing for counting query:")

        if not sample_data:
            print(f"     - Empty result: No industries found or no manufacturer-industry relationships")
        elif sample_data.startswith("ERROR"):
            print(f"     - Error detected: Likely aggregation syntax issue")
            print(f"     - Trigger retry with corrected GROUP BY clause")
        elif "count" in sample_data.lower():
            print(f"     - Count data: Parse industry names and counts")
            print(f"     - Sort by count (descending) for ranking")
            if sample_data.startswith('['):
                print(f"     - JSON format: Extract industryName and count fields")
            else:
                print(f"     - CSV format: Parse header and data rows")
        else:
            print(f"     - Text format: Process line by line, look for patterns")

    print(f"\n4. Expected result analysis for target query:")
    target_query = "Get the number of manufacturers in the KG serving each type of Industry. Display the Industry name and corresponding count."
    print(f"   Target query: {target_query}")
    print(f"   Expected columns: industryName, count")
    print(f"   Expected sorting: By count (descending)")
    print(f"   Expected data types: String (industry name), Integer (count)")
    print(f"   Expected insights: Which industries have most manufacturers")

    print(f"\n5. Saving demo query results:")
    for query_name, sparql_query in test_queries.items():
        filename = f"demo_query_{query_name}.sparql"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"-- Demo Query: {query_name}\n")
            f.write(f"-- Description: Industry manufacturer count analysis\n")
            f.write(f"-- Target: Get manufacturer count by industry type\n\n")
            f.write(sparql_query)
        print(f"   ✅ Saved {filename}")

    print("\n✅ GraphDB query execution demo complete!")

if __name__ == "__main__":
    demo_graphdb_query()
