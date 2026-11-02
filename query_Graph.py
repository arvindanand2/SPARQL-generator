#!/usr/bin/env python3
"""Simple CLI wrapper around the GraphDB SPARQL endpoint with retry logging."""

import sys
import io
from SPARQLWrapper import SPARQLWrapper, JSON
import csv
import argparse
import traceback
import json
import re
from datetime import datetime

# Replace with your GraphDB SPARQL endpoint URL
endpoint_url = "http://localhost:7200/repositories/sudokn"

# Create a SPARQL client
sparql = SPARQLWrapper(endpoint_url)

def compress_error(traceback_str: str) -> str:
    """Extract a concise summary from a traceback string."""
    exception_lines = [line for line in traceback_str.splitlines() if re.match(r'^\w+(\.\w+)*:', line)]
    last_exception = exception_lines[-1] if exception_lines else ""
    response_match = re.search(r'(Response:\n.*)', traceback_str, re.DOTALL)
    response = response_match.group(1).strip() if response_match else ""
    return f"{last_exception}\n{response}" if last_exception or response else "No relevant error info found."

def test_graphdb_connection():
    """Test connection to GraphDB endpoint and print result."""
    try:
        sparql.setQuery("ASK { ?s ?p ?o }")
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
        if result.get('boolean') is True or result.get('boolean') is False:
            print("GraphDB connection successful.")
            return 0
        else:
            print("GraphDB connection failed: Unexpected response.")
            return 1
    except Exception as e:
        print(f"GraphDB connection failed: {e}")
        return 1

def _safe_stdout():
    """
    Return a text-writable stdout that works even when sys.stdout has no .buffer
    (e.g., when redirected by some environments).
    """
    try:
        # Most environments
        return io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', newline='')
    except Exception:
        # Fallback: assume sys.stdout is already text
        return sys.stdout

def main():
    """Parse CLI arguments, execute the SPARQL query, and emit results."""
    parser = argparse.ArgumentParser(description="Run a SPARQL query against the endpoint.")
    parser.add_argument("query", type=str, nargs="?", default="SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10",
                        help="SPARQL query to run.")
    parser.add_argument("--retries", type=int, default=1, help="Number of retries on error.")
    parser.add_argument("--test-connection", action="store_true", help="Test connection to GraphDB and exit.")
    parser.add_argument("--max-rows", type=int, default=10000, help="Maximum number of result rows to output (default: 10000).")
    parser.add_argument("--timeout", type=int, default=120, help="Query timeout in seconds (default: 120).")
    parser.add_argument("--output-file", type=str, default=None, help="Optional file to write results to (TSV format).")
    args = parser.parse_args()

    if args.test_connection:
        exit_code = test_graphdb_connection()
        sys.exit(exit_code)

    sparql_query = args.query
    retries = args.retries
    max_rows = args.max_rows
    timeout = args.timeout
    output_file = args.output_file
    reasoning_effort = "high"
    stats = []
    success = False

    for run in range(1, retries + 1):
        run_stat = {
            "run_number": run,
            "timestamp": datetime.now().isoformat(),
            "query": sparql_query,
            "reasoning_effort": reasoning_effort,
        }
        try:
            sparql.setQuery(sparql_query)
            sparql.setReturnFormat(JSON)
            sparql.setTimeout(timeout * 1000)  # milliseconds
            results = sparql.query().convert()

            # Check if this is an ASK query (returns boolean)
            if "boolean" in results:
                boolean_result = results["boolean"]
                answer = "YES" if boolean_result else "NO"
                run_stat["success"] = True
                run_stat["error"] = None
                run_stat["ask_result"] = boolean_result
                success = True

                if output_file:
                    with open(output_file, "w", encoding="utf-8") as out:
                        out.write(f"ASK Query Result: {answer}\n")
                else:
                    print(f"ASK Query Result: {answer}")

                stats.append(run_stat)
                break  # Exit loop on success

            # Handle SELECT queries (returns bindings)
            bindings = results.get("results", {}).get("bindings", [])
            run_stat["success"] = True
            run_stat["error"] = None
            success = True

            # Prepare output
            if bindings:
                var_names = list(bindings[0].keys())
                if output_file:
                    out = open(output_file, "w", encoding="utf-8", newline="")
                else:
                    out = _safe_stdout()
                writer = csv.writer(out, delimiter='\t', lineterminator='\n')
                writer.writerow(var_names)
                row_count = 0
                for result in bindings:
                    if row_count >= max_rows:
                        print(f"Warning: Result set truncated at {max_rows} rows.", file=sys.stderr)
                        break
                    row = [result.get(var, {}).get("value", "") for var in var_names]
                    writer.writerow(row)
                    row_count += 1
                if output_file:
                    out.close()
                else:
                    try:
                        out.flush()
                    except Exception:
                        pass
                print(f"Total rows output: {min(row_count, max_rows)}")
            else:
                print("No results found.")
            stats.append(run_stat)
            break  # Exit loop on success

        except Exception:
            tb_str = traceback.format_exc()
            compressed = compress_error(tb_str)
            run_stat["success"] = False
            run_stat["error"] = compressed
            stats.append(run_stat)
            print(f"Run {run} failed. Compressed error:\n{compressed}", file=sys.stderr)
            if run == retries:
                print("All retries failed.", file=sys.stderr)

    # Save all run stats to a file (JSONL: one JSON object per line)
    try:
        with open("query_Graph_stats.jsonl", "a", encoding="utf-8") as f:
            for s in stats:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Failed to write stats file: {e}", file=sys.stderr)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
