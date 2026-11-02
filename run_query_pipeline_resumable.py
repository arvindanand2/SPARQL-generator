"""
End-to-end SPARQL pipeline with resume functionality using normal OpenAI API calls.
Enhanced with state persistence and multi-query processing.

This module orchestrates:
1. Clause extraction via direct API calls
2. SPARQL generation via direct API calls
3. Error correction via direct API calls (with retries)
4. Query execution against GraphDB

Features:
- Automatic state persistence and resume
- Process multiple queries from CSV file
- Resume from any interrupted stage (extraction, generation, correction)
- Uses normal OpenAI API instead of batch API for faster responses
"""

import subprocess
import json
import sys
import time
import os
import argparse
import csv
from time import sleep

import unicodedata
from openai import OpenAI
from dotenv import load_dotenv
from compress_chunk import compact_chunk
from enhanced_csv_embedder import EnhancedCSVEmbedder
import datetime
import re
from typing import Dict, List, Tuple, Optional

load_dotenv()

# --- Initialize embedder once for reuse ---
_embedder_instance = None

# ---------- System prompts (defaults) ----------
SYSTEM_PROMPT_INITIAL = (
    "You are an expert in semantic web and SPARQL query generation. "
    "Given the user request and reference Turtle, synthesize a single, valid SPARQL query. "
    "Strictly follow the ontology naming conventions present in the reference Turtle. "
    "Return ONLY the SPARQL query—no explanations."
)

SYSTEM_PROMPT_ERROR = (
    "You are an expert in semantic web and SPARQL debugging. "
    "You are given the last SPARQL query and the GraphDB error details. "
    "Diagnose the error, minimally correct the SPARQL (preserve intent), and return ONLY the corrected SPARQL. "
    "Honor the ontology naming conventions from the reference Turtle. No explanations."
)

SYSTEM_PROMPT_EXTRACT = (
    "You convert natural-language requests into a normalized JSON array of clauses for SPARQL planning."
)


class PipelineState:
    """Manages pipeline state persistence and recovery."""

    def __init__(self, state_dir: str):
        self.state_dir = state_dir
        self.state_file = os.path.join(state_dir, "pipeline_state.json")
        os.makedirs(state_dir, exist_ok=True)

    def state_exists(self) -> bool:
        """Check if a state file exists."""
        return os.path.exists(self.state_file)

    def save_state(self, state: dict):
        """Save current pipeline state."""
        try:
            state["last_updated"] = datetime.datetime.now().isoformat()
            # Write to temporary file first to avoid corruption
            temp_file = self.state_file + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            # Atomic rename
            if os.path.exists(self.state_file):
                backup_file = self.state_file + ".backup"
                try:
                    os.replace(self.state_file, backup_file)
                except Exception as e:
                    print(f"⚠️  Warning: Could not create backup: {e}", file=sys.stderr)
            os.replace(temp_file, self.state_file)
            print(f"💾 State saved to {self.state_file}")
        except PermissionError as e:
            print(f"❌ ERROR: Permission denied saving state to {self.state_file}: {e}", file=sys.stderr)
            raise RuntimeError(f"Cannot save state - permission denied: {e}")
        except OSError as e:
            print(f"❌ ERROR: Disk error saving state (disk full?): {e}", file=sys.stderr)
            raise RuntimeError(f"Cannot save state - disk error: {e}")
        except Exception as e:
            print(f"❌ ERROR: Failed to save state: {e}", file=sys.stderr)
            raise

    def load_state(self) -> Optional[dict]:
        """Load pipeline state if exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                print(f"📂 Loaded state from {self.state_file}")
                # Validate state structure
                if not isinstance(state, dict):
                    raise ValueError("State file is not a valid dictionary")
                if "queries" not in state:
                    print("⚠️  Warning: State missing 'queries' key, initializing empty", file=sys.stderr)
                    state["queries"] = {}
                return state
            except json.JSONDecodeError as e:
                print(f"❌ ERROR: Corrupted state file (invalid JSON): {e}", file=sys.stderr)
                # Try to load backup
                backup_file = self.state_file + ".backup"
                if os.path.exists(backup_file):
                    print(f"🔄 Attempting to load backup state from {backup_file}", file=sys.stderr)
                    try:
                        with open(backup_file, "r", encoding="utf-8") as f:
                            state = json.load(f)
                        print(f"✅ Loaded backup state successfully", file=sys.stderr)
                        return state
                    except Exception as backup_error:
                        print(f"❌ Backup state also corrupted: {backup_error}", file=sys.stderr)
                print(f"⚠️  Starting fresh - previous state could not be recovered", file=sys.stderr)
                return None
            except FileNotFoundError:
                print(f"⚠️  State file not found: {self.state_file}", file=sys.stderr)
                return None
            except PermissionError as e:
                print(f"❌ ERROR: Permission denied reading state: {e}", file=sys.stderr)
                return None
            except Exception as e:
                print(f"❌ ERROR: Failed to load state: {e}", file=sys.stderr)
                return None
        return None


def compress_error(traceback_str: str) -> str:
    """Condense a traceback to its headline exception and response snippet."""
    exception_lines = [line for line in traceback_str.splitlines() if re.match(r'^\w+(\.\w+)*:', line)]
    last_exception = exception_lines[-1] if exception_lines else ""
    response_match = re.search(r'(Response:\n.*)', traceback_str, re.DOTALL)
    response = response_match.group(1).strip() if response_match else ""
    return f"{last_exception}\n{response}" if last_exception or response else "No relevant error info found."


def get_embedder() -> EnhancedCSVEmbedder:
    """Lazily instantiate and warm the enhanced CSV embedder."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = EnhancedCSVEmbedder()
        _embedder_instance.load_iri_mapping("label_iri_mappings.csv")
        csv_files = {
            "industries": "industries_enhanced.csv",
            "material_caps": "material_caps_enhanced.csv",
            "certificates": "certificates_enhanced_fixed.csv",
            "process_embeddings": "process_embeddings_enhanced_with_def.csv",
        }
        _embedder_instance.embed_csv_files(csv_files, force_recompute=False)
    return _embedder_instance


def extract_iris_from_query(user_query: str, top_k: int = 5):
    """Leverage the embedder to surface likely IRIs for the user query."""
    try:
        embedder = get_embedder()
        results = embedder.search_closest_labels(user_query, top_k=top_k)
        iris_info = []
        for result in results:
            if getattr(result, "iri", None) and "No IRI found" not in result.iri:
                iris_info.append({"label": result.label, "iri": result.iri})
        return iris_info
    except Exception as e:
        print(f"Error extracting IRIs from query: {e}", file=sys.stderr)
        return []


def extract_chunk_direct(
    user_query: str,
    client: OpenAI,
    model: str
) -> Dict:
    """
    Extract clause chunks for a single query using direct API call.

    Args:
        user_query: Query text
        client: OpenAI client instance
        model: Model name to use

    Returns:
        Extracted chunks dictionary
    """
    # Load schema and prompt
    try:
        with open("clause_chunk_schema.json", encoding="utf-8") as f:
            clause_chunk_schema = json.load(f)
    except FileNotFoundError:
        print("❌ ERROR: Required file 'clause_chunk_schema.json' not found", file=sys.stderr)
        raise RuntimeError("Missing required file: clause_chunk_schema.json")
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in clause_chunk_schema.json: {e}", file=sys.stderr)
        raise RuntimeError(f"Corrupted schema file: {e}")

    try:
        with open("clause_chunker_prompt_new.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print("❌ ERROR: Required file 'clause_chunker_prompt_new.txt' not found", file=sys.stderr)
        raise RuntimeError("Missing required file: clause_chunker_prompt_new.txt")

    # Make API call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract clauses from this query into the schema. Query:\n{user_query}"}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ExtractedClauses",
                    "strict": True,
                    "schema": clause_chunk_schema
                }
            }
        )

        content = response.choices[0].message.content
        extracted = json.loads(content)
        compressed = compact_chunk(extracted)

        # Add IRIs
        print(f"🔍 Extracting relevant IRIs from query...")
        iris_info = extract_iris_from_query(user_query, top_k=10)
        if iris_info:
            compressed["relevant_iris"] = iris_info
            print(f"✅ Found {len(iris_info)} relevant IRIs")
        else:
            compressed["relevant_iris"] = []

        return compressed

    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON from extraction: {e}", file=sys.stderr)
        raise RuntimeError(f"JSON parse error: {e}")
    except Exception as e:
        print(f"❌ Extraction API call failed: {e}", file=sys.stderr)
        raise RuntimeError(f"Extraction failed: {e}")


def build_llm_prompt(
    extracted_chunk,
    user_query: str,
    relevant_iris=None,
    template_path="sparql_prompt_template_new.txt",
    reference_turtle_path="reference_turtle.txt",
) -> str:
    """Render the initial SPARQL generation prompt from templates."""
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        print(f"❌ ERROR: Template file not found: {template_path}", file=sys.stderr)
        print(f"⚠️  Using fallback minimal prompt", file=sys.stderr)
        return json.dumps({"user_query": user_query, "extracted": extracted_chunk}, ensure_ascii=False)
    except Exception as e:
        print(f"❌ ERROR: Failed to read template file: {e}", file=sys.stderr)
        return json.dumps({"user_query": user_query, "extracted": extracted_chunk}, ensure_ascii=False)

    try:
        with open(reference_turtle_path, "r", encoding="utf-8") as f:
            reference_turtle_data = f.read()
    except FileNotFoundError:
        print(f"⚠️  Warning: Reference turtle file not found: {reference_turtle_path}", file=sys.stderr)
        reference_turtle_data = "# Reference turtle data not available"
    except Exception as e:
        print(f"⚠️  Warning: Failed to read reference turtle: {e}", file=sys.stderr)
        reference_turtle_data = "# Reference turtle data not available"

    try:
        chunk_str = json.dumps(extracted_chunk, separators=(",", ":"))
        iris_str = json.dumps(relevant_iris, separators=(",", ":")) if relevant_iris is not None else "[]"
        txt_body = template.replace("{USER_QUERY}", user_query)
        txt_body = txt_body.replace("{EXTRACTED_Clauses}", chunk_str)
        txt_body = txt_body.replace("{EXTRACTED_IRI}", iris_str)
        txt_body = txt_body.replace("{REFERENCE_TURTLE_DATA}", reference_turtle_data)
        return template.replace("{TXT}", txt_body) if "{TXT}" in template else txt_body
    except Exception as e:
        print(f"❌ ERROR: Failed to build LLM prompt: {e}", file=sys.stderr)
        return json.dumps({"user_query": user_query, "extracted": extracted_chunk}, ensure_ascii=False)


def build_error_correction_prompt(
    error_message: str,
    previous_sparql: str,
    extracted_chunk,
    user_query: str,
    relevant_iris=None,
    template_path="sparql_error_correction_prompt.txt",
    reference_turtle_path="reference_turtle.txt",
) -> str:
    """Render an error correction prompt using the previous attempt context."""
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
    except FileNotFoundError:
        print(f"⚠️  Warning: Error correction template not found: {template_path}", file=sys.stderr)
        return f"Error: {error_message}\nPrevious SPARQL: {previous_sparql}\nPlease fix the SPARQL query."
    except Exception as e:
        print(f"⚠️  Warning: Failed to read error correction template: {e}", file=sys.stderr)
        return f"Error: {error_message}\nPrevious SPARQL: {previous_sparql}"

    try:
        with open(reference_turtle_path, "r", encoding="utf-8") as f:
            reference_turtle_data = f.read()
    except FileNotFoundError:
        print(f"⚠️  Warning: Reference turtle file not found: {reference_turtle_path}", file=sys.stderr)
        reference_turtle_data = "# Reference turtle data not available"
    except Exception as e:
        print(f"⚠️  Warning: Failed to read reference turtle: {e}", file=sys.stderr)
        reference_turtle_data = "# Reference turtle data not available"

    try:
        chunk_str = json.dumps(extracted_chunk, separators=(",", ":"))
        iris_str = json.dumps(relevant_iris, separators=(",", ":")) if relevant_iris is not None else "[]"
        txt_body = template.replace("{USER_QUERY}", user_query)
        txt_body = txt_body.replace("{ERROR_MESSAGE}", error_message)
        txt_body = txt_body.replace("{PREVIOUS_SPARQL}", previous_sparql)
        txt_body = txt_body.replace("{EXTRACTED_Clauses}", chunk_str)
        txt_body = txt_body.replace("{EXTRACTED_IRI}", iris_str)
        txt_body = txt_body.replace("{REFERENCE_TURTLE_DATA}", reference_turtle_data)
        return template.replace("{TXT}", txt_body) if "{TXT}" in template else txt_body
    except Exception as e:
        print(f"❌ ERROR: Failed to build error correction prompt: {e}", file=sys.stderr)
        return f"Error: {error_message}\nPrevious SPARQL: {previous_sparql}"


def get_sparql_from_llm(
    prompt: str,
    client: OpenAI,
    model: str,
    system_prompt: str
) -> str:
    """
    Call OpenAI chat completions to obtain SPARQL text using direct API call.

    Args:
        prompt: User prompt
        client: OpenAI client instance
        model: Model name
        system_prompt: System prompt

    Returns:
        SPARQL query string
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "text"},
        )
        content = response.choices[0].message.content or ""
        return content.strip()
    except Exception as e:
        print(f"❌ LLM API call failed: {e}", file=sys.stderr)
        raise RuntimeError(f"LLM call failed: {e}")


def run_query_graph_with_retries(sparql_query: str, max_retries: int = 5):
    """Execute the SPARQL query script with simple retry logic.

    Returns:
        tuple: (result, error_msgs, sleep_time) where sleep_time is the total time spent in sleep calls
    """
    sleep_time = 0.0
    sleep_time += 2.0
    sleep(2)
    error_msgs = []
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                [sys.executable, "query_Graph.py", sparql_query],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
            )

            # Check if stderr contains only warnings (not actual errors)
            stderr_is_warning = False
            if result.stderr:
                stderr_lower = result.stderr.lower()
                # Truncation warnings are not errors
                if "warning: result set truncated" in stderr_lower or "result set truncated" in stderr_lower:
                    stderr_is_warning = True

            if result.returncode == 0 and (not result.stderr or stderr_is_warning):
                print("GraphDB result:\n", result.stdout)
                if stderr_is_warning:
                    print(f"Note: {result.stderr.strip()}", file=sys.stderr)
                return result.stdout, error_msgs, sleep_time
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                error_msgs.append(error_msg)
                print(f"Attempt {attempt+1} failed: {error_msg}", file=sys.stderr)
                sleep_time += 1.0
                time.sleep(1)
        except subprocess.TimeoutExpired:
            timeout_error = (
                "Query execution timeout (600 seconds exceeded). "
                "The SPARQL query is likely too complex or inefficient. "
                "Consider:\n"
                "1. Adding more specific filters to reduce result set size\n"
                "2. Avoiding cartesian products (multiple OPTIONAL clauses without shared variables)\n"
                "3. Simplifying complex property paths or nested queries"
            )
            error_msgs.append(timeout_error)
            print(f"Attempt {attempt+1} timed out after 600 seconds", file=sys.stderr)
            sleep_time += 1.0
            time.sleep(1)
        except FileNotFoundError as e:
            error_msg = f"ERROR: query_Graph.py not found or Python executable issue: {e}"
            print(f"❌ {error_msg}", file=sys.stderr)
            error_msgs.append(error_msg)
            break
        except Exception as e:
            error_msg = f"ERROR: Subprocess execution failed: {e}"
            print(f"❌ {error_msg}", file=sys.stderr)
            error_msgs.append(error_msg)
            if attempt < max_retries - 1:
                sleep_time += 1.0
                time.sleep(1)
    print(f"All {max_retries} attempts failed.", file=sys.stderr)
    return None, error_msgs, sleep_time


def check_graphdb_connection():
    """Check connection to GraphDB by invoking query_Graph.py with --test-connection."""
    try:
        result = subprocess.run(
            [sys.executable, "query_Graph.py", "--test-connection"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            print("ERROR: Could not connect to GraphDB.\n" + result.stdout + result.stderr, file=sys.stderr)
            sys.exit(1)
        else:
            print(result.stdout.strip())
    except FileNotFoundError:
        print("❌ ERROR: query_Graph.py not found in current directory", file=sys.stderr)
        print("   Please ensure query_Graph.py exists in the same directory as this script", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("❌ ERROR: GraphDB connection test timed out after 30 seconds", file=sys.stderr)
        print("   GraphDB may be unreachable or query_Graph.py is hung", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Failed to test GraphDB connection: {e}", file=sys.stderr)
        sys.exit(1)


def load_queries_from_csv(csv_path: str) -> List[Tuple[int, str, str]]:
    """
    Load queries from CSV file with comprehensive error handling.

    Returns:
        List of (query_id, query_text, expected_label) tuples
    """
    try:
        queries = []
        # Try multiple encodings
        encodings = ['utf-8', 'cp1252', 'latin-1']
        last_error = None

        for encoding in encodings:
            try:
                with open(csv_path, 'r', encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    for idx, row in enumerate(reader):
                        try:
                            # Handle different CSV formats
                            query_text = row.get('Question') or row.get('question') or row.get('query', '')
                            expected_label = row.get('Category') or row.get('category') or row.get('label', '')

                            if not query_text:
                                print(f"⚠️  Warning: Row {idx} has no query text, skipping", file=sys.stderr)
                                continue

                            queries.append((idx, query_text, expected_label))
                        except Exception as e:
                            print(f"⚠️  Warning: Failed to parse row {idx}: {e}", file=sys.stderr)
                            continue

                if queries:
                    print(f"📚 Loaded {len(queries)} queries from {csv_path} using {encoding} encoding")
                    return queries

            except UnicodeDecodeError as e:
                last_error = e
                continue

        # If we got here, all encodings failed
        if queries:
            print(f"📚 Loaded {len(queries)} queries from {csv_path}")
            return queries
        elif last_error:
            raise last_error
        else:
            raise ValueError(f"No valid queries found in {csv_path}")

    except FileNotFoundError:
        print(f"❌ ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        raise RuntimeError(f"CSV file not found: {csv_path}")
    except PermissionError as e:
        print(f"❌ ERROR: Permission denied reading CSV file: {e}", file=sys.stderr)
        raise RuntimeError(f"Cannot read CSV file: {e}")
    except csv.Error as e:
        print(f"❌ ERROR: Malformed CSV file: {e}", file=sys.stderr)
        raise RuntimeError(f"Invalid CSV format: {e}")
    except UnicodeDecodeError as e:
        print(f"❌ ERROR: Encoding issue in CSV file: {e}", file=sys.stderr)
        print(f"   Try saving the CSV with UTF-8 encoding", file=sys.stderr)
        raise RuntimeError(f"CSV encoding error: {e}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load CSV file: {e}", file=sys.stderr)
        raise RuntimeError(f"Cannot load CSV: {e}")


def save_query_log(user_query: str, sparql_query: str, log_path: str = "query_log.json"):
    """Append the user query and resulting SPARQL to the log file."""
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"user_query": user_query, "sparql_query": sparql_query}, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Failed to save query log: {e}", file=sys.stderr)


def _ensure_dir(path: str):
    """Ensure the target directory exists."""
    os.makedirs(path, exist_ok=True)


def save_query_outputs(query_id: int, run_data: dict, query_dir: str):
    """Save outputs for a single query."""
    _ensure_dir(query_dir)

    def _write(path, content):
        """Write content to a path while catching IO errors."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"Failed to write {path}: {e}", file=sys.stderr)

    _write(os.path.join(query_dir, "user_query.txt"), run_data.get("user_query", ""))
    _write(os.path.join(query_dir, "llm_prompt.txt"), run_data.get("llm_prompt", ""))
    _write(os.path.join(query_dir, "sparql_query.txt"), run_data.get("sparql_query", ""))
    _write(os.path.join(query_dir, "graphdb_result.txt"), run_data.get("graphdb_result", ""))

    try:
        with open(os.path.join(query_dir, "extracted.json"), "w", encoding="utf-8") as f:
            json.dump(run_data.get("extracted", {}), f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to write extracted.json: {e}", file=sys.stderr)

    if run_data.get("error_log"):
        _write(os.path.join(query_dir, "error_log.txt"), run_data["error_log"])


def save_attempt_outputs(query_dir: str, attempt_idx: int, sparql_query: str, result: str = None,
                         error_msg: str = None, error_correction_prompt: str = None):
    """Persist per-attempt SPARQL runs, including errors and correction prompts."""
    attempt_dir = os.path.join(query_dir, f"attempt_{attempt_idx}")
    _ensure_dir(attempt_dir)

    def _write(path, content):
        """Write helper used for attempt artifacts."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content or "")
        except Exception as e:
            print(f"Failed to write {path}: {e}", file=sys.stderr)

    _write(os.path.join(attempt_dir, "sparql_query.txt"), sparql_query)
    if result is not None:
        _write(os.path.join(attempt_dir, "graphdb_result.txt"), result)
    if error_msg is not None:
        _write(os.path.join(attempt_dir, "error.txt"), error_msg)
    if error_correction_prompt is not None:
        _write(os.path.join(attempt_dir, "error_correction_prompt.txt"), error_correction_prompt)


def _read_text_file(path: str) -> str:
    """Read a UTF-8 text file entirely."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def process_single_query(
    query_id: int,
    query_text: str,
    expected_label: str,
    client: OpenAI,
    model_name: str,
    system_initial: str,
    system_error: str,
    max_attempts: int,
    run_dir: str,
    state_manager: PipelineState,
    state: dict
) -> Dict:
    """
    Process a single query through the full pipeline with state management.

    Returns:
        Result dictionary with success status and details
    """
    # Start timing for this query
    query_start_time = time.time()
    timing_breakdown = {
        "extraction": 0.0,
        "initial_sparql": 0.0,
        "validation": 0.0,
        "correction": 0.0
    }

    query_state = state["queries"].get(str(query_id), {})

    # Set up query directory
    base_query_dir = os.path.join(run_dir, f"query_{query_id}")
    sanitized_category = sanitize_filename(expected_label) if expected_label else ""
    query_dir = f"{base_query_dir}_{sanitized_category}" if sanitized_category else base_query_dir
    os.makedirs(query_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"PROCESSING QUERY {query_id}: {query_text[:80]}...")
    print(f"{'='*80}")

    # ========== STAGE 1: EXTRACT CLAUSES ==========
    if query_state.get("stage") in ["extraction_complete", "initial_sparql_complete", "complete"]:
        print(f"✅ Using cached extraction for query {query_id}")
        extracted = query_state["extracted"]
    else:
        print(f"\n🔄 Extracting clauses for query {query_id}...")
        extraction_start = time.time()
        try:
            extracted = extract_chunk_direct(query_text, client, model_name)
            timing_breakdown["extraction"] = time.time() - extraction_start
            query_state["extracted"] = extracted
            query_state["stage"] = "extraction_complete"
            state["queries"][str(query_id)] = query_state
            state_manager.save_state(state)
            print(f"✅ Extracted clauses for query {query_id} (took {timing_breakdown['extraction']:.2f}s)")
        except Exception as e:
            print(f"❌ Extraction failed for query {query_id}: {e}", file=sys.stderr)
            total_time = time.time() - query_start_time
            return {
                "success": False,
                "result": None,
                "query": query_text,
                "expected_label": expected_label,
                "error": f"Extraction failed: {e}",
                "attempts": 0,
                "total_time": total_time,
                "timing_breakdown": timing_breakdown
            }

    # ========== STAGE 2: GENERATE INITIAL SPARQL ==========
    if query_state.get("stage") in ["initial_sparql_complete", "complete"]:
        print(f"✅ Using cached SPARQL for query {query_id}")
        sparql_query = query_state["sparql_query"]
        llm_prompt = query_state.get("llm_prompt", "")
        current_attempt = query_state.get("current_attempt", 1)
    else:
        print(f"\n🔄 Generating initial SPARQL for query {query_id}...")
        sparql_gen_start = time.time()
        try:
            relevant_iris = extracted.get("relevant_iris", [])
            llm_prompt = build_llm_prompt(extracted, query_text, relevant_iris=relevant_iris)

            sparql_query = get_sparql_from_llm(llm_prompt, client, model_name, system_initial)

            timing_breakdown["initial_sparql"] = time.time() - sparql_gen_start
            query_state["llm_prompt"] = llm_prompt
            query_state["sparql_query"] = sparql_query
            query_state["stage"] = "initial_sparql_complete"
            query_state["current_attempt"] = 1
            state["queries"][str(query_id)] = query_state
            state_manager.save_state(state)
            print(f"✅ Generated initial SPARQL for query {query_id} (took {timing_breakdown['initial_sparql']:.2f}s)")
            current_attempt = 1
        except Exception as e:
            print(f"❌ SPARQL generation failed for query {query_id}: {e}", file=sys.stderr)
            total_time = time.time() - query_start_time
            return {
                "success": False,
                "result": None,
                "query": query_text,
                "expected_label": expected_label,
                "error": f"SPARQL generation failed: {e}",
                "attempts": 0,
                "total_time": total_time,
                "timing_breakdown": timing_breakdown
            }

    # ========== STAGE 3: VALIDATE AND ITERATIVELY CORRECT ==========
    print(f"\n🔄 Validating and correcting query {query_id}...")

    for attempt in range(current_attempt, max_attempts + 1):
        print(f"\n--- Attempt {attempt}/{max_attempts} for query {query_id} ---")

        # Check if we already validated this attempt
        if query_state.get(f"attempt_{attempt}_result"):
            result = query_state[f"attempt_{attempt}_result"]
            if result != "error":
                print(f"✅ Query {query_id}: Using cached successful result from attempt {attempt}")
                total_time = time.time() - query_start_time
                return {
                    "success": True,
                    "result": result,
                    "query": query_text,
                    "expected_label": expected_label,
                    "sparql": sparql_query,
                    "attempts": attempt,
                    "total_time": total_time,
                    "timing_breakdown": timing_breakdown
                }
            else:
                # Error was cached, continue to correction
                error_msg = query_state.get(f"attempt_{attempt}_error", "Unknown error")
                print(f"⚠️  Using cached error from attempt {attempt}")
        else:
            # Execute query against GraphDB
            validation_start = time.time()
            result, query_errors, sleep_time = run_query_graph_with_retries(sparql_query, max_retries=1)
            validation_time = time.time() - validation_start
            # Subtract sleep time from validation time to get actual processing time
            timing_breakdown["validation"] += max(0, validation_time - sleep_time)

            save_attempt_outputs(query_dir, attempt, sparql_query)

            if result is not None:
                # Success!
                total_time = time.time() - query_start_time
                print(f"✅ Query {query_id}: Succeeded on attempt {attempt}")
                print(f"⏱️  Total time: {total_time:.2f}s (extraction: {timing_breakdown['extraction']:.2f}s, "
                      f"SPARQL gen: {timing_breakdown['initial_sparql']:.2f}s, "
                      f"validation: {timing_breakdown['validation']:.2f}s, "
                      f"correction: {timing_breakdown['correction']:.2f}s)")

                # Save success to state
                query_state[f"attempt_{attempt}_result"] = result
                query_state["stage"] = "complete"
                query_state["success"] = True
                query_state["sparql_query"] = sparql_query
                query_state["total_time"] = total_time
                query_state["timing_breakdown"] = timing_breakdown
                state["queries"][str(query_id)] = query_state
                state_manager.save_state(state)

                # Save outputs
                run_data = {
                    "user_query": query_text,
                    "query_id": query_id,
                    "extracted": extracted,
                    "llm_prompt": llm_prompt,
                    "sparql_query": sparql_query,
                    "graphdb_result": result,
                    "total_time": total_time,
                    "timing_breakdown": timing_breakdown
                }
                save_query_outputs(query_id, run_data, query_dir)
                save_attempt_outputs(query_dir, attempt, sparql_query, result=result)
                save_query_log(query_text, sparql_query)

                return {
                    "success": True,
                    "result": result,
                    "query": query_text,
                    "expected_label": expected_label,
                    "sparql": sparql_query,
                    "attempts": attempt,
                    "total_time": total_time,
                    "timing_breakdown": timing_breakdown
                }
            else:
                # Failed - prepare for correction
                error_msg = query_errors[-1] if query_errors else "Unknown error occurred"
                print(f"❌ Query {query_id}: Failed on attempt {attempt}")

                # Save error to state
                error_log = query_state.get("error_log", "")
                error_log += f"Attempt {attempt}: {error_msg}\n"
                query_state[f"attempt_{attempt}_result"] = "error"
                query_state[f"attempt_{attempt}_error"] = error_msg
                query_state["error_log"] = error_log
                state["queries"][str(query_id)] = query_state
                state_manager.save_state(state)

                save_attempt_outputs(query_dir, attempt, sparql_query, error_msg=error_msg)

        # If this was the last attempt, give up
        if attempt >= max_attempts:
            total_time = time.time() - query_start_time
            print(f"⚠️  Query {query_id} failed after {max_attempts} attempts")
            print(f"⏱️  Total time: {total_time:.2f}s")
            query_state["stage"] = "failed"
            query_state["success"] = False
            query_state["total_time"] = total_time
            query_state["timing_breakdown"] = timing_breakdown
            state["queries"][str(query_id)] = query_state
            state_manager.save_state(state)

            return {
                "success": False,
                "result": None,
                "query": query_text,
                "expected_label": expected_label,
                "error": query_state.get("error_log", "Max attempts reached"),
                "attempts": max_attempts,
                "total_time": total_time,
                "timing_breakdown": timing_breakdown
            }

        # Generate correction
        print(f"🔄 Generating correction for query {query_id}, attempt {attempt + 1}...")
        correction_start = time.time()
        try:
            relevant_iris = extracted.get("relevant_iris", [])
            error_prompt = build_error_correction_prompt(
                compress_error(error_msg),
                sparql_query,
                extracted,
                query_text,
                relevant_iris=relevant_iris
            )

            save_attempt_outputs(query_dir, attempt, sparql_query, error_correction_prompt=error_prompt)

            # Get corrected SPARQL
            sparql_query = get_sparql_from_llm(error_prompt, client, model_name, system_error)

            timing_breakdown["correction"] += time.time() - correction_start
            query_state["sparql_query"] = sparql_query
            query_state["current_attempt"] = attempt + 1
            state["queries"][str(query_id)] = query_state
            state_manager.save_state(state)
            print(f"✅ Generated corrected SPARQL for query {query_id} (took {time.time() - correction_start:.2f}s)")

        except Exception as e:
            total_time = time.time() - query_start_time
            print(f"❌ Correction failed for query {query_id}: {e}", file=sys.stderr)
            return {
                "success": False,
                "result": None,
                "query": query_text,
                "expected_label": expected_label,
                "error": f"Correction failed at attempt {attempt}: {e}",
                "attempts": attempt,
                "total_time": total_time,
                "timing_breakdown": timing_breakdown
            }


def process_all_queries(
    queries: List[Tuple[int, str, str]],
    client: OpenAI,
    model_name: str,
    system_initial: str,
    system_error: str,
    max_attempts: int,
    run_dir: str,
    state_manager: PipelineState,
    state: dict
) -> Dict[int, Dict]:
    """
    Process all queries through the pipeline using direct API calls.

    Returns:
        Dictionary mapping query_id to result info
    """
    results = {}

    for query_id, query_text, expected_label in queries:
        # Check if query is already complete
        query_state = state["queries"].get(str(query_id), {})
        if query_state.get("stage") == "complete" and query_state.get("success"):
            print(f"\n✅ Query {query_id} already completed successfully, skipping")
            # Reconstruct result from state with timing info
            results[query_id] = {
                "success": True,
                "result": query_state.get(f"attempt_{query_state.get('current_attempt', 1)}_result", ""),
                "query": query_text,
                "expected_label": expected_label,
                "sparql": query_state.get("sparql_query", ""),
                "attempts": query_state.get("current_attempt", 1),
                "total_time": query_state.get("total_time", 0.0),
                "timing_breakdown": query_state.get("timing_breakdown", {
                    "extraction": 0.0,
                    "initial_sparql": 0.0,
                    "validation": 0.0,
                    "correction": 0.0
                })
            }
            continue

        # Process the query
        result = process_single_query(
            query_id,
            query_text,
            expected_label,
            client,
            model_name,
            system_initial,
            system_error,
            max_attempts,
            run_dir,
            state_manager,
            state
        )

        results[query_id] = result

    successful = len([r for r in results.values() if r["success"]])
    failed = len([r for r in results.values() if not r["success"]])

    # Calculate timing statistics
    successful_times = [r.get("total_time", 0) for r in results.values() if r["success"] and r.get("total_time")]
    if successful_times:
        avg_time = sum(successful_times) / len(successful_times)
        min_time = min(successful_times)
        max_time = max(successful_times)

        print(f"\n{'='*80}")
        print(f"✅ PIPELINE COMPLETED: {successful} successful, {failed} failed")
        print(f"{'='*80}")
        print(f"\n⏱️  TIMING STATISTICS (Production Response Time per Query):")
        print(f"   Average: {avg_time:.2f}s")
        print(f"   Min: {min_time:.2f}s")
        print(f"   Max: {max_time:.2f}s")
        print(f"   Total: {sum(successful_times):.2f}s")
    else:
        print(f"\n{'='*80}")
        print(f"✅ PIPELINE COMPLETED: {successful} successful, {failed} failed")
        print(f"{'='*80}")

    return results


def sanitize_filename(name: str, fallback: str = "no_category") -> str:
    """Return a filesystem-safe short string for folder names."""
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "", name)
    name = name.strip("_-")
    return name[:100] or fallback


def save_results_to_csv(
    results: Dict[int, Dict],
    queries: List[Tuple[int, str, str]],
    output_path: str,
    state: dict
):
    """
    Save pipeline results to a CSV file similar to input format but with additional columns.

    Args:
        results: Dictionary mapping query_id to result info
        queries: Original list of (query_id, query_text, expected_label) tuples
        output_path: Path to save the CSV file
        state: Pipeline state dict with run_metadata
    """
    try:
        # Create mapping from query_id to original data
        queries_map = {qid: (qtext, expected) for qid, qtext, expected in queries}

        # Get model name from state
        model_name = state.get("model", "unknown")

        # Prepare CSV rows
        csv_rows = []
        for query_id in sorted(results.keys()):
            result_info = results[query_id]
            query_text, expected_label = queries_map.get(query_id, ("", ""))

            # Get state info for this query
            query_state = state["queries"].get(str(query_id), {})
            timing = result_info.get("timing_breakdown", {})

            row = {
                "Query_ID": query_id,
                "Question": query_text,
                "Category": expected_label,
                "Model": model_name,
                "Success": "Yes" if result_info["success"] else "No",
                "Attempts": result_info.get("attempts", 0),
                "Total_Time_Seconds": round(result_info.get("total_time", 0.0), 2),
                "Extraction_Time": round(timing.get("extraction", 0.0), 2),
                "SPARQL_Gen_Time": round(timing.get("initial_sparql", 0.0), 2),
                "Validation_Time": round(timing.get("validation", 0.0), 2),
                "Correction_Time": round(timing.get("correction", 0.0), 2),
                "SPARQL_Query": query_state.get("sparql_query", result_info.get("sparql", "")),
                "Result_Preview": (result_info.get("result", "")[:500] if result_info.get("result") else ""),
                "Error_Message": (result_info.get("error", "")[:500] if result_info.get("error") else ""),
                "Stage": query_state.get("stage", "unknown")
            }
            csv_rows.append(row)

        # Write to CSV
        if csv_rows:
            fieldnames = [
                "Query_ID", "Question", "Category", "Model", "Success", "Attempts",
                "Total_Time_Seconds", "Extraction_Time", "SPARQL_Gen_Time", "Validation_Time", "Correction_Time",
                "SPARQL_Query", "Result_Preview", "Error_Message", "Stage"
            ]

            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)

            print(f"📄 Results CSV saved to: {output_path}")
            return True
        else:
            print(f"⚠️  No results to save to CSV", file=sys.stderr)
            return False

    except Exception as e:
        print(f"❌ ERROR: Failed to save results CSV: {e}", file=sys.stderr)
        return False


def main():
    """
    Coordinate clause extraction, SPARQL generation, retries, and persistence using direct API calls.
    """
    check_graphdb_connection()

    parser = argparse.ArgumentParser(description="Run end-to-end SPARQL pipeline with direct OpenAI API calls and resume capability.")
    parser.add_argument("--query", type=str, help="Single natural-language user query.")
    parser.add_argument("--csv", type=str, help="Path to CSV file with queries (e.g., benchmark_questions_master.csv)")
    parser.add_argument("--max-attempts", type=int, default=10, help="Max correction attempts per query.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to process from CSV (e.g., --limit 4 for first 4 queries)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run if state exists.")
    parser.add_argument("--state-dir", type=str, default="pipeline_state_direct", help="Directory for state files.")
    parser.add_argument("--outputs-dir", type=str, default="outputs_direct", help="Directory for outputs.")

    # System prompt overrides
    parser.add_argument("--system-initial", type=str, help="Inline system prompt for initial generation.")
    parser.add_argument("--system-error", type=str, help="Inline system prompt for error correction.")
    parser.add_argument("--system-initial-file", type=str, help="Path to file with system prompt for initial generation.")
    parser.add_argument("--system-error-file", type=str, help="Path to file with system prompt for error correction.")

    args = parser.parse_args()

    # Validate input
    if not args.query and not args.csv:
        print("❌ ERROR: Must provide either --query or --csv", file=sys.stderr)
        sys.exit(1)

    # Resolve system prompts
    system_initial = (
        args.system_initial if args.system_initial is not None
        else (_read_text_file(args.system_initial_file) if args.system_initial_file else SYSTEM_PROMPT_INITIAL)
    )
    system_error = (
        args.system_error if args.system_error is not None
        else (_read_text_file(args.system_error_file) if args.system_error_file else SYSTEM_PROMPT_ERROR)
    )

    model_name = os.getenv("LLM_MODEL", "gpt-5")
    max_attempts = max(1, args.max_attempts)

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment.")

    client = OpenAI(api_key=api_key)

    # Initialize state manager
    state_manager = PipelineState(args.state_dir)

    # Load or initialize state
    if args.resume and state_manager.state_exists():
        print("\n🔄 RESUMING FROM PREVIOUS STATE")
        state = state_manager.load_state()
    else:
        print("\n🆕 STARTING NEW PIPELINE RUN")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        state = {
            "run_id": timestamp,
            "model": model_name,
            "max_attempts": max_attempts,
            "queries": {},
            "started_at": datetime.datetime.now().isoformat()
        }
        state_manager.save_state(state)

    # Ensure queries dict exists
    if "queries" not in state:
        state["queries"] = {}

    # Load queries
    if args.csv:
        queries = load_queries_from_csv(args.csv)
    else:
        queries = [(0, args.query, "")]

    # Apply limit if specified
    if args.limit is not None and len(queries) > args.limit:
        queries = queries[:args.limit]
        print(f"🔍 Limiting to first {args.limit} queries")

    # Prepare outputs directory with model name
    outputs_dir = args.outputs_dir
    os.makedirs(outputs_dir, exist_ok=True)
    run_dir = os.path.join(outputs_dir, f"run_{state['run_id']}_{model_name}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n📊 Processing {len(queries)} queries")
    print(f"🤖 Model: {model_name}")
    print(f"📁 Output directory: {run_dir}")
    print(f"💾 State directory: {args.state_dir}")

    # Start timing AFTER all one-time setup is complete (GraphDB connection, state loading, etc.)
    # This represents actual production processing time without setup overhead
    pipeline_start_time = time.time()

    # Process all queries
    results = process_all_queries(
        queries,
        client,
        model_name,
        system_initial,
        system_error,
        max_attempts,
        run_dir,
        state_manager,
        state
    )

    # Calculate total pipeline time (excluding one-time setup)
    total_pipeline_time = time.time() - pipeline_start_time

    # Build results summary
    results_summary = {
        "total": len(queries),
        "successful": sum(1 for r in results.values() if r["success"]),
        "failed": sum(1 for r in results.values() if not r["success"]),
        "total_pipeline_time_seconds": round(total_pipeline_time, 2),
        "results": []
    }

    for query_id, result_info in sorted(results.items()):
        results_summary["results"].append({
            "query_id": query_id,
            "query": result_info["query"],
            "expected_label": result_info.get("expected_label", ""),
            "success": result_info["success"],
            "result": result_info.get("result"),
            "attempts": result_info.get("attempts", 0),
            "error": result_info.get("error"),
            "total_time": result_info.get("total_time", 0.0)
        })

    # Save final summary
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    # Save results to CSV
    csv_path = os.path.join(run_dir, "results_summary.csv")
    save_results_to_csv(results, queries, csv_path, state)

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"✅ Successful: {results_summary['successful']}/{results_summary['total']}")
    print(f"❌ Failed: {results_summary['failed']}/{results_summary['total']}")
    print(f"⏱️  Total Pipeline Time (excluding one-time setup): {total_pipeline_time:.2f}s ({total_pipeline_time/60:.2f} minutes)")
    print(f"📊 Summary saved to: {summary_path}")
    print(f"📄 CSV results saved to: {csv_path}")

if __name__ == "__main__":
    main()
