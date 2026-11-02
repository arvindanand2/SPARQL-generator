"""End-to-end SPARQL pipeline orchestrating extraction, prompting, and retries."""

import subprocess
import json
import sys
import time
import os
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from compress_chunk import compact_chunk
from enhanced_csv_embedder import EnhancedCSVEmbedder
import datetime
import re

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

def run_extract_chunk(user_query: str):
    """Invoke clause extraction and augment the result with relevant IRIs."""
    try:
        result = subprocess.run(
            [sys.executable, "extract_chunk.py", user_query],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            try:
                extracted = json.loads(result.stdout)
                compressed = compact_chunk(extracted)
                print("🔍 Extracting relevant IRIs from query...")
                iris_info = extract_iris_from_query(user_query, top_k=10)
                if iris_info:
                    compressed["relevant_iris"] = iris_info
                    print(f"✅ Found {len(iris_info)} relevant IRIs")
                    for iri_data in iris_info[:5]:
                        print(f"   • {iri_data['label']} - {iri_data['iri']}")
                else:
                    print("⚠️  No relevant IRIs found")
                return compressed
            except Exception as e:
                raise RuntimeError(f"JSON parse error: {e}\nOutput: {result.stdout}")
        else:
            raise RuntimeError(f"extract_chunk.py error: {result.stderr}")
    except Exception as e:
        raise RuntimeError(str(e))

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
        with open(reference_turtle_path, "r", encoding="utf-8") as f:
            reference_turtle_data = f.read()
        chunk_str = json.dumps(extracted_chunk, separators=(",", ":"))
        iris_str = json.dumps(relevant_iris, separators=(",", ":")) if relevant_iris is not None else "[]"
        txt_body = template.replace("{USER_QUERY}", user_query)
        txt_body = txt_body.replace("{EXTRACTED_Clauses}", chunk_str)
        txt_body = txt_body.replace("{EXTRACTED_IRI}", iris_str)
        txt_body = txt_body.replace("{REFERENCE_TURTLE_DATA}", reference_turtle_data)
        return template.replace("{TXT}", txt_body) if "{TXT}" in template else txt_body
    except Exception as e:
        print(f"Failed to build LLM prompt: {e}", file=sys.stderr)
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
        with open(reference_turtle_path, "r", encoding="utf-8") as f:
            reference_turtle_data = f.read()
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
        print(f"Failed to build error correction prompt: {e}", file=sys.stderr)
        return f"Error: {error_message}\nPrevious SPARQL: {previous_sparql}"

def get_sparql_from_llm(prompt: str, model: str, system_prompt: str) -> str:
    """Call OpenAI chat completions to obtain SPARQL text."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment.")
    client = OpenAI(api_key=api_key)
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "text"},
    )
    content = resp.choices[0].message.content or ""
    return content.strip()

def run_query_graph_with_retries(sparql_query: str, max_retries: int = 5):
    """Execute the SPARQL query script with simple retry logic."""
    error_msgs = []
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                [sys.executable, "query_Graph.py", sparql_query],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            if result.returncode == 0 and not result.stderr:
                print("GraphDB result:\n", result.stdout)
                return result.stdout, error_msgs
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                error_msgs.append(error_msg)
                print(f"Attempt {attempt+1} failed: {error_msg}", file=sys.stderr)
                time.sleep(1)
        except subprocess.TimeoutExpired:
            # Provide helpful error message for SPARQL generator
            timeout_error = (
                "Query execution timeout (60 seconds exceeded). "
                "The SPARQL query is likely too complex or inefficient. "
                "Consider:\n"
                "1. Adding more specific filters to reduce result set size\n"
                "2. Avoiding cartesian products (multiple OPTIONAL clauses without shared variables)\n"
                "3. Simplifying complex property paths or nested queries"
            )
            error_msgs.append(timeout_error)
            print(f"Attempt {attempt+1} timed out after 60 seconds", file=sys.stderr)
            time.sleep(1)
    print(f"All {max_retries} attempts failed.", file=sys.stderr)
    return None, error_msgs

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

def save_run_outputs(run_data: dict, outputs_dir: str = "outputs", model_name: str = None, run_dir: str = None):
    """Persist run artifacts (query, prompts, results) into structured files."""
    _ensure_dir(outputs_dir)
    if run_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = os.path.join(outputs_dir, f"run_{timestamp}")
    _ensure_dir(run_dir)

    def _write(path, content):
        """Write content to a path while catching IO errors."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"Failed to write {path}: {e}", file=sys.stderr)

    _write(os.path.join(run_dir, "user_query.txt"), run_data.get("user_query", ""))
    _write(os.path.join(run_dir, "llm_prompt.txt"), run_data.get("llm_prompt", ""))
    _write(os.path.join(run_dir, "sparql_query.txt"), run_data.get("sparql_query", ""))
    _write(os.path.join(run_dir, "graphdb_result.txt"), run_data.get("graphdb_result", ""))
    try:
        with open(os.path.join(run_dir, "extracted.json"), "w", encoding="utf-8") as f:
            json.dump(run_data.get("extracted", {}), f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to write extracted.json: {e}", file=sys.stderr)
    if run_data.get("error_log"):
        _write(os.path.join(run_dir, "error_log.txt"), run_data["error_log"])
    if model_name:
        _write(os.path.join(run_dir, "model.txt"), model_name)
    print(f"Run outputs saved to: {run_dir}")
    return run_dir

def save_partial_run_outputs(run_data: dict, run_dir: str, model_name: str = None):
    """Persist early-stage artifacts before the pipeline completes."""
    _ensure_dir(run_dir)
    with open(os.path.join(run_dir, "user_query.txt"), "w", encoding="utf-8") as f:
        f.write(run_data.get("user_query", ""))
    with open(os.path.join(run_dir, "extracted.json"), "w", encoding="utf-8") as f:
        json.dump(run_data.get("extracted", {}), f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "llm_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(run_data.get("llm_prompt", ""))
    if model_name:
        with open(os.path.join(run_dir, "model.txt"), "w", encoding="utf-8") as f:
            f.write(model_name)

def save_attempt_outputs(run_dir: str, attempt_idx: int, sparql_query: str, result: str = None,
                         error_msg: str = None, error_correction_prompt: str = None):
    """Persist per-attempt SPARQL runs, including errors and correction prompts."""
    attempt_dir = os.path.join(run_dir, f"attempt_{attempt_idx}")
    _ensure_dir(attempt_dir)
    def _write(path, content):
        """Write helper used for attempt artifacts."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(content or "")
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

def check_graphdb_connection():
    """Check connection to GraphDB by invoking query_Graph.py with --test-connection."""
    import subprocess
    result = subprocess.run([sys.executable, "query_Graph.py", "--test-connection"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: Could not connect to GraphDB.\n" + result.stdout + result.stderr, file=sys.stderr)
        sys.exit(1)
    else:
        print(result.stdout.strip())

def main():
    check_graphdb_connection()  # Ensure GraphDB is available before proceeding

    """Coordinate clause extraction, SPARQL generation, retries, and persistence."""
    parser = argparse.ArgumentParser(description="Run end-to-end SPARQL pipeline.")
    parser.add_argument("user_query", type=str, help="Natural-language user query.")
    parser.add_argument("--max-attempts", type=int, default=5, help="Max correction attempts.")
    # System prompt overrides
    parser.add_argument("--system-initial", type=str, help="Inline system prompt for initial generation.")
    parser.add_argument("--system-error", type=str, help="Inline system prompt for error correction.")
    parser.add_argument("--system-initial-file", type=str, help="Path to file with system prompt for initial generation.")
    parser.add_argument("--system-error-file", type=str, help="Path to file with system prompt for error correction.")
    args = parser.parse_args()

    # Resolve system prompts (CLI > file > defaults)
    system_initial = (
        args.system_initial if args.system_initial is not None
        else (_read_text_file(args.system_initial_file) if args.system_initial_file else SYSTEM_PROMPT_INITIAL)
    )
    system_error = (
        args.system_error if args.system_error is not None
        else (_read_text_file(args.system_error_file) if args.system_error_file else SYSTEM_PROMPT_ERROR)
    )

    user_query = args.user_query
    model_name = os.getenv("LLM_MODEL", "gpt-5")
    max_attempts = max(1, args.max_attempts)

    run_data = {"user_query": user_query}

    # Prepare run dir
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(outputs_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 1) Extract clauses (+ IRIs)
    extracted = run_extract_chunk(user_query)
    run_data["extracted"] = extracted
    print("Extracted clauses (compressed):\n", json.dumps(extracted, indent=2, ensure_ascii=False))

    relevant_iris = extracted.get("relevant_iris", [])

    # 2) Build initial prompt
    llm_prompt = build_llm_prompt(extracted, user_query, relevant_iris=relevant_iris,
                                  template_path="sparql_prompt_template_new.txt")
    run_data["llm_prompt"] = llm_prompt
    save_partial_run_outputs(run_data, run_dir)
    print("Prompt for LLM:\n", llm_prompt)

    # 3) Get initial SPARQL (uses INITIAL system prompt)
    sparql_query = get_sparql_from_llm(llm_prompt, model=model_name, system_prompt=system_initial)
    run_data["sparql_query"] = sparql_query
    print("LLM SPARQL response:\n", sparql_query)

    attempt = 1
    error_log = ""
    last_query_errors = []

    # Save attempt 1 SPARQL before execution
    save_attempt_outputs(run_dir, attempt_idx=attempt, sparql_query=sparql_query)

    while attempt <= max_attempts:
        result, query_errors = run_query_graph_with_retries(sparql_query, max_retries=1)
        last_query_errors = query_errors

        if result is not None:
            print(f"GraphDB result (attempt {attempt}):\n", result)
            run_data["graphdb_result"] = result
            save_attempt_outputs(run_dir, attempt_idx=attempt, sparql_query=sparql_query, result=result)
            save_query_log(user_query, sparql_query)
            break

        # Failure path
        error_msg = (query_errors[-1] if query_errors else "Unknown error occurred")
        print(f"GraphDB error (attempt {attempt}):\n", error_msg, file=sys.stderr)
        error_log += f"Attempt {attempt}: {error_msg}\n"
        save_attempt_outputs(run_dir, attempt_idx=attempt, sparql_query=sparql_query, error_msg=error_msg)

        if attempt == max_attempts:
            # Exhausted attempts
            break

        # 4) Build error-correction prompt
        error_prompt = build_error_correction_prompt(
            compress_error(error_msg), sparql_query, extracted, user_query,
            relevant_iris=relevant_iris, template_path="sparql_error_correction_prompt.txt"
        )
        save_attempt_outputs(run_dir, attempt_idx=attempt, sparql_query=sparql_query,
                             error_correction_prompt=error_prompt)

        # 5) Get corrected SPARQL (uses ERROR system prompt)
        sparql_query = get_sparql_from_llm(error_prompt, model=model_name, system_prompt=system_error)
        run_data["sparql_query"] = sparql_query
        print(f"LLM SPARQL retry response (attempt {attempt+1}):\n", sparql_query)

        # Pre-save next attempt SPARQL
        save_attempt_outputs(run_dir, attempt_idx=attempt+1, sparql_query=sparql_query)

        attempt += 1

    if run_data.get("graphdb_result") is None:
        final_error = (last_query_errors[-1] if last_query_errors else "Unknown error")
        print(f"All {max_attempts} attempts failed. Last error: {final_error}", file=sys.stderr)
        run_data["error_log"] = error_log

    save_run_outputs(run_data, outputs_dir=outputs_dir, run_dir=run_dir)

if __name__ == "__main__":
    # Local helpers used above
    main()
