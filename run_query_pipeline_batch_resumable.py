"""
End-to-end SPARQL pipeline using OpenAI Batch API for all LLM calls.
Enhanced with resume functionality and multi-query batch processing.

This module orchestrates:
1. Clause extraction via batch API
2. SPARQL generation via batch API
3. Error correction via batch API (with retries)
4. Query execution against GraphDB

Features:
- Automatic state persistence and resume
- Process multiple queries from CSV file
- Resume from any interrupted stage (extraction, generation, correction)
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


class BatchAPIHandler:
    """
    Handles OpenAI Batch API operations with retry logic and comprehensive logging.
    """

    def __init__(self, client: OpenAI, log_dir: str = "batch_logs"):
        """
        Initialize the batch API handler.

        Args:
            client: OpenAI client instance
            log_dir: Directory to store batch operation logs
        """
        self.client = client
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def create_batch_request(
        self,
        custom_id: str,
        messages: List[Dict],
        model: str,
        response_format: Optional[Dict] = None
    ) -> Dict:
        """
        Create a single batch request object.

        Args:
            custom_id: Unique identifier for this request
            messages: List of message objects (system, user, assistant)
            model: Model name to use
            response_format: Optional response format specification

        Returns:
            Batch request dictionary
        """
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
            }
        }
        if response_format:
            request["body"]["response_format"] = response_format
        return request

    def submit_batch(
        self,
        requests: List[Dict],
        description: str = "Batch request"
    ) -> str:
        """
        Submit a batch of requests to OpenAI Batch API.

        Args:
            requests: List of batch request objects
            description: Description for the batch job

        Returns:
            Batch ID

        Raises:
            RuntimeError: If submission fails after retries
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        input_file_path = os.path.join(self.log_dir, f"batch_input_{timestamp}.jsonl")

        # Write requests to JSONL file
        try:
            with open(input_file_path, "w", encoding="utf-8") as f:
                for request in requests:
                    f.write(json.dumps(request) + "\n")
            print(f"📝 Created batch input file: {input_file_path}")
        except (OSError, PermissionError) as e:
            raise RuntimeError(f"Failed to write batch input file: {e}")

        print(f"📦 Submitting batch with {len(requests)} requests...")

        # Upload the file with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(input_file_path, "rb") as f:
                    batch_input_file = self.client.files.create(
                        file=f,
                        purpose="batch"
                    )
                break
            except FileNotFoundError as e:
                raise RuntimeError(f"Batch input file not found: {e}")
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  File upload attempt {attempt + 1} failed: {e}. Retrying...", file=sys.stderr)
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise RuntimeError(f"Failed to upload batch file after {max_retries} attempts: {e}")

        # Create the batch with retry logic
        for attempt in range(max_retries):
            try:
                batch = self.client.batches.create(
                    input_file_id=batch_input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": description,
                        "timestamp": timestamp
                    }
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  Batch creation attempt {attempt + 1} failed: {e}. Retrying...", file=sys.stderr)
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Failed to create batch after {max_retries} attempts: {e}")

        # Log batch submission
        try:
            log_path = os.path.join(self.log_dir, f"batch_{batch.id}_submission.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump({
                    "batch_id": batch.id,
                    "input_file_id": batch_input_file.id,
                    "input_file_path": input_file_path,
                    "description": description,
                    "timestamp": timestamp,
                    "num_requests": len(requests)
                }, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Failed to write batch submission log: {e}", file=sys.stderr)

        print(f"✅ Batch submitted: {batch.id}")
        return batch.id

    def wait_for_batch(
        self,
        batch_id: str,
        poll_interval: int = 5,
        max_wait_time: int = 86400  # 24 hours for OpenAI batch API
    ) -> Dict:
        """
        Wait for a batch to complete with status polling.

        Args:
            batch_id: Batch ID to monitor
            poll_interval: Seconds between status checks
            max_wait_time: Maximum seconds to wait (default: 24 hours)

        Returns:
            Batch status object

        Raises:
            TimeoutError: If batch doesn't complete within max_wait_time
            RuntimeError: If batch fails
        """
        start_time = time.time()
        print(f"⏳ Waiting for batch {batch_id} to complete...")

        while True:
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                raise TimeoutError(f"Batch {batch_id} exceeded max wait time of {max_wait_time}s")

            batch = self.client.batches.retrieve(batch_id)
            status = batch.status

            # Log status
            print(f"   Status: {status} | Request counts: {batch.request_counts}")

            if status == "completed":
                print(f"✅ Batch {batch_id} completed successfully")
                return batch
            elif status == "failed" or status == "expired" or status == "cancelled":
                error_msg = f"Batch {batch_id} ended with status: {status}"
                if hasattr(batch, 'errors') and batch.errors:
                    error_msg += f"\nErrors: {batch.errors}"
                raise RuntimeError(error_msg)

            # Still processing
            time.sleep(poll_interval)

    def retrieve_batch_results(
        self,
        batch_id: str
    ) -> Dict[str, Dict]:
        """
        Retrieve and parse batch results.

        Args:
            batch_id: Batch ID to retrieve results for

        Returns:
            Dictionary mapping custom_id to result data
        """
        batch = self.client.batches.retrieve(batch_id)

        if not batch.output_file_id:
            raise RuntimeError(f"Batch {batch_id} has no output file")

        # Download results
        file_response = self.client.files.content(batch.output_file_id)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_file_path = os.path.join(self.log_dir, f"batch_output_{batch_id}_{timestamp}.jsonl")

        # Save to file
        with open(output_file_path, "wb") as f:
            f.write(file_response.content)

        print(f"💾 Saved batch results to: {output_file_path}")

        # Parse results
        results = {}
        with open(output_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    custom_id = result.get("custom_id")
                    results[custom_id] = result

        # Log errors if any
        if batch.error_file_id:
            error_response = self.client.files.content(batch.error_file_id)
            error_file_path = os.path.join(self.log_dir, f"batch_errors_{batch_id}_{timestamp}.jsonl")
            with open(error_file_path, "wb") as f:
                f.write(error_response.content)
            print(f"⚠️  Batch had errors, saved to: {error_file_path}")

        print(f"📊 Retrieved {len(results)} results from batch {batch_id}")
        return results

    def extract_response_content(self, result: Dict) -> Optional[str]:
        """
        Extract the content from a batch result.

        Args:
            result: Batch result dictionary

        Returns:
            Response content string or None if error
        """
        if result.get("response", {}).get("status_code") != 200:
            error = result.get("error") or result.get("response", {}).get("body")
            print(f"❌ Request {result.get('custom_id')} failed: {error}", file=sys.stderr)
            return None

        body = result.get("response", {}).get("body", {})
        choices = body.get("choices", [])
        if not choices:
            return None

        content = choices[0].get("message", {}).get("content")
        return content


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


def extract_chunk_batch(
    user_queries: List[Tuple[int, str]],  # List of (query_id, query_text)
    batch_handler: BatchAPIHandler,
    model: str,
    state_manager: PipelineState = None,
    state: dict = None
) -> Dict[int, Dict]:
    """
    Extract clause chunks for multiple queries using batch API.

    Args:
        user_queries: List of (query_id, query_text) tuples
        batch_handler: BatchAPIHandler instance
        model: Model name to use
        state_manager: Optional state manager for saving batch ID
        state: Optional state dict for saving batch ID

    Returns:
        Dictionary mapping query_id to extracted chunks
    """
    # Check if there's a pending batch from a previous run
    batch_id = None
    if state_manager and state:
        # Check if all queries have the same extraction_batch_id
        batch_ids = set()
        for query_id, _ in user_queries:
            query_state = state["queries"].get(str(query_id), {})
            if "extraction_batch_id" in query_state:
                batch_ids.add(query_state["extraction_batch_id"])

        if len(batch_ids) == 1:
            # All queries share the same batch ID - try to resume
            batch_id = batch_ids.pop()
            print(f"🔄 Found pending extraction batch from previous run: {batch_id}")
            try:
                # Check batch status
                batch = batch_handler.client.batches.retrieve(batch_id)
                if batch.status == "completed":
                    print(f"✅ Extraction batch already completed, retrieving results...")
                    results = batch_handler.retrieve_batch_results(batch_id)

                    # Parse results
                    extracted_chunks = {}
                    for query_id, query in user_queries:
                        custom_id = f"extract_{query_id}"
                        if custom_id in results:
                            content = batch_handler.extract_response_content(results[custom_id])
                            if content:
                                try:
                                    extracted = json.loads(content)
                                    compressed = compact_chunk(extracted)

                                    # Add IRIs
                                    print(f"🔍 Extracting relevant IRIs for query {query_id}...")
                                    iris_info = extract_iris_from_query(query, top_k=10)
                                    if iris_info:
                                        compressed["relevant_iris"] = iris_info
                                        print(f"✅ Found {len(iris_info)} relevant IRIs")

                                    extracted_chunks[query_id] = compressed
                                except Exception as e:
                                    print(f"❌ Failed to parse extraction for query {query_id}: {e}", file=sys.stderr)

                    if extracted_chunks:
                        print(f"✅ Successfully resumed extraction batch with {len(extracted_chunks)} results")
                        return extracted_chunks
                elif batch.status in ["validating", "in_progress", "finalizing"]:
                    print(f"⏳ Extraction batch still in progress (status: {batch.status}), waiting...")
                    # Continue to wait for the batch below
                elif batch.status in ["failed", "expired", "cancelled"]:
                    print(f"⚠️  Previous extraction batch {batch.status}, creating new batch...")
                    batch_id = None
                else:
                    print(f"⚠️  Unknown batch status: {batch.status}, creating new batch...")
                    batch_id = None
            except Exception as e:
                print(f"⚠️  Could not resume extraction batch {batch_id}: {e}", file=sys.stderr)
                print(f"   Creating new batch instead...")
                batch_id = None

    # Load schema and prompt with error handling (only if we need to create a new batch)
    if batch_id is None:
        try:
            with open("clause_chunk_schema.json", encoding="utf-8") as f:
                clause_chunk_schema = json.load(f)
        except FileNotFoundError:
            print("❌ ERROR: Required file 'clause_chunk_schema.json' not found", file=sys.stderr)
            raise RuntimeError("Missing required file: clause_chunk_schema.json")
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Invalid JSON in clause_chunk_schema.json: {e}", file=sys.stderr)
            raise RuntimeError(f"Corrupted schema file: {e}")
        except PermissionError as e:
            print(f"❌ ERROR: Permission denied reading clause_chunk_schema.json: {e}", file=sys.stderr)
            raise RuntimeError(f"Cannot read schema file: {e}")
        except Exception as e:
            print(f"❌ ERROR: Failed to load clause_chunk_schema.json: {e}", file=sys.stderr)
            raise RuntimeError(f"Cannot load schema file: {e}")

        try:
            with open("clause_chunker_prompt_new.txt", "r", encoding="utf-8") as f:
                system_prompt = f.read()
        except FileNotFoundError:
            print("❌ ERROR: Required file 'clause_chunker_prompt_new.txt' not found", file=sys.stderr)
            raise RuntimeError("Missing required file: clause_chunker_prompt_new.txt")
        except PermissionError as e:
            print(f"❌ ERROR: Permission denied reading clause_chunker_prompt_new.txt: {e}", file=sys.stderr)
            raise RuntimeError(f"Cannot read prompt file: {e}")
        except Exception as e:
            print(f"❌ ERROR: Failed to load clause_chunker_prompt_new.txt: {e}", file=sys.stderr)
            raise RuntimeError(f"Cannot load prompt file: {e}")

        # Create batch requests
        requests = []
        for query_id, query in user_queries:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract clauses from this query into the schema. Query:\n{query}"}
            ]

            request = batch_handler.create_batch_request(
                custom_id=f"extract_{query_id}",
                messages=messages,
                model=model,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ExtractedClauses",
                        "strict": True,
                        "schema": clause_chunk_schema
                    }
                }
            )
            requests.append(request)

        # Submit and wait
        batch_id = batch_handler.submit_batch(requests, description="Clause extraction")

        # Save batch ID to state immediately after submission
        if state_manager and state:
            for query_id, _ in user_queries:
                query_state = state["queries"].get(str(query_id), {})
                query_state["extraction_batch_id"] = batch_id
                state["queries"][str(query_id)] = query_state
            state_manager.save_state(state)
            print(f"💾 Saved extraction batch ID to state: {batch_id}")

    # Wait for batch completion
    batch = batch_handler.wait_for_batch(batch_id)
    results = batch_handler.retrieve_batch_results(batch_id)

    # Parse results with comprehensive error handling
    extracted_chunks = {}
    for query_id, query in user_queries:
        custom_id = f"extract_{query_id}"
        if custom_id in results:
            content = batch_handler.extract_response_content(results[custom_id])
            if content:
                try:
                    extracted = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse JSON for query {query_id}: {e}", file=sys.stderr)
                    print(f"   Content preview: {content[:200]}...", file=sys.stderr)
                    continue

                try:
                    compressed = compact_chunk(extracted)
                except Exception as e:
                    print(f"⚠️  Warning: Failed to compress chunk for query {query_id}: {e}", file=sys.stderr)
                    compressed = extracted  # Use uncompressed if compression fails

                # Add IRIs
                try:
                    print(f"🔍 Extracting relevant IRIs for query {query_id}...")
                    iris_info = extract_iris_from_query(query, top_k=10)
                    if iris_info:
                        compressed["relevant_iris"] = iris_info
                        print(f"✅ Found {len(iris_info)} relevant IRIs")
                except Exception as e:
                    print(f"⚠️  Warning: Failed to extract IRIs for query {query_id}: {e}", file=sys.stderr)
                    compressed["relevant_iris"] = []

                extracted_chunks[query_id] = compressed
        else:
            print(f"❌ No result for extraction {custom_id}", file=sys.stderr)

    return extracted_chunks


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
    except PermissionError as e:
        print(f"❌ ERROR: Permission denied reading template: {e}", file=sys.stderr)
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


def run_query_graph_with_retries(sparql_query: str, max_retries: int = 5):
    """Execute the SPARQL query script with simple retry logic."""
    sleep(2)  # Initial wait before starting to avoid immediate overload
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
                return result.stdout, error_msgs
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                error_msgs.append(error_msg)
                print(f"Attempt {attempt+1} failed: {error_msg}", file=sys.stderr)
                time.sleep(1)
        except subprocess.TimeoutExpired:
            # Provide helpful error message for SPARQL generator
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
            time.sleep(1)
        except FileNotFoundError as e:
            error_msg = f"ERROR: query_Graph.py not found or Python executable issue: {e}"
            print(f"❌ {error_msg}", file=sys.stderr)
            error_msgs.append(error_msg)
            break  # Don't retry if file doesn't exist
        except Exception as e:
            error_msg = f"ERROR: Subprocess execution failed: {e}"
            print(f"❌ {error_msg}", file=sys.stderr)
            error_msgs.append(error_msg)
            if attempt < max_retries - 1:
                time.sleep(1)
    print(f"All {max_retries} attempts failed.", file=sys.stderr)
    return None, error_msgs


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
        encodings = ['utf-8']
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


def generate_sparql_batch(
    prompts: List[Tuple[str, str]],  # List of (custom_id, prompt) tuples
    batch_handler: BatchAPIHandler,
    model: str,
    system_prompt: str,
    state_manager: PipelineState = None,
    state: dict = None,
    batch_type: str = "SPARQL generation"
) -> Dict[str, str]:
    """
    Generate SPARQL queries for multiple prompts using batch API.

    Args:
        prompts: List of (custom_id, prompt_text) tuples
        batch_handler: BatchAPIHandler instance
        model: Model name to use
        system_prompt: System prompt for SPARQL generation
        state_manager: Optional state manager for saving batch ID
        state: Optional state dict for saving batch ID
        batch_type: Type of batch for identification

    Returns:
        Dictionary mapping custom_id to SPARQL query string
    """
    # Check if there's a pending batch from a previous run
    batch_id = None
    if state_manager and state:
        # Check if all prompts have the same batch ID
        batch_ids = set()
        for custom_id, _ in prompts:
            # Parse query_id from custom_id
            if custom_id.startswith("initial_sparql_"):
                query_id = custom_id.replace("initial_sparql_", "")
                query_state = state["queries"].get(str(query_id), {})
                if "initial_sparql_batch_id" in query_state:
                    batch_ids.add(query_state["initial_sparql_batch_id"])
            elif custom_id.startswith("correction_"):
                # Format: correction_{query_id}_{attempt}
                parts = custom_id.replace("correction_", "").split("_")
                if len(parts) >= 2:
                    query_id = parts[0]
                    attempt = parts[1]
                    query_state = state["queries"].get(str(query_id), {})
                    if f"correction_{attempt}_batch_id" in query_state:
                        batch_ids.add(query_state[f"correction_{attempt}_batch_id"])

        if len(batch_ids) == 1:
            # All prompts share the same batch ID - try to resume
            batch_id = batch_ids.pop()
            print(f"🔄 Found pending {batch_type} batch from previous run: {batch_id}")
            try:
                # Check batch status
                batch = batch_handler.client.batches.retrieve(batch_id)
                if batch.status == "completed":
                    print(f"✅ {batch_type} batch already completed, retrieving results...")
                    results = batch_handler.retrieve_batch_results(batch_id)

                    # Parse results
                    sparql_queries = {}
                    for custom_id, _ in prompts:
                        if custom_id in results:
                            content = batch_handler.extract_response_content(results[custom_id])
                            if content:
                                sparql_queries[custom_id] = content.strip()
                        else:
                            print(f"❌ No result for SPARQL generation {custom_id}", file=sys.stderr)

                    if sparql_queries:
                        print(f"✅ Successfully resumed {batch_type} batch with {len(sparql_queries)} results")
                        return sparql_queries
                elif batch.status in ["validating", "in_progress", "finalizing"]:
                    print(f"⏳ {batch_type} batch still in progress (status: {batch.status}), waiting...")
                    # Continue to wait for the batch below
                elif batch.status in ["failed", "expired", "cancelled"]:
                    print(f"⚠️  Previous {batch_type} batch {batch.status}, creating new batch...")
                    batch_id = None
                else:
                    print(f"⚠️  Unknown batch status: {batch.status}, creating new batch...")
                    batch_id = None
            except Exception as e:
                print(f"⚠️  Could not resume {batch_type} batch {batch_id}: {e}", file=sys.stderr)
                print(f"   Creating new batch instead...")
                batch_id = None

    # Create batch requests (only if we need to create a new batch)
    if batch_id is None:
        requests = []
        for custom_id, prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            request = batch_handler.create_batch_request(
                custom_id=custom_id,
                messages=messages,
                model=model,
                response_format={"type": "text"}  # Match original run_query_pipeline.py
            )
            requests.append(request)

        # Submit and wait
        batch_id = batch_handler.submit_batch(requests, description=batch_type)

        # Save batch ID to state immediately after submission
        if state_manager and state:
            for custom_id, _ in prompts:
                # Parse query_id from custom_id
                # Format: "initial_sparql_{query_id}" or "correction_{query_id}_{attempt}"
                if custom_id.startswith("initial_sparql_"):
                    query_id = custom_id.replace("initial_sparql_", "")
                    query_state = state["queries"].get(str(query_id), {})
                    query_state["initial_sparql_batch_id"] = batch_id
                    state["queries"][str(query_id)] = query_state
                elif custom_id.startswith("correction_"):
                    # Format: correction_{query_id}_{attempt}
                    parts = custom_id.replace("correction_", "").split("_")
                    if len(parts) >= 2:
                        query_id = parts[0]
                        attempt = parts[1]
                        query_state = state["queries"].get(str(query_id), {})
                        query_state[f"correction_{attempt}_batch_id"] = batch_id
                        state["queries"][str(query_id)] = query_state
            state_manager.save_state(state)
            print(f"💾 Saved {batch_type} batch ID to state: {batch_id}")

    # Wait for batch completion
    batch = batch_handler.wait_for_batch(batch_id)
    results = batch_handler.retrieve_batch_results(batch_id)

    # Parse results
    sparql_queries = {}
    for custom_id, _ in prompts:
        if custom_id in results:
            content = batch_handler.extract_response_content(results[custom_id])
            if content:
                sparql_queries[custom_id] = content.strip()
        else:
            print(f"❌ No result for SPARQL generation {custom_id}", file=sys.stderr)

    return sparql_queries


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


def process_all_queries_batch(
    queries: List[Tuple[int, str, str]],
    batch_handler: BatchAPIHandler,
    model_name: str,
    system_initial: str,
    system_error: str,
    max_attempts: int,
    run_dir: str,
    state_manager: PipelineState,
    state: dict
) -> Dict[int, Dict]:
    """
    Process all queries using full batch approach:
    1. Batch extract all queries
    2. Batch generate all initial SPARQL
    3. Validate each sequentially
    4. Batch correct all failures
    5. Repeat until no failures or max attempts

    Returns:
        Dictionary mapping query_id to result info
    """
    results = {}

    # Track which queries need processing at each stage
    queries_dict = {qid: (qtext, expected) for qid, qtext, expected in queries}

    # ========== STAGE 1: BATCH EXTRACT ALL CLAUSES ==========
    print("\n" + "="*80)
    print("STAGE 1: EXTRACTING CLAUSES FOR ALL QUERIES")
    print("="*80)

    queries_needing_extraction = []
    extracted_data = {}

    for query_id, query_text, _ in queries:
        query_state = state["queries"].get(str(query_id), {})
        if query_state.get("stage") in ["extraction_complete", "initial_sparql_complete", "complete"]:
            print(f"✅ Query {query_id}: Using cached extraction")
            extracted_data[query_id] = query_state["extracted"]
        else:
            queries_needing_extraction.append((query_id, query_text))

    if queries_needing_extraction:
        print(f"\n🔄 Extracting clauses for {len(queries_needing_extraction)} queries in single batch...")
        batch_extracted = extract_chunk_batch(
            queries_needing_extraction,
            batch_handler,
            model_name,
            state_manager,
            state
        )

        # Save extraction results
        for query_id in batch_extracted:
            extracted_data[query_id] = batch_extracted[query_id]
            query_state = state["queries"].get(str(query_id), {})
            query_state["extracted"] = batch_extracted[query_id]
            query_state["stage"] = "extraction_complete"
            if "extraction_batch_id" in query_state:
                del query_state["extraction_batch_id"]
            state["queries"][str(query_id)] = query_state
        state_manager.save_state(state)
        print(f"✅ Extracted clauses for {len(batch_extracted)} queries")

    # ========== STAGE 2: BATCH GENERATE ALL INITIAL SPARQL ==========
    print("\n" + "="*80)
    print("STAGE 2: GENERATING INITIAL SPARQL FOR ALL QUERIES")
    print("="*80)

    queries_needing_sparql = []
    sparql_queries = {}

    for query_id, query_text, _ in queries:
        query_state = state["queries"].get(str(query_id), {})
        if query_state.get("stage") in ["initial_sparql_complete", "complete"]:
            print(f"✅ Query {query_id}: Using cached SPARQL")
            sparql_queries[query_id] = query_state["sparql_query"]
        else:
            # Build prompt for this query
            extracted = extracted_data[query_id]
            relevant_iris = extracted.get("relevant_iris", [])
            llm_prompt = build_llm_prompt(extracted, query_text, relevant_iris=relevant_iris)
            queries_needing_sparql.append((f"initial_sparql_{query_id}", llm_prompt, query_id))

            # Save prompt to state so we can retrieve it later
            query_state = state["queries"].get(str(query_id), {})
            query_state["llm_prompt"] = llm_prompt
            state["queries"][str(query_id)] = query_state

    if queries_needing_sparql:
        print(f"\n🔄 Generating SPARQL for {len(queries_needing_sparql)} queries in single batch...")
        prompts = [(custom_id, prompt) for custom_id, prompt, _ in queries_needing_sparql]
        batch_sparql = generate_sparql_batch(
            prompts,
            batch_handler,
            model_name,
            system_initial,
            state_manager,
            state,
            "Initial SPARQL generation (all queries)"
        )

        # Save SPARQL results
        for custom_id, _, query_id in queries_needing_sparql:
            if custom_id in batch_sparql:
                sparql_queries[query_id] = batch_sparql[custom_id]
                query_state = state["queries"].get(str(query_id), {})
                query_state["sparql_query"] = batch_sparql[custom_id]
                query_state["stage"] = "initial_sparql_complete"
                query_state["current_attempt"] = 1
                if "initial_sparql_batch_id" in query_state:
                    del query_state["initial_sparql_batch_id"]
                state["queries"][str(query_id)] = query_state
            else:
                print(f"❌ Query {query_id}: No SPARQL generated")
        state_manager.save_state(state)
        print(f"✅ Generated SPARQL for {len(batch_sparql)} queries")

    # ========== STAGE 3: VALIDATE AND ITERATIVELY CORRECT ==========
    print("\n" + "="*80)
    print("STAGE 3: VALIDATING AND CORRECTING QUERIES")
    print("="*80)

    # Track which queries still need work
    active_queries = set(sparql_queries.keys())
    attempt = 1

    while attempt <= max_attempts and active_queries:
        print(f"\n{'='*80}")
        print(f"ATTEMPT {attempt}/{max_attempts} - Validating {len(active_queries)} queries")
        print(f"{'='*80}")

        failed_queries = []

        # Validate each query sequentially
        for query_id in list(active_queries):
            query_text, expected_label = queries_dict[query_id]
            sparql_query = sparql_queries[query_id]
            query_state = state["queries"].get(str(query_id), {})

            # Set up query directory
            base_query_dir = os.path.join(run_dir, f"query_{query_id}")
            sanitized_category = sanitize_filename(expected_label) if expected_label else ""
            query_dir = f"{base_query_dir}_{sanitized_category}" if sanitized_category else base_query_dir
            os.makedirs(query_dir, exist_ok=True)

            print(f"\n🔍 Validating Query {query_id} (attempt {attempt})...")

            # Check if we already validated this attempt
            if query_state.get(f"attempt_{attempt}_result"):
                result = query_state[f"attempt_{attempt}_result"]
                if result != "error":
                    print(f"✅ Query {query_id}: Using cached successful result")
                    active_queries.discard(query_id)
                    results[query_id] = {
                        "success": True,
                        "result": result,
                        "query": query_text,
                        "expected_label": expected_label,
                        "sparql": sparql_query,
                        "attempts": attempt
                    }
                    continue
                else:
                    error_msg = query_state.get(f"attempt_{attempt}_error", "Unknown error")
                    failed_queries.append((query_id, query_text, sparql_query, error_msg))
                    continue

            # Execute query against GraphDB
            result, query_errors = run_query_graph_with_retries(sparql_query, max_retries=1)

            save_attempt_outputs(query_dir, attempt, sparql_query)

            if result is not None:
                # Success!
                print(f"✅ Query {query_id}: Succeeded on attempt {attempt}")
                active_queries.discard(query_id)

                # Save success to state
                query_state[f"attempt_{attempt}_result"] = result
                query_state["stage"] = "complete"
                query_state["success"] = True
                query_state["sparql_query"] = sparql_query
                state["queries"][str(query_id)] = query_state
                state_manager.save_state(state)

                # Save outputs
                run_data = {
                    "user_query": query_text,
                    "query_id": query_id,
                    "extracted": extracted_data[query_id],
                    "llm_prompt": query_state.get("llm_prompt", ""),
                    "sparql_query": sparql_query,
                    "graphdb_result": result
                }
                save_query_outputs(query_id, run_data, query_dir)
                save_attempt_outputs(query_dir, attempt, sparql_query, result=result)
                save_query_log(query_text, sparql_query)

                results[query_id] = {
                    "success": True,
                    "result": result,
                    "query": query_text,
                    "expected_label": expected_label,
                    "sparql": sparql_query,
                    "attempts": attempt
                }
            else:
                # Failed - add to correction batch
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

                failed_queries.append((query_id, query_text, sparql_query, error_msg))

        # If no failures or max attempts reached, we're done
        if not failed_queries or attempt >= max_attempts:
            if failed_queries and attempt >= max_attempts:
                print(f"\n⚠️  {len(failed_queries)} queries failed after {max_attempts} attempts")
                for query_id, query_text, _, _ in failed_queries:
                    query_state = state["queries"].get(str(query_id), {})
                    query_state["stage"] = "failed"
                    query_state["success"] = False
                    state["queries"][str(query_id)] = query_state

                    results[query_id] = {
                        "success": False,
                        "result": None,
                        "query": query_text,
                        "expected_label": queries_dict[query_id][1],
                        "error": query_state.get("error_log", "Max attempts reached"),
                        "attempts": max_attempts
                    }
                state_manager.save_state(state)
            break

        # ========== BATCH CORRECT ALL FAILURES ==========
        print(f"\n🔄 Batch correcting {len(failed_queries)} failed queries...")

        correction_prompts = []
        for query_id, query_text, sparql_query, error_msg in failed_queries:
            extracted = extracted_data[query_id]
            relevant_iris = extracted.get("relevant_iris", [])

            error_prompt = build_error_correction_prompt(
                compress_error(error_msg),
                sparql_query,
                extracted,
                query_text,
                relevant_iris=relevant_iris
            )

            # Save error correction prompt
            base_query_dir = os.path.join(run_dir, f"query_{query_id}")
            sanitized_category = sanitize_filename(queries_dict[query_id][1])
            query_dir = f"{base_query_dir}_{sanitized_category}" if sanitized_category else base_query_dir
            save_attempt_outputs(query_dir, attempt, sparql_query, error_correction_prompt=error_prompt)

            custom_id = f"correction_{query_id}_{attempt}"
            correction_prompts.append((custom_id, error_prompt, query_id))

        # Submit correction batch
        prompts = [(custom_id, prompt) for custom_id, prompt, _ in correction_prompts]
        corrected_sparql = generate_sparql_batch(
            prompts,
            batch_handler,
            model_name,
            system_error,
            state_manager,
            state,
            f"Error correction batch (attempt {attempt})"
        )

        # Update SPARQL queries with corrections
        for custom_id, _, query_id in correction_prompts:
            if custom_id in corrected_sparql:
                sparql_queries[query_id] = corrected_sparql[custom_id]
                query_state = state["queries"].get(str(query_id), {})
                query_state["sparql_query"] = corrected_sparql[custom_id]
                query_state["current_attempt"] = attempt + 1
                if f"correction_{attempt}_batch_id" in query_state:
                    del query_state[f"correction_{attempt}_batch_id"]
                state["queries"][str(query_id)] = query_state
            else:
                print(f"❌ Query {query_id}: No correction received")
                active_queries.discard(query_id)

        state_manager.save_state(state)
        attempt += 1

    print(f"\n✅ Pipeline completed: {len([r for r in results.values() if r['success']])} successful, {len([r for r in results.values() if not r['success']])} failed")
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

            row = {
                "Query_ID": query_id,
                "Question": query_text,
                "Category": expected_label,
                "Model": model_name,
                "Success": "Yes" if result_info["success"] else "No",
                "Attempts": result_info.get("attempts", 0),
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
    Coordinate clause extraction, SPARQL generation, retries, and persistence using Batch API.
    Enhanced with full batch processing: all queries batched together at each stage.
    """
    check_graphdb_connection()

    parser = argparse.ArgumentParser(description="Run end-to-end SPARQL pipeline with full Batch API processing.")
    parser.add_argument("--query", type=str, help="Single natural-language user query.")
    parser.add_argument("--csv", type=str, help="Path to CSV file with queries (e.g., test_queries_200.csv)")
    parser.add_argument("--max-attempts", type=int, default=10, help="Max correction attempts per query.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to process from CSV (e.g., --limit 4 for first 4 queries)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run if state exists.")
    parser.add_argument("--state-dir", type=str, default="pipeline_state", help="Directory for state files.")
    parser.add_argument("--outputs-dir", type=str, default="outputs_batch", help="Directory for outputs.")

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

    # Initialize OpenAI client and batch handler
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment.")

    client = OpenAI(api_key=api_key)
    batch_handler = BatchAPIHandler(client)

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

    # Process all queries with batch approach
    results = process_all_queries_batch(
        queries=queries,
        batch_handler=batch_handler,
        model_name=model_name,
        system_initial=system_initial,
        system_error=system_error,
        max_attempts=max_attempts,
        run_dir=run_dir,
        state_manager=state_manager,
        state=state
    )

    # Build results summary
    results_summary = {
        "total": len(queries),
        "successful": sum(1 for r in results.values() if r["success"]),
        "failed": sum(1 for r in results.values() if not r["success"]),
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
            "error": result_info.get("error")
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
    print(f"📊 Summary saved to: {summary_path}")
    print(f"📄 CSV results saved to: {csv_path}")


if __name__ == "__main__":
    main()
