# SPARQL Query Generation Pipeline - Technical Report

## Executive Summary

This document provides a detailed technical analysis of the `run_query_pipeline.py` system, which implements an end-to-end natural language to SPARQL query generation pipeline with intelligent error correction and retry mechanisms.

**System Purpose:** Convert natural language queries into valid SPARQL queries for execution against a GraphDB knowledge base, with automatic error detection and correction.

**Date Generated:** October 15, 2025

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Pipeline Stages](#pipeline-stages)
3. [Function-Level Documentation](#function-level-documentation)
4. [Data Flow Diagram](#data-flow-diagram)
5. [Error Handling & Retry Logic](#error-handling--retry-logic)
6. [External Dependencies](#external-dependencies)
7. [File Outputs & Artifacts](#file-outputs--artifacts)

---

## Architecture Overview

### System Components

The pipeline consists of the following major components:

1. **Argument Parser & Configuration Layer** - Handles CLI arguments and environment variables
2. **Clause Extraction Engine** - Extracts structured semantic clauses from natural language
3. **IRI Resolution System** - Maps natural language terms to ontology IRIs using embeddings
4. **Prompt Construction Layer** - Builds context-rich prompts for LLM consumption
5. **LLM Interface** - Communicates with OpenAI API for SPARQL generation
6. **Query Execution Engine** - Executes SPARQL against GraphDB
7. **Error Correction Loop** - Iteratively fixes query errors
8. **Persistence Layer** - Saves all artifacts for debugging and analysis

### Technology Stack

- **Language:** Python 3.x
- **LLM Provider:** OpenAI (GPT-5 default)
- **Database:** GraphDB (SPARQL endpoint)
- **Embedding Model:** text-embedding-3-large
- **Key Libraries:** openai, subprocess, json, argparse, dotenv

---

## Pipeline Stages

### Stage 0: Initialization & Setup

**Entry Point:** `main()` function

**Process:**
1. Parse command-line arguments (user query, max attempts, system prompts)
2. Load environment variables from `.env` file
3. Resolve system prompts (priority: CLI args > file > defaults)
4. Initialize output directory with timestamp
5. Prepare run data dictionary for tracking

**Key Variables:**
- `user_query`: Natural language input from user
- `model_name`: LLM model from `LLM_MODEL` env var (default: "gpt-5")
- `max_attempts`: Maximum correction cycles (default: 5)
- `system_initial`: System prompt for initial SPARQL generation
- `system_error`: System prompt for error correction

**Outputs:**
- `run_dir`: Timestamped directory (`outputs/run_YYYYMMDD_HHMMSS_microseconds/`)

---

### Stage 1: Clause Extraction + IRI Resolution

**Primary Function:** `run_extract_chunk(user_query: str)`

#### Sub-Process 1A: Clause Extraction via LLM

**External Script:** `extract_chunk.py`

**Execution:**
```python
subprocess.run([sys.executable, "extract_chunk.py", user_query])
```

**What `extract_chunk.py` Does:**

1. **Loads Schema & Prompt:**
   - Reads `clause_chunk_schema.json` (structured JSON schema for clause extraction)
   - Reads `clause_chunker_prompt_new.txt` (system prompt for extraction task)

2. **LLM Call (Structured Output):**
   ```python
   client.chat.completions.create(
       model=model_name,
       messages=[
           {"role": "system", "content": system_prompt},
           {"role": "user", "content": "Extract clauses from this query into the schema. Query:\n" + user_query}
       ],
       response_format={
           "type": "json_schema",
           "json_schema": {"name": "ExtractedClauses", "strict": True, "schema": clause_chunk_schema}
       }
   )
   ```

3. **Output Format:**
   ```json
   {
       "clauses": [
           {
               "type": "filter|modifier|select|connective",
               "subject_hint": "domain entity",
               "relation_hint": "property or relationship",
               "object_hint": "target value",
               "filter": {"op": "eq|lt|gt|between", "value": "...", "unit": "..."},
               "modifier": {"kind": "group|limit|sort", "k": 10, "by": "field"},
               "enum_values": ["option1", "option2"]
           }
       ],
       "warnings": ["list of parsing warnings"]
   }
   ```

**Example:**
- Input: "Find steel manufacturers in Germany with ISO 9001 certification"
- Output:
  ```json
  {
      "clauses": [
          {"type": "select", "subject_hint": "manufacturer", "object_hint": "steel"},
          {"type": "filter", "subject_hint": "location", "object_hint": "Germany"},
          {"type": "filter", "subject_hint": "certificate", "object_hint": "ISO 9001"}
      ],
      "warnings": []
  }
  ```

#### Sub-Process 1B: Clause Compression

**Function:** `compact_chunk(extracted)` from `compress_chunk.py`

**Purpose:** Remove redundant/empty fields to reduce token usage

**Logic:**
1. **Field Filtering:** Only keeps meaningful keys: `type`, `subject_hint`, `relation_hint`, `object_hint`, `enum_values`, `filter`, `modifier`, `connective`
2. **Empty Value Removal:** Strips `None`, `""`, `[]`, `{}`
3. **Filter Validation:** Only keeps filters with meaningful operators (`lt`, `lte`, `eq`, `gte`, `gt`, `between`, etc.)
4. **Modifier Validation:** Removes no-op modifiers like `{"kind": "group", "k": 0, "by": ""}`

**Example Transformation:**
```json
// Before
{"type": "filter", "subject_hint": "age", "relation_hint": "", "object_hint": "", "filter": {"op": "eq", "value": 5}}

// After
{"type": "filter", "subject_hint": "age", "filter": {"op": "eq", "value": 5}}
```

#### Sub-Process 1C: IRI Extraction via Embeddings

**Function:** `extract_iris_from_query(user_query: str, top_k: int = 5)`

**Purpose:** Find ontology IRIs that semantically match the user query

**Embedding System Pipeline:**

1. **Embedder Initialization:** `get_embedder()`
   - Singleton pattern (lazy initialization)
   - Loads once and reuses globally

2. **Sub-Function: `get_embedder()`**
   
   **Creates:** `EnhancedCSVEmbedder` instance
   
   **Loads:**
   - `label_iri_mappings.csv` - Maps natural language labels to IRIs
   - `industries_enhanced.csv` - Industry classifications with embeddings
   - `material_caps_enhanced.csv` - Material capabilities
   - `certificates_enhanced_fixed.csv` - Certification data
   - `process_embeddings_enhanced_with_def.csv` - Process definitions
   
   **Caching:** Checks `embedding_cache/enhanced_csv_embeddings.pkl` for pre-computed embeddings

3. **Embedding Search:** `embedder.search_closest_labels(user_query, top_k=10)`
   
   **Process:**
   
   a. **Query Embedding Generation:**
   ```python
   response = client.embeddings.create(
       model="text-embedding-3-large",
       input=[user_query]
   )
   query_embedding = np.array(response.data[0].embedding)
   ```
   
   b. **Normalization:**
   ```python
   query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
   ```
   
   c. **Cosine Similarity Calculation:**
   ```python
   similarities = np.dot(embedding_norms, query_norm)
   ```
   
   d. **IRI Matching Strategies (in order of precedence):**
   
   - **Exact Match:** Direct lookup in `iri_mapping` dictionary
   - **Capability Variants:** 
     - Remove spaces + add "Capability": `no_spaces + 'capability'`
     - CamelCase + Capability: `(CamelCase + 'Capability').lower()`
   - **Substring Matching:**
     - Query contained in label: `score = len(query) / len(label)`
     - Label contained in query: `score = len(label) / len(query)`
   - **Word Overlap:** Jaccard similarity of word sets
   
   e. **Result Construction:**
   ```python
   EmbeddingResult(
       label=label,
       score=float(cosine_similarity),
       iri=matched_iri,
       category=metadata['category'],
       hierarchy=metadata['hierarchy'],
       definition=metadata['definition'],
       source_file=source_name
   )
   ```

4. **Filtering & Output:**
   - Filters out results with "No IRI found"
   - Returns top 10 results sorted by similarity score

**Output Example:**
```json
"relevant_iris": [
    {"label": "Steel Manufacturing", "iri": "http://example.org/ontology#SteelManufacturingCapability"},
    {"label": "Germany", "iri": "http://example.org/ontology#Germany"},
    {"label": "ISO 9001", "iri": "http://example.org/ontology#ISO9001Certificate"}
]
```

#### Final Stage 1 Output:

**Compressed Extraction with IRIs:**
```json
{
    "clauses": [...],
    "mapping": ["warnings"],
    "relevant_iris": [
        {"label": "...", "iri": "..."}
    ]
}
```

**Storage:** Saved to `run_dir/extracted.json`

---

### Stage 2: Prompt Construction

**Function:** `build_llm_prompt(extracted_chunk, user_query, relevant_iris, template_path, reference_turtle_path)`

**Purpose:** Create a comprehensive prompt for SPARQL generation

**Process:**

1. **Load Template:** `sparql_prompt_template_new.txt`
   - Contains instructions for SPARQL generation
   - Includes placeholders for dynamic content

2. **Load Reference Ontology:** `reference_turtle.txt`
   - Contains Turtle/RDF ontology definitions
   - Provides schema context for SPARQL generation

3. **Template Variable Substitution:**
   ```python
   txt_body = template.replace("{USER_QUERY}", user_query)
   txt_body = txt_body.replace("{EXTRACTED_Clauses}", json.dumps(extracted_chunk))
   txt_body = txt_body.replace("{EXTRACTED_IRI}", json.dumps(relevant_iris))
   txt_body = txt_body.replace("{REFERENCE_TURTLE_DATA}", reference_turtle_data)
   ```

4. **Final Assembly:**
   - If template has `{TXT}` wrapper: `template.replace("{TXT}", txt_body)`
   - Otherwise: returns `txt_body` directly

**Prompt Structure:**
```
[System Instructions]
You are an expert in semantic web and SPARQL query generation...

[User Query]
{user_query}

[Extracted Semantic Clauses]
{compressed_clauses_json}

[Relevant Ontology IRIs]
{iri_list_json}

[Reference Ontology (Turtle)]
{ontology_definitions}

[Task]
Generate a valid SPARQL query that...
```

**Storage:** Saved to `run_dir/llm_prompt.txt`

---

### Stage 3: Initial SPARQL Generation

**Function:** `get_sparql_from_llm(prompt: str, model: str, system_prompt: str)`

**Purpose:** Call OpenAI API to generate SPARQL query

**Process:**

1. **API Key Validation:**
   ```python
   api_key = os.getenv("OPENAI_API_KEY")
   if not api_key:
       raise RuntimeError("OPENAI_API_KEY not found")
   ```

2. **OpenAI Client Initialization:**
   ```python
   client = OpenAI(api_key=api_key)
   ```

3. **Message Construction:**
   ```python
   messages = [
       {"role": "system", "content": SYSTEM_PROMPT_INITIAL},
       {"role": "user", "content": llm_prompt}
   ]
   ```
   
   **System Prompt (Initial):**
   ```
   You are an expert in semantic web and SPARQL query generation. 
   Given the user request and reference Turtle, synthesize a single, valid SPARQL query. 
   Strictly follow the ontology naming conventions present in the reference Turtle. 
   Return ONLY the SPARQL query—no explanations.
   ```

4. **API Call:**
   ```python
   resp = client.chat.completions.create(
       model=model,  # e.g., "gpt-5"
       messages=messages,
       response_format={"type": "text"}
   )
   ```

5. **Response Extraction:**
   ```python
   content = resp.choices[0].message.content or ""
   return content.strip()
   ```

**Output Example:**
```sparql
PREFIX ex: <http://example.org/ontology#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?manufacturer ?name
WHERE {
    ?manufacturer rdf:type ex:Manufacturer ;
                  ex:hasCapability ex:SteelManufacturingCapability ;
                  ex:locatedIn ex:Germany ;
                  ex:hasCertificate ex:ISO9001Certificate ;
                  ex:name ?name .
}
```

**Storage:** Saved to `run_dir/attempt_1/sparql_query.txt`

---

### Stage 4: Query Execution Loop (with Retry & Correction)

**Main Loop Structure:**
```python
attempt = 1
while attempt <= max_attempts:
    # Execute query
    result, query_errors = run_query_graph_with_retries(sparql_query, max_retries=1)
    
    if result is not None:
        # SUCCESS - break loop
        save_results()
        break
    else:
        # FAILURE - attempt correction
        error_prompt = build_error_correction_prompt(error_msg, sparql_query, ...)
        sparql_query = get_sparql_from_llm(error_prompt, model, SYSTEM_PROMPT_ERROR)
        attempt += 1
```

#### Sub-Process 4A: Query Execution

**Function:** `run_query_graph_with_retries(sparql_query: str, max_retries: int = 5)`

**Purpose:** Execute SPARQL against GraphDB with basic retry logic

**External Script:** `query_Graph.py`

**Process:**

1. **Subprocess Execution:**
   ```python
   result = subprocess.run(
       [sys.executable, "query_Graph.py", sparql_query],
       capture_output=True,
       text=True,
       encoding="utf-8",
       errors="replace",
       timeout=60
   )
   ```

2. **What `query_Graph.py` Does:**
   
   a. **SPARQL Wrapper Initialization:**
   ```python
   from SPARQLWrapper import SPARQLWrapper, JSON
   endpoint_url = "http://localhost:7200/repositories/sudokn"
   sparql = SPARQLWrapper(endpoint_url)
   ```
   
   b. **Query Execution:**
   ```python
   sparql.setQuery(sparql_query)
   sparql.setReturnFormat(JSON)
   results = sparql.query().convert()
   bindings = results.get("results", {}).get("bindings", [])
   ```
   
   c. **Result Formatting (TSV):**
   ```python
   if bindings:
       var_names = list(bindings[0].keys())
       writer = csv.writer(stdout, delimiter='\t')
       writer.writerow(var_names)  # Header
       for result in bindings:
           row = [result.get(var, {}).get("value", "") for var in var_names]
           writer.writerow(row)
   ```
   
   d. **Error Handling:**
   ```python
   except Exception:
       tb_str = traceback.format_exc()
       compressed = compress_error(tb_str)
       print(compressed, file=sys.stderr)
   ```
   
   e. **Statistics Logging:**
   - Saves to `query_Graph_stats.jsonl` (JSONL format)
   - Tracks: run_number, timestamp, query, success, error

3. **Return Value Interpretation:**
   - **Success:** `returncode == 0` and no stderr → returns (stdout, [])
   - **Failure:** Any error → returns (None, [error_messages])

4. **Retry Logic:**
   - Attempts up to `max_retries` times (default: 1 in this context)
   - Sleeps 1 second between retries: `time.sleep(1)`

**Output on Success:**
```
?manufacturer	?name
http://example.org/data#Manufacturer1	Steel Corp Germany
http://example.org/data#Manufacturer2	Deutsche Stahl GmbH
```

**Output on Failure:**
```
QueryBadFormed: Invalid property IRI
Response:
HTTP 400 Bad Request
Unbound variable in SELECT clause
```

#### Sub-Process 4B: Error Analysis & Compression

**Function:** `compress_error(traceback_str: str)`

**Purpose:** Extract essential error information from verbose tracebacks

**Algorithm:**

1. **Exception Line Extraction:**
   ```python
   exception_lines = [line for line in traceback_str.splitlines() 
                      if re.match(r'^\w+(\.\w+)*:', line)]
   last_exception = exception_lines[-1] if exception_lines else ""
   ```
   
   Matches patterns like: `QueryBadFormed:`, `ValueError:`, `SPARQLWrapperException:`

2. **Response Section Extraction:**
   ```python
   response_match = re.search(r'(Response:\n.*)', traceback_str, re.DOTALL)
   response = response_match.group(1).strip() if response_match else ""
   ```
   
   Captures everything after "Response:" including HTTP details

3. **Compressed Output:**
   ```python
   return f"{last_exception}\n{response}"
   ```

**Example:**
```
Input (verbose):
Traceback (most recent call last):
  File "query_Graph.py", line 45, in main
    results = sparql.query().convert()
  File "/usr/lib/python3/SPARQLWrapper.py", line 234, in convert
    raise QueryBadFormed(response.text)
SPARQLWrapper.SPARQLExceptions.QueryBadFormed: Invalid IRI
Response:
HTTP 400 Bad Request
{"error": "Unbound variable ?invalidVar in SELECT clause"}

Output (compressed):
SPARQLWrapper.SPARQLExceptions.QueryBadFormed: Invalid IRI
Response:
HTTP 400 Bad Request
{"error": "Unbound variable ?invalidVar in SELECT clause"}
```

#### Sub-Process 4C: Error Correction Prompt Construction

**Function:** `build_error_correction_prompt(error_message, previous_sparql, extracted_chunk, user_query, relevant_iris, ...)`

**Purpose:** Guide LLM to fix the broken SPARQL query

**Process:**

1. **Load Error Template:** `sparql_error_correction_prompt.txt`

2. **Load Reference Ontology:** `reference_turtle.txt` (same as initial)

3. **Template Substitution:**
   ```python
   txt_body = template.replace("{USER_QUERY}", user_query)
   txt_body = txt_body.replace("{ERROR_MESSAGE}", error_message)  # Compressed error
   txt_body = txt_body.replace("{PREVIOUS_SPARQL}", previous_sparql)
   txt_body = txt_body.replace("{EXTRACTED_Clauses}", json.dumps(extracted_chunk))
   txt_body = txt_body.replace("{EXTRACTED_IRI}", json.dumps(relevant_iris))
   txt_body = txt_body.replace("{REFERENCE_TURTLE_DATA}", reference_turtle_data)
   ```

**Error Correction Prompt Structure:**
```
[System Instructions]
You are an expert in semantic web and SPARQL debugging...

[Original User Query]
{user_query}

[Error from GraphDB]
{compressed_error_message}

[Previous SPARQL Query (FAILED)]
{previous_sparql}

[Extracted Semantic Clauses]
{compressed_clauses_json}

[Relevant Ontology IRIs]
{iri_list_json}

[Reference Ontology (Turtle)]
{ontology_definitions}

[Task]
Diagnose the error and return a CORRECTED SPARQL query. 
Preserve the original intent but fix syntax/semantic errors.
Return ONLY the corrected SPARQL—no explanations.
```

**Storage:** Saved to `run_dir/attempt_{N}/error_correction_prompt.txt`

#### Sub-Process 4D: Corrected SPARQL Generation

**Function:** `get_sparql_from_llm(error_prompt, model, SYSTEM_PROMPT_ERROR)`

**Same as Stage 3, but with different inputs:**

**System Prompt (Error Correction):**
```
You are an expert in semantic web and SPARQL debugging. 
You are given the last SPARQL query and the GraphDB error details. 
Diagnose the error, minimally correct the SPARQL (preserve intent), 
and return ONLY the corrected SPARQL. 
Honor the ontology naming conventions from the reference Turtle. 
No explanations.
```

**Output:** Corrected SPARQL query

**Storage:** Saved to `run_dir/attempt_{N+1}/sparql_query.txt`

#### Loop Termination Conditions:

1. **Success:** Query executes without errors → save results and break
2. **Max Attempts Reached:** All correction attempts exhausted → save error log
3. **Manual Interruption:** User cancels execution

---

### Stage 5: Result Persistence

**Multiple persistence functions handle different stages:**

#### 5A: Partial Run Outputs (Early Persistence)

**Function:** `save_partial_run_outputs(run_data, run_dir, model_name)`

**Called:** After Stage 2 (prompt construction), before execution

**Files Created:**
- `run_dir/user_query.txt` - Original natural language query
- `run_dir/extracted.json` - Compressed clauses + IRIs
- `run_dir/llm_prompt.txt` - Initial generation prompt
- `run_dir/model.txt` - LLM model name (e.g., "gpt-5")

**Purpose:** Enable debugging even if pipeline crashes during execution

#### 5B: Attempt Outputs (Per-Iteration Persistence)

**Function:** `save_attempt_outputs(run_dir, attempt_idx, sparql_query, result, error_msg, error_correction_prompt)`

**Called:** After each execution attempt (success or failure)

**Directory Structure:**
```
run_dir/
├── attempt_1/
│   ├── sparql_query.txt
│   ├── graphdb_result.txt (if success)
│   ├── error.txt (if failure)
│   └── error_correction_prompt.txt (if failure, for next attempt)
├── attempt_2/
│   ├── sparql_query.txt
│   ├── error.txt
│   └── error_correction_prompt.txt
└── attempt_3/
    ├── sparql_query.txt
    └── graphdb_result.txt (success!)
```

**Purpose:** Track evolution of SPARQL across correction iterations

#### 5C: Final Run Outputs (Complete Persistence)

**Function:** `save_run_outputs(run_data, outputs_dir, model_name, run_dir)`

**Called:** At pipeline completion (success or failure)

**Files Created:**
- `run_dir/user_query.txt` - User's original query
- `run_dir/llm_prompt.txt` - Initial LLM prompt
- `run_dir/sparql_query.txt` - Final SPARQL query (last attempt)
- `run_dir/graphdb_result.txt` - Query results (if successful)
- `run_dir/extracted.json` - Clause extraction + IRI data
- `run_dir/error_log.txt` - Concatenated error log (if failures occurred)
- `run_dir/model.txt` - LLM model identifier

**Additional Action:** Prints `"Run outputs saved to: {run_dir}"`

#### 5D: Query Log (Append-Only History)

**Function:** `save_query_log(user_query, sparql_query, log_path="query_log.json")`

**Called:** Only on successful query execution

**Format:** JSONL (JSON Lines) - one JSON object per line
```jsonl
{"user_query": "Find steel manufacturers in Germany", "sparql_query": "PREFIX ex: ..."}
{"user_query": "Show all ISO 9001 certificates", "sparql_query": "PREFIX ex: ..."}
```

**Purpose:** Build historical dataset of successful query pairs for fine-tuning or analysis

---

## Function-Level Documentation

### Helper Functions

#### `compress_error(traceback_str: str) -> str`
**Module:** `run_query_pipeline.py`  
**Purpose:** Extract concise error summary from verbose exception tracebacks  
**Algorithm:** Regex-based extraction of exception lines + response sections  
**Returns:** Compressed error string  

#### `get_embedder() -> EnhancedCSVEmbedder`
**Module:** `run_query_pipeline.py`  
**Purpose:** Lazy singleton initialization of embedding system  
**Side Effects:** Loads 4 CSV files + IRI mappings into memory (~100MB)  
**Caching:** Uses global `_embedder_instance` to avoid reloading  
**Returns:** Configured embedder instance  

#### `_ensure_dir(path: str)`
**Module:** `run_query_pipeline.py`  
**Purpose:** Create directory if it doesn't exist  
**Implementation:** `os.makedirs(path, exist_ok=True)`  
**Usage:** All persistence functions call this before writing  

#### `_read_text_file(path: str) -> str`
**Module:** `run_query_pipeline.py`  
**Purpose:** Read entire UTF-8 text file  
**Usage:** Load system prompts from files when `--system-initial-file` or `--system-error-file` provided  
**Returns:** File contents as string  

#### `_write(path, content)` (nested in `save_run_outputs`)
**Purpose:** Write content with exception handling  
**Error Handling:** Catches IO errors and prints to stderr, but doesn't crash pipeline  

### EnhancedCSVEmbedder Methods

#### `load_iri_mapping(iri_mapping_file: str)`
**Purpose:** Load label→IRI mappings from CSV into dictionary  
**Data Structure:** `self.iri_mapping = {label_lowercase: {iri, type, id, source_file}}`  
**Usage:** Called once during embedder initialization  

#### `embed_text_batch(texts: List[str], batch_size: int = 100) -> List[np.ndarray]`
**Purpose:** Generate embeddings for multiple texts efficiently  
**Batching:** Submits 100 texts per API call to reduce latency  
**Error Handling:** Returns zero vectors on failure (1536-dimensional)  
**Returns:** List of numpy arrays (embeddings)  

#### `process_csv_file(csv_file: str, source_name: str) -> Tuple[List[str], List[Dict]]`
**Purpose:** Parse CSV and prepare texts for embedding  
**Text Construction:** Combines `Label + ": " + Definition` for richer embeddings  
**Metadata Extraction:** Extracts Category, Hierarchy, Definition, Also Called  
**Returns:** (texts_to_embed, metadata_list)  

#### `embed_csv_files(csv_files: Dict[str, str], force_recompute: bool = False)`
**Purpose:** Batch embed all CSV files with caching  
**Cache Location:** `embedding_cache/enhanced_csv_embeddings.pkl`  
**Cache Contents:** embeddings dict, metadata dict, timestamp, model name  
**Behavior:** Skips computation if cache exists and `force_recompute=False`  

#### `find_best_iri_match(query_label: str, threshold: float = 0.7) -> Optional[str]`
**Purpose:** Fast heuristic IRI matching without embeddings  
**Strategies:** Exact match → Capability variants → Substring → Word overlap  
**Threshold:** Minimum similarity score 0.3 (30% match required)  
**Returns:** Best matching IRI or None  

#### `search_closest_labels(query: str, top_k: int = 5, source_filter: Optional[List[str]] = None)`
**Purpose:** Semantic search using cosine similarity  
**Algorithm:** 
1. Embed query
2. Normalize embeddings
3. Compute cosine similarities
4. Sort by score
5. Map to IRIs
**Returns:** List of `EmbeddingResult` objects  

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INPUT                                  │
│                   (Natural Language Query)                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: CLAUSE EXTRACTION + IRI RESOLUTION                         │
├─────────────────────────────────────────────────────────────────────┤
│  1A. extract_chunk.py                                               │
│      ├─ LLM (Structured JSON)                                       │
│      └─ Output: {clauses: [...], warnings: [...]}                   │
│                                                                      │
│  1B. compact_chunk()                                                │
│      └─ Remove empty/redundant fields                               │
│                                                                      │
│  1C. extract_iris_from_query()                                      │
│      ├─ get_embedder() [Singleton]                                  │
│      │   ├─ Load CSVs (industries, materials, certs, processes)    │
│      │   └─ Load IRI mappings                                       │
│      ├─ Embed query                                                 │
│      ├─ Cosine similarity search                                    │
│      ├─ IRI matching (exact/variant/substring/word)                 │
│      └─ Output: [{label, iri}, ...]                                 │
│                                                                      │
│  Output: {clauses, mapping, relevant_iris}                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: PROMPT CONSTRUCTION                                        │
├─────────────────────────────────────────────────────────────────────┤
│  build_llm_prompt()                                                 │
│  ├─ Load template (sparql_prompt_template_new.txt)                 │
│  ├─ Load ontology (reference_turtle.txt)                           │
│  ├─ Substitute: USER_QUERY, EXTRACTED_Clauses, EXTRACTED_IRI       │
│  └─ Output: Complete prompt string                                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 3: INITIAL SPARQL GENERATION                                  │
├─────────────────────────────────────────────────────────────────────┤
│  get_sparql_from_llm(prompt, model, SYSTEM_PROMPT_INITIAL)         │
│  ├─ OpenAI API call                                                 │
│  ├─ Model: gpt-5 (or env override)                                 │
│  └─ Output: SPARQL query string                                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 4: EXECUTION LOOP (max_attempts iterations)                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ 4A. run_query_graph_with_retries()                         │    │
│  │     ├─ Subprocess: query_Graph.py                          │    │
│  │     │   ├─ SPARQLWrapper                                   │    │
│  │     │   ├─ GraphDB endpoint (localhost:7200)               │    │
│  │     │   └─ Return: TSV results or error                    │    │
│  │     └─ Output: (result, errors)                            │    │
│  └────────────────────────┬───────────────────────────────────┘    │
│                            │                                         │
│                    ┌───────┴───────┐                                │
│                    │   SUCCESS?    │                                │
│                    └───┬───────┬───┘                                │
│                        │       │                                     │
│                 YES ───┘       └─── NO                              │
│                  │                   │                               │
│                  │                   ▼                               │
│                  │    ┌─────────────────────────────────────┐       │
│                  │    │ 4B. compress_error()                │       │
│                  │    │     └─ Extract key error info       │       │
│                  │    └──────────────┬──────────────────────┘       │
│                  │                   │                               │
│                  │                   ▼                               │
│                  │    ┌─────────────────────────────────────┐       │
│                  │    │ 4C. build_error_correction_prompt() │       │
│                  │    │     ├─ Include: error, prev query   │       │
│                  │    │     └─ Template: error_correction..│       │
│                  │    └──────────────┬──────────────────────┘       │
│                  │                   │                               │
│                  │                   ▼                               │
│                  │    ┌─────────────────────────────────────┐       │
│                  │    │ 4D. get_sparql_from_llm()           │       │
│                  │    │     (SYSTEM_PROMPT_ERROR)           │       │
│                  │    │     └─ Output: Corrected SPARQL     │       │
│                  │    └──────────────┬──────────────────────┘       │
│                  │                   │                               │
│                  │                   └───────┐                       │
│                  │                           │                       │
│                  │                    attempt += 1                   │
│                  │                           │                       │
│                  │              ┌────────────┘                       │
│                  │              │ (Loop back to 4A)                  │
│                  │              └─ if attempt <= max_attempts        │
│                  │                                                   │
│                  └────────────────── BREAK ──────────────────────────┤
│                                                                      │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 5: PERSISTENCE                                                │
├─────────────────────────────────────────────────────────────────────┤
│  ├─ save_partial_run_outputs() [After Stage 2]                     │
│  │   └─ user_query.txt, extracted.json, llm_prompt.txt             │
│  │                                                                  │
│  ├─ save_attempt_outputs() [Each iteration]                        │
│  │   └─ attempt_{N}/ directory with query, result/error            │
│  │                                                                  │
│  ├─ save_run_outputs() [Final]                                     │
│  │   └─ All artifacts + error_log.txt                              │
│  │                                                                  │
│  └─ save_query_log() [Success only]                                │
│      └─ Append to query_log.json (JSONL)                           │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                       FINAL OUTPUT
              (GraphDB results or error report)
```

---

## Error Handling & Retry Logic

### Multi-Level Retry Architecture

The pipeline implements a **nested retry system** with two levels:

#### Level 1: Query Execution Retries (Network/Transient Errors)

**Function:** `run_query_graph_with_retries(max_retries=5)`

**Purpose:** Handle temporary failures (network timeouts, GraphDB busy)

**Strategy:**
- Retry same query up to 5 times
- 1-second delay between retries: `time.sleep(1)`
- No query modification between retries
- Collect all error messages

**Use Case Example:**
```
Attempt 1: Timeout (60 seconds)
Wait 1 second...
Attempt 2: Connection refused
Wait 1 second...
Attempt 3: Success!
```

#### Level 2: Semantic Correction Loop (Query Errors)

**Function:** `main()` while loop (`attempt <= max_attempts`)

**Purpose:** Fix semantic/syntactic errors in SPARQL via LLM

**Strategy:**
- Up to 5 correction iterations (default)
- Each iteration:
  1. Execute query (Level 1 retries)
  2. If failure → extract error
  3. Build error correction prompt
  4. Generate corrected SPARQL via LLM
  5. Loop back

**Use Case Example:**
```
Attempt 1: QueryBadFormed (unbound variable)
   → LLM corrects: adds missing WHERE clause
Attempt 2: InvalidIRI (wrong namespace)
   → LLM corrects: fixes IRI prefix
Attempt 3: Success!
```

### Error Categories & Handling

| Error Type | Level | Handling Strategy | Example |
|------------|-------|-------------------|---------|
| Network timeout | 1 | Simple retry | Connection to localhost:7200 timeout |
| HTTP 500 (server error) | 1 | Simple retry | GraphDB internal error |
| Malformed SPARQL | 2 | LLM correction | Missing PREFIX, syntax error |
| Unbound variable | 2 | LLM correction | Variable in SELECT not in WHERE |
| Invalid IRI | 2 | LLM correction | Wrong namespace or typo |
| Empty results | - | No retry | Query valid but no data matches |
| JSON parse error | - | Pipeline crash | extract_chunk.py output corrupt |

### Graceful Degradation

**If all retries fail:**
1. `error_log` accumulates all attempts
2. Final error saved to `run_dir/error_log.txt`
3. Pipeline completes (doesn't crash)
4. User receives: "All 5 attempts failed. Last error: ..."

**Partial persistence ensures:**
- Intermediate artifacts preserved for debugging
- Each attempt's SPARQL saved separately
- Logs available for post-mortem analysis

---

## External Dependencies

### Python Packages

| Package | Version | Purpose | Usage |
|---------|---------|---------|-------|
| `openai` | Latest | LLM API client | Clause extraction, SPARQL generation, embeddings |
| `python-dotenv` | Latest | Environment variables | Load API keys, config |
| `numpy` | Latest | Numerical operations | Embedding normalization, cosine similarity |
| `pandas` | Latest | CSV processing | Load/parse enhanced CSVs |
| `SPARQLWrapper` | Latest | SPARQL execution | GraphDB query interface |

### External Scripts

| Script | Language | Purpose | Input | Output |
|--------|----------|---------|-------|--------|
| `extract_chunk.py` | Python | Clause extraction | User query (CLI arg) | JSON (stdout) |
| `query_Graph.py` | Python | SPARQL execution | SPARQL query (CLI arg) | TSV results (stdout) or error (stderr) |
| `compress_chunk.py` | Python | Clause compression | JSON object | Compressed JSON |

### External Files (Data)

| File | Format | Purpose | Size |
|------|--------|---------|------|
| `clause_chunk_schema.json` | JSON | Structured extraction schema | ~5KB |
| `clause_chunker_prompt_new.txt` | Text | System prompt for extraction | ~2KB |
| `sparql_prompt_template_new.txt` | Text | SPARQL generation template | ~3KB |
| `sparql_error_correction_prompt.txt` | Text | Error correction template | ~3KB |
| `reference_turtle.txt` | Turtle/RDF | Ontology definitions | ~50KB |
| `label_iri_mappings.csv` | CSV | Label→IRI mappings | ~200KB |
| `industries_enhanced.csv` | CSV | Industry classifications | ~100KB |
| `material_caps_enhanced.csv` | CSV | Material capabilities | ~150KB |
| `certificates_enhanced_fixed.csv` | CSV | Certification data | ~80KB |
| `process_embeddings_enhanced_with_def.csv` | CSV | Process definitions | ~120KB |

### External Services

| Service | Type | Endpoint | Purpose |
|---------|------|----------|---------|
| OpenAI API | REST | api.openai.com | LLM inference + embeddings |
| GraphDB | SPARQL | localhost:7200/repositories/sudokn | Knowledge base queries |

---

## File Outputs & Artifacts

### Directory Structure (Example Run)

```
outputs/
└── run_20251015_143527_482341/
    ├── user_query.txt                      # Original NL query
    ├── extracted.json                      # Clauses + IRIs
    ├── llm_prompt.txt                      # Initial generation prompt
    ├── model.txt                           # "gpt-5"
    ├── attempt_1/
    │   ├── sparql_query.txt                # First SPARQL attempt
    │   ├── error.txt                       # GraphDB error (if failed)
    │   └── error_correction_prompt.txt     # Prompt for attempt 2
    ├── attempt_2/
    │   ├── sparql_query.txt                # Corrected SPARQL
    │   ├── error.txt                       # Still failed
    │   └── error_correction_prompt.txt     # Prompt for attempt 3
    ├── attempt_3/
    │   ├── sparql_query.txt                # Final SPARQL
    │   └── graphdb_result.txt              # SUCCESS!
    ├── sparql_query.txt                    # Final query (copy)
    └── graphdb_result.txt                  # Final result (copy)
```

### Global Files

| File | Format | Purpose | Append/Overwrite |
|------|--------|---------|------------------|
| `query_log.json` | JSONL | Historical query pairs | Append |
| `query_Graph_stats.jsonl` | JSONL | GraphDB execution stats | Append |
| `embedding_cache/enhanced_csv_embeddings.pkl` | Pickle | Cached embeddings | Overwrite |

---

## Performance Characteristics

### Time Complexity

| Stage | Operation | Typical Time |
|-------|-----------|--------------|
| 1A | Clause extraction (LLM) | 2-5 seconds |
| 1B | Compression | <0.1 seconds |
| 1C | IRI embedding search | 1-3 seconds (cached) / 30-60 seconds (first run) |
| 2 | Prompt construction | <0.1 seconds |
| 3 | Initial SPARQL (LLM) | 3-8 seconds |
| 4A | Query execution | 0.5-2 seconds (per attempt) |
| 4C+4D | Error correction (LLM) | 3-8 seconds (per iteration) |
| 5 | File persistence | <0.5 seconds |

**Total Pipeline Time:**
- **Best case (success on attempt 1):** 8-15 seconds
- **Average case (2-3 attempts):** 20-35 seconds
- **Worst case (5 failed attempts):** 50-70 seconds

### Space Complexity

| Component | Memory Usage |
|-----------|--------------|
| Embedder singleton | ~200MB (4 CSV files + embeddings) |
| Clause extraction JSON | ~5-20KB |
| LLM prompt | ~60-100KB |
| SPARQL query | ~1-5KB |
| Results (per query) | Variable (10KB - 10MB) |

**Disk Usage per Run:** 100KB - 2MB (depending on result size)

---

## Configuration Options

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | (required) | OpenAI API authentication |
| `LLM_MODEL` | `gpt-5` | Model for clause extraction & SPARQL generation |

### Command-Line Arguments

```bash
python run_query_pipeline.py "query" [options]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `user_query` | str | (required) | Natural language query |
| `--max-attempts` | int | 5 | Maximum correction iterations |
| `--system-initial` | str | (default prompt) | Inline system prompt for initial generation |
| `--system-error` | str | (default prompt) | Inline system prompt for error correction |
| `--system-initial-file` | path | None | File path for initial system prompt |
| `--system-error-file` | path | None | File path for error correction prompt |

**Example:**
```bash
python run_query_pipeline.py \
    "Find steel manufacturers in Germany" \
    --max-attempts 3 \
    --system-initial-file custom_prompts/sparql_gen.txt
```

---

## Usage Examples

### Example 1: Simple Query (Success on First Attempt)

**Input:**
```bash
python run_query_pipeline.py "Show all manufacturers"
```

**Pipeline Flow:**
1. Extracts clauses: `[{"type": "select", "subject_hint": "manufacturer"}]`
2. Finds IRIs: `[{"label": "Manufacturer", "iri": "...#Manufacturer"}]`
3. Generates SPARQL:
   ```sparql
   SELECT ?m WHERE { ?m rdf:type ex:Manufacturer }
   ```
4. Executes successfully on attempt 1
5. Outputs TSV results

**Files Created:**
- `outputs/run_*/user_query.txt`
- `outputs/run_*/extracted.json`
- `outputs/run_*/attempt_1/sparql_query.txt`
- `outputs/run_*/attempt_1/graphdb_result.txt`
- `query_log.json` (appended)

---

### Example 2: Complex Query with Error Correction

**Input:**
```bash
python run_query_pipeline.py "Find companies with ISO 9001 that produce steel and are located in Germany"
```

**Pipeline Flow:**

**Attempt 1:**
- Generates SPARQL with unbound variable `?cert`
- GraphDB error: "Unbound variable ?cert in SELECT"
- Saves error, builds correction prompt

**Attempt 2:**
- LLM adds `?cert` to WHERE clause
- Generates SPARQL with wrong IRI namespace `iso:9001`
- GraphDB error: "Invalid IRI prefix 'iso'"
- Saves error, builds correction prompt

**Attempt 3:**
- LLM fixes namespace to `ex:ISO9001Certificate`
- Query executes successfully
- Returns matching companies

**Files Created:**
- 3 attempt directories with full history
- Final successful result
- Error log documenting all attempts

---

## Conclusion

This pipeline implements a sophisticated **natural language to SPARQL** translation system with:

✅ **Structured clause extraction** via LLM with strict JSON schemas  
✅ **Semantic IRI resolution** using embedding-based similarity search  
✅ **Multi-strategy matching** (exact, variant, substring, word overlap)  
✅ **Intelligent prompt engineering** with ontology context  
✅ **Multi-level retry logic** for robustness  
✅ **Automated error correction** via LLM-in-the-loop  
✅ **Comprehensive persistence** for debugging and analysis  
✅ **Configurable system prompts** for customization  

The system balances **automation** (minimal user intervention) with **transparency** (all artifacts saved) and **robustness** (graceful degradation on failures).

---

## Future Enhancements

### Potential Improvements

1. **Caching:** Cache successful query transformations for similar inputs
2. **Fine-tuning:** Use `query_log.json` to fine-tune a specialized model
3. **Parallel execution:** Run multiple correction strategies simultaneously
4. **Interactive mode:** Allow user feedback during error correction
5. **Query optimization:** Analyze execution plans, suggest optimizations
6. **Metrics dashboard:** Visualize success rates, error patterns, latency

---

**Document Version:** 1.0  
**Last Updated:** October 15, 2025  
**Generated by:** AI Technical Documentation System

