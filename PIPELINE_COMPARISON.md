# Pipeline Comparison: Batch vs Direct API

## Feature Parity Verification

Both `run_query_pipeline_batch_resumable.py` and `run_query_pipeline_resumable.py` now have **complete feature parity**, with the **only difference** being the API call mechanism.

## Shared Features ✅

### 1. State Management
- ✅ `PipelineState` class with save/load functionality
- ✅ Atomic file writes with backup
- ✅ State corruption recovery
- ✅ Resume capability from any stage

### 2. Query Processing
- ✅ Single query mode (`--query`)
- ✅ Batch CSV processing (`--csv`)
- ✅ Query limit support (`--limit`)
- ✅ Stage-based processing (extraction → generation → validation → correction)

### 3. Error Handling
- ✅ Comprehensive error logging
- ✅ Per-attempt error tracking
- ✅ Error compression for LLM context
- ✅ Timeout handling (600 seconds)
- ✅ GraphDB connection testing

### 4. Output Management
- ✅ Per-query directory structure with sanitized names
- ✅ Per-attempt output saving
- ✅ Summary JSON generation
- ✅ Results CSV with all metadata
- ✅ Query logging to `query_log.json`
- ✅ Model name in output directory path

### 5. Command-line Arguments
- ✅ `--query`: Single query
- ✅ `--csv`: CSV file input
- ✅ `--max-attempts`: Max correction attempts (default: 10)
- ✅ `--limit`: Limit number of queries
- ✅ `--resume`: Resume from previous run
- ✅ `--state-dir`: State directory (different defaults)
- ✅ `--outputs-dir`: Outputs directory (different defaults)
- ✅ `--system-initial`: Inline system prompt
- ✅ `--system-error`: Inline error correction prompt
- ✅ `--system-initial-file`: System prompt file
- ✅ `--system-error-file`: Error prompt file

### 6. Helper Functions
- ✅ `compress_error()`: Error message compression
- ✅ `get_embedder()`: Lazy embedder initialization
- ✅ `extract_iris_from_query()`: IRI extraction
- ✅ `build_llm_prompt()`: Prompt building
- ✅ `build_error_correction_prompt()`: Error prompt building
- ✅ `run_query_graph_with_retries()`: GraphDB execution
- ✅ `check_graphdb_connection()`: Connection testing
- ✅ `load_queries_from_csv()`: Multi-encoding CSV loading
- ✅ `save_query_log()`: Query logging
- ✅ `save_query_outputs()`: Query output saving
- ✅ `save_attempt_outputs()`: Per-attempt saving
- ✅ `sanitize_filename()`: Filename sanitization
- ✅ `save_results_to_csv()`: Results CSV generation

### 7. Resume Functionality
- ✅ Skip completed queries on resume
- ✅ Cache extraction results
- ✅ Cache SPARQL generation
- ✅ Cache validation results per attempt
- ✅ Resume from any failed correction attempt

## Key Differences (API Implementation Only)

| Feature | Batch Version | Direct Version |
|---------|---------------|----------------|
| **API Call Method** | OpenAI Batch API | Direct `chat.completions.create()` |
| **Processing Mode** | All queries batched together at each stage | Sequential query processing |
| **Response Time** | Wait for entire batch to complete (minutes-hours) | Immediate response per query (seconds) |
| **State Directory Default** | `pipeline_state` | `pipeline_state_direct` |
| **Output Directory Default** | `outputs_batch` | `outputs_direct` |
| **Extraction Function** | `extract_chunk_batch()` | `extract_chunk_direct()` |
| **SPARQL Generation** | `generate_sparql_batch()` | `get_sparql_from_llm()` |
| **Processing Function** | `process_all_queries_batch()` | `process_all_queries()` + `process_single_query()` |
| **Batch Handler** | `BatchAPIHandler` class (comprehensive) | N/A (uses direct calls) |

## Function Count
- **Batch version**: 19 functions + 1 class (BatchAPIHandler)
- **Direct version**: 20 functions + 1 class (PipelineState)

The direct version has one additional function (`process_single_query()`) to handle individual query processing, while the batch version integrates this into `process_all_queries_batch()`.

## When to Use Each Version

### Use `run_query_pipeline_batch_resumable.py` when:
- Processing large numbers of queries (100+)
- Cost optimization is priority (batch API is 50% cheaper)
- You can wait for results (hours are acceptable)
- You want to minimize rate limit issues

### Use `run_query_pipeline_resumable.py` when:
- Processing small to medium number of queries (<100)
- Need immediate feedback/results
- Want to see progress in real-time
- Debugging or testing queries
- Speed is more important than cost

## Verification Complete ✅

Both implementations have:
- ✅ Identical feature sets
- ✅ Identical error handling
- ✅ Identical state management
- ✅ Identical output formats
- ✅ Identical resume capability
- ✅ Compatible state files (can switch between versions)

**The only difference is the API call mechanism.**

