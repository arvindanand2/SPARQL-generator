# SPARQL-generator

Code and benchmark supplement for *ARGUS: Text-to-SPARQL Natural Language Interfaces with Embedding-Guided Grounding and Self-Healing Generation* (IEEE ICTAI 2026).

## Repository layout

- Entry points: `run_query_pipeline.py` (single query, writes `outputs/`), `run_query_pipeline_batch_resumable.py` (OpenAI Batch API, resumable, writes `outputs_batch/`), `run_query_pipeline_resumable.py` (direct API, resumable, writes `outputs_direct/`). `query_Graph.py` executes SPARQL.
- Prompts: `clause_chunker_prompt_new.txt` with `clause_chunk_schema.json` (clause extraction), `sparql_prompt_template_new.txt` (generation), `sparql_error_correction_prompt.txt` (self-healing); `sparql_prompt_template.txt` is used by the demos.
- Grounding: `extract_chunk.py`, `compress_chunk.py`, `enhanced_csv_embedder.py`, `permanent_embedding_storage.py`, with vocabularies `*_enhanced*.csv` and `label_iri_mappings.csv`.
- Pre-computed embeddings: `embedding_cache/`, `permanent_embeddings/`, `artifacts/`, `data/`, `*_with_embeddings.jsonl`; builders `embed_ontology.py`, `embed_classes.py`, `embed_predicates.py`, `extract_label_iri_mappings.py`.
- Benchmark: `benchmark_questions_master.csv`. Reference schema exemplar (Turtle): `reference_turtle.txt`.
- Execution logs: `outputs_batch/run_20251019_000123_gpt-5/` is the archived GPT-5 run (its state is in `pipeline_state/`); GPT-5 Mini runs are `outputs_batch/run_20251017_191256_gpt-5-mini/` and `outputs_batch/run_20251018_193510_gpt-5-mini/`; other runs are in `outputs_batch/`, `outputs_direct/`, `outputs/`, `outputs_original_test/`; Batch API requests and responses are in `batch_logs/`.
- Demos: `demo/`. Tests: `test_*.py`.
- Data-preparation utilities: `TTL generation*.py`, `shacl_to_llm.py`, `convert_*_to_csv.py`, `process_embeddings*_to_csv.py`, `generate_process_embeddings*.py`, `reorder_*.py`, `order_and_deduplicate_embeddings.py`, `check_*.py`, `fillmissingjson.py`, `detectmissingjson.py`, `extracttext.py`; embedding retrieval benchmark: `embedding_benchmark.py`.

## Data

The full SUDOKN knowledge graph is not redistributed here. Experiments used a historical SUDOKN snapshot; the current graph is accessible via the public SPARQL endpoint https://apps.okn.us/sudokn/sparql.

Saved execution logs retain questions, generated SPARQL, errors, attempts, and status; query result rows are redacted.

## Running

Install the main dependencies with `pip install -r requirements.txt`. See `README_run_query_pipeline.md` and `demo/README_DEMOS.md`. Queries are sent to the SPARQL endpoint configured in `query_Graph.py` (`endpoint_url`).

API keys are read from a local `.env` file (not included).
