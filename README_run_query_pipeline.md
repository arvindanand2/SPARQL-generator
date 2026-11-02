# run_query_pipeline.py

## Overview

`run_query_pipeline.py` is a Python script that automates the process of:
- Extracting relevant clauses and IRIs from a user query
- Building a prompt for a Large Language Model (LLM) to generate a SPARQL query
- Executing the generated SPARQL query against a GraphDB endpoint
- Handling errors and retrying with LLM-based error correction if needed
- Saving all relevant outputs and logs for each run

## Requirements

- Python 3.8+
- Required Python packages:
  - openai
  - python-dotenv
  - SPARQLWrapper
- A running GraphDB instance accessible at the configured endpoint
- Environment variable `OPENAI_API_KEY` set (can be loaded from a `.env` file)
- Supporting files:
  - `label_iri_mappings.csv`
  - Enhanced CSV files for embedding (e.g., `industries_enhanced.csv`, etc.)
  - Prompt templates: `sparql_prompt_template_new.txt`, `sparql_error_correction_prompt.txt`
  - `reference_turtle.txt`

## Usage

```bash
python run_query_pipeline.py "<your natural language query>"
```

Example:
```bash
python run_query_pipeline.py "List all certificates for steel manufacturing processes."
```

## How It Works

1. **Clause Extraction:**
   - Extracts and compresses relevant clauses from the user query.
   - Finds relevant IRIs using enhanced CSV embeddings.
2. **Prompt Construction:**
   - Builds a detailed prompt for the LLM, including extracted clauses, IRIs, and reference Turtle data.
3. **SPARQL Generation:**
   - Sends the prompt to the LLM (e.g., OpenAI GPT) to generate a SPARQL query.
4. **Query Execution:**
   - Runs the SPARQL query against GraphDB using `query_Graph.py`.
   - Results are saved as a JSON file.
5. **Error Handling:**
   - If the query fails, the error is logged and a new prompt is built for the LLM to correct the SPARQL.
   - Retries up to 5 times.
6. **Output:**
   - All run artifacts (inputs, prompts, queries, results, errors) are saved in a timestamped directory under `outputs/`.

## Output Files

Each run creates a directory in `outputs/` containing:
- `user_query.txt`: The original user query
- `extracted.json`: Extracted and compressed clauses
- `llm_prompt.txt`: The prompt sent to the LLM
- `sparql_query.txt`: The generated SPARQL query
- `graphdb_result.txt`: Path to the JSON results file
- `error_log.txt`: Any error messages (if applicable)
- `model.txt`: The LLM model used (if specified)
- Per-attempt subfolders with detailed logs

## Customization
- You can modify the prompt templates or reference Turtle data to suit your ontology or use case.
- The script can be adapted to use different LLMs or endpoints by changing the relevant sections.

## Troubleshooting
- Ensure all required files and environment variables are present.
- Check `error_log.txt` in the output directory for details if a run fails.
- Make sure your GraphDB endpoint is running and accessible.

## License
This project is for research and educational purposes.

