"""Invoke OpenAI to extract clause chunks according to the structured schema."""

import sys
from openai import OpenAI
import json, os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the clause chunk extraction JSON Schema
with open("clause_chunk_schema.json") as f:
    clause_chunk_schema = json.load(f)

# Load the system prompt from clause_chunker_prompt.json
with open("clause_chunker_prompt_new.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

if len(sys.argv) > 1:
    user_query = sys.argv[1]

model_name = os.getenv("LLM_MODEL", "gpt-5")

resp = client.chat.completions.create(
    model=model_name,                 # model now loaded from .env
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": "Extract clauses from this query into the schema. Query:\n" + user_query}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "ExtractedClauses",
            "strict": True,          # enforce exact schema conformance
            "schema": clause_chunk_schema
        }
    },
)

extracted = json.loads(resp.choices[0].message.content)
print(json.dumps(extracted, indent=2))
