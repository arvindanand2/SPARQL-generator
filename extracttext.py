"""Run schema-guided term extraction against a fixed test query."""

from openai import OpenAI
import json, os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the unified extraction JSON Schema you’re using for the whole object
with open("extraction_schema.json") as f:
    extraction_schema = json.load(f)

# Load the system prompt from extractprompt.txt
with open("extractprompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

user_query = "Where can i find gloves which are made for medical grade in savannah georgia which has more than 250 employees"

resp = client.chat.completions.create(
    model="gpt-5",                 # upgraded to GPT-5 for structured outputs
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": "Extract terms from this query into the schema. Query:\n" + user_query}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "ExtractedTerms",
            "strict": True,          # enforce exact schema conformance
            "schema": extraction_schema
        }
    },
)

extracted = json.loads(resp.choices[0].message.content)
print(json.dumps(extracted, indent=2))
