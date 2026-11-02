"""Quick inspection utility for predicate embedding JSONL artifacts."""

from pathlib import Path
import json
p = Path('predicates_with_embeddings.jsonl')
if not p.exists():
    print('MISSING')
    raise SystemExit(1)
raw = p.read_bytes()
if not raw:
    print('EMPTY')
    raise SystemExit(0)
lines = [l for l in raw.splitlines() if l.strip()]
print('LINES', len(lines))
# show first line length and a safe preview
first = lines[0]
try:
    obj = json.loads(first)
    print('FIRST_KEYS', list(obj.keys())[:10])
    # print lengths: embedding vector length if present
    if 'embedding' in obj:
        print('EMBED_LEN', len(obj['embedding']))
    if 'embedding_text' in obj:
        print('EMBED_TEXT_SNIPPET', obj['embedding_text'][:200])
except Exception as e:
    print('FIRST_RAW_REPR', repr(first[:400]))
    print('PARSE_ERROR', e)

