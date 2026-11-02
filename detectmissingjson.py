"""Locate null entries in JSON documents and report normalized paths."""

import json
import sys
import re

def normalize_path(path):
    """Replace list indices in a JSON path with wildcards for deduplication."""
    # Replace [number] with []
    return re.sub(r'\[\d+\]', '[]', path)

def find_missing(data, path=""):
    """Recursively discover ``None`` values and return their dotted paths."""
    missing = []
    if data is None:
        missing.append(path)
    elif isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            if value is None:
                missing.append(new_path)
            else:
                missing.extend(find_missing(value, new_path))
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            new_path = f"{path}[{idx}]"
            if item is None:
                missing.append(new_path)
            else:
                missing.extend(find_missing(item, new_path))
    return missing

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detectmissingjson.py <file.json>")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
    missing = find_missing(data)
    normalized_missing = set(normalize_path(m) for m in missing)
    print("Missing/null values found at:")
    for m in normalized_missing:
        print(m)
