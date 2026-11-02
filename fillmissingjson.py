"""Backfill missing JSON fields with defaults and report null slots."""

import json
import sys
import re
from copy import deepcopy

DEFAULTS = {
    '[].is_manufacturer': True,
    '[].is_contract_manufacturer': True,
    '[].industries.results': ["Renewable", "Energy", "Utilities"],
    '[].process_caps.results': ["Machining", "Fabricating"],
    '[].material_caps.results': ["Bronze", "Iron", "Stainless Steel", "Steel"],
    '[].num_employees': 100,
    '[].is_product_manufacturer': True,
    '[].addresses[].line_1': "10255 Beech Avenue",
    '[].addresses[].city': "Fontana",
    '[].addresses[].county': "San Bernardino",
    '[].addresses[].state': "California",
    '[].addresses[].zip': 11111,
    '[].addresses[].latitude': 100.0677902,
    '[].addresses[].longitude': -100.4708951,
    '[].addresses[].phone_num': "(111) 111-1111",
    '[].addresses[].fax_num': "(222) 222-2222",
    '[].addresses[].city_geoid': "0000000",
    '[].addresses[].state_geoid': "00",
    '[].business_desc': "Screenmobile Llc is primarily engaged in manufacturing fabricated wood millwork, including wood millwork covered with materials such as metal and plastics. Planing mills primarily engaged in producing millwork are included in this industry.",
    '[].naics.code': "332321",
    '[].naics.desc': "naics description",
    '[].certificates.results': ["ISO 9001", "ISO 9001"],
    '[].business_statuses.results': ["Active"]
}

def normalize_path(path):
    """Collapse numeric list indices to wildcards for consistent path matching."""
    return re.sub(r'\[\d+\]', '[]', path)

def fill_missing(data, path=""):
    """Recursively apply DEFAULTS to missing nodes in the JSON-like structure."""
    if isinstance(data, dict):
        for key in list(data.keys()):
            new_path = f"{path}.{key}" if path else key
            norm_path = normalize_path(new_path)
            if data[key] is None:
                if norm_path in DEFAULTS:
                    data[key] = deepcopy(DEFAULTS[norm_path])
            elif isinstance(data[key], (dict, list)):
                fill_missing(data[key], new_path)
        for def_path in DEFAULTS:
            if def_path.startswith(normalize_path(path)):
                key = def_path.split('.')[-1]
                norm_key_path = normalize_path(f"{path}.{key}" if path else key)
                if key not in data and norm_key_path == def_path:
                    data[key] = deepcopy(DEFAULTS[def_path])
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            new_path = f"{path}[{idx}]"
            fill_missing(item, new_path)
        if path:
            norm_path = normalize_path(path)
            for def_path in DEFAULTS:
                if def_path.startswith(norm_path):
                    subfield = def_path.split('.')[-1]
                    for item in data:
                        if isinstance(item, dict) and subfield not in item:
                            item[subfield] = deepcopy(DEFAULTS[def_path])

def find_missing(data, path=""):
    """Return dotted paths for null values within the JSON-like structure."""
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
    if len(sys.argv) < 3:
        print("Usage: python fillmissingjson.py <input.json> <output.json>")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
    fill_missing(data)
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Missing/null values filled and saved to {sys.argv[2]}.")

    missing = find_missing(data)
    if missing:
        print("Still missing/null values at:")
        for m in missing:
            print(m)
    else:
        print("No missing/null values remain.")
