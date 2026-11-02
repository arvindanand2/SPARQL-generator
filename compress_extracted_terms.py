"""Utilities for trimming verbose extraction payloads into compact summaries."""

import json
import sys
from pathlib import Path

def compress_extracted_terms(extracted_data):
    """Convert extracted terms into a compact, LLM-friendly format.

    Parameters
    ----------
    extracted_data : dict
        Raw output produced by clause/term extraction routines.

    Returns
    -------
    dict
        Sanitized structure retaining only high-signal fields and values.
    """

    compressed = {}

    # Handle term arrays - compress to essential info only
    term_fields = [
        'company_name_terms', 'process_capabilities_terms', 'material_capabilities_terms',
        'industries_terms', 'certificates_terms', 'ownership_status_terms',
        'naics_terms', 'business_description_terms', 'employee_count_terms',
        'year_established_terms', 'web_address_terms'
    ]

    for field in term_fields:
        if field in extracted_data and extracted_data[field]:
            items = extracted_data[field]
            if items:
                compressed_items = []
                for item in items:
                    # Keep only essential fields
                    essential = {}
                    if 'text' in item and item['text']:
                        essential['text'] = item['text']
                    if 'confidence' in item and item['confidence']:
                        essential['conf'] = round(item['confidence'], 2)
                    if 'aliases' in item and item['aliases']:
                        essential['aliases'] = item['aliases']
                    if essential:
                        compressed_items.append(essential)

                if compressed_items:
                    # Use shorter field names
                    short_name = field.replace('_terms', '').replace('_capabilities', '_caps')
                    compressed[short_name] = compressed_items

    # Handle special fields with different structures

    # Product field
    if 'product' in extracted_data and extracted_data['product']:
        products = []
        for item in extracted_data['product']:
            product = {}
            if 'text' in item and item['text']:
                product['text'] = item['text']
            if 'confidence' in item and item['confidence']:
                product['conf'] = round(item['confidence'], 2)
            if product:
                products.append(product)
        if products:
            compressed['product'] = products

    # Location terms (special handling for address structure)
    if 'location_terms' in extracted_data and extracted_data['location_terms']:
        locations = []
        for item in extracted_data['location_terms']:
            loc = {}
            if 'city' in item and item['city']:
                loc['city'] = item['city']
            if 'state' in item and item['state']:
                loc['state'] = item['state']
            if 'text_address' in item and item['text_address']:
                loc['address'] = item['text_address']
            if 'confidence' in item and item['confidence']:
                loc['conf'] = round(item['confidence'], 2)
            if loc:
                locations.append(loc)
        if locations:
            compressed['locations'] = locations

    # Employee count filter (special handling for constraints)
    if 'employee_count_filter' in extracted_data and extracted_data['employee_count_filter']:
        filter_data = extracted_data['employee_count_filter']
        emp_filter = {}
        if 'op' in filter_data and filter_data['op']:
            emp_filter['op'] = filter_data['op']
        if 'value' in filter_data and filter_data['value']:
            emp_filter['val'] = filter_data['value']
        if 'unit' in filter_data and filter_data['unit']:
            emp_filter['unit'] = filter_data['unit']
        if 'confidence' in filter_data and filter_data['confidence']:
            emp_filter['conf'] = round(filter_data['confidence'], 2)
        if emp_filter:
            compressed['employee_filter'] = emp_filter

    # Add other useful fields if present
    for field in ['query_type', 'intent', 'query_intent']:
        if field in extracted_data and extracted_data[field]:
            compressed[field] = extracted_data[field]

    # Only include warnings if they exist and are meaningful
    if 'extraction_warnings' in extracted_data and extracted_data['extraction_warnings']:
        if len(extracted_data['extraction_warnings']) > 0:
            compressed['warnings'] = len(extracted_data['extraction_warnings'])  # Just show count

    return compressed

def main():
    """Entry point that handles IO concerns and prints the compact JSON payload."""
    try:
        # Read from stdin or file with encoding detection
        if len(sys.argv) > 1:
            input_file = Path(sys.argv[1])

            # Try different encodings to handle encoding issues
            encodings = ['utf-8-sig', 'utf-8', 'utf-16', 'utf-16le', 'utf-16be', 'latin1']
            data = None

            for encoding in encodings:
                try:
                    with open(input_file, 'r', encoding=encoding) as f:
                        content = f.read()
                        # Clean up any null bytes or BOM issues
                        content = content.replace('\x00', '').strip()
                        if content.startswith('\ufeff'):  # Remove BOM
                            content = content[1:]
                        data = json.loads(content)
                        break
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue

            if data is None:
                raise ValueError("Could not decode file with any supported encoding")

        else:
            # Read from stdin
            content = sys.stdin.read()
            content = content.replace('\x00', '').strip()
            if content.startswith('\ufeff'):  # Remove BOM
                content = content[1:]
            data = json.loads(content)

        # Compress the data
        compressed = compress_extracted_terms(data)

        # Output compressed JSON
        print(json.dumps(compressed, separators=(',', ':')))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
