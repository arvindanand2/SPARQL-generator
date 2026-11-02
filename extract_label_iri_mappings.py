#!/usr/bin/env python3
"""
Extract Label-to-IRI Mappings from Embedding Files

This script processes JSONL embedding files to create a CSV mapping
of labels to their corresponding IRIs.

Features:
- Handles large JSONL files efficiently
- Processes multiple embedding files
- Creates a single CSV with label-IRI mappings
- Removes duplicates and handles encoding properly
"""

import json
import csv
import os
from pathlib import Path
from typing import List, Dict
import argparse

def extract_label_iri_from_jsonl(file_path: str) -> List[Dict[str, str]]:
    """Extract label and IRI pairs from a JSONL file.

    Parameters
    ----------
    file_path : str
        Path to the JSONL export containing label/iri records.

    Returns
    -------
    list[dict[str, str]]
        Sequence of mappings enriched with provenance metadata.
    """
    mappings = []
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Warning: File {file_path} does not exist")
        return mappings

    print(f"Processing: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue

                line_count += 1
                try:
                    data = json.loads(line)

                    # Extract label and IRI if they exist
                    label = data.get('label', '')
                    iri = data.get('iri', '')

                    if label and iri:
                        mappings.append({
                            'label': label,
                            'iri': iri,
                            'type': data.get('type', ''),
                            'id': data.get('id', ''),
                            'source_file': file_path.name
                        })

                except json.JSONDecodeError as e:
                    print(f"JSON decode error on line {line_count} in {file_path}: {e}")
                    continue

                # Progress indicator for large files
                if line_count % 1000 == 0:
                    print(f"  Processed {line_count} lines...")

        print(f"  Found {len(mappings)} label-IRI pairs in {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return mappings

def create_label_iri_csv(embedding_files: List[str], output_file: str = "label_iri_mappings.csv"):
    """Create CSV file mapping labels to IRIs from multiple embedding files."""

    all_mappings = []
    seen_pairs = set()  # To avoid duplicates

    print("Starting label-IRI extraction...")

    for file_path in embedding_files:
        mappings = extract_label_iri_from_jsonl(file_path)

        # Add unique mappings only
        for mapping in mappings:
            pair_key = f"{mapping['label']}|{mapping['iri']}"
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                all_mappings.append(mapping)

    print(f"\nTotal unique label-IRI pairs found: {len(all_mappings)}")

    # Sort by label for better organization
    all_mappings.sort(key=lambda x: x['label'].lower())

    # Write to CSV
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['label', 'iri', 'type', 'id', 'source_file']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for mapping in all_mappings:
                writer.writerow(mapping)

        print(f"\n✅ Successfully created {output_file}")
        print(f"   Contains {len(all_mappings)} label-IRI mappings")

        # Show sample of results
        print(f"\n📋 Sample mappings:")
        for i, mapping in enumerate(all_mappings[:5]):
            print(f"   {i+1}. Label: '{mapping['label']}' -> IRI: '{mapping['iri']}'")

        if len(all_mappings) > 5:
            print(f"   ... and {len(all_mappings) - 5} more mappings")

    except Exception as e:
        print(f"❌ Error writing CSV file: {e}")
        return False

    return True

def main():
    """CLI entrypoint to collate label-IRI pairs into a CSV."""
    parser = argparse.ArgumentParser(description="Extract label-IRI mappings from embedding files")
    parser.add_argument("--output", "-o", default="label_iri_mappings.csv",
                       help="Output CSV file path")
    parser.add_argument("--files", "-f", nargs="+",
                       help="Specific embedding files to process")

    args = parser.parse_args()

    # Default embedding files to process if none specified
    default_embedding_files = [
        "classes_with_embeddings.jsonl",
        "classes_embedding.jsonl",
        "predicates_with_embeddings.jsonl",
        "predicates.jsonl",
        "data/classes_ready.jsonl",
        "data/predicates_ready.jsonl"
    ]

    # Use specified files or default list
    embedding_files = args.files if args.files else default_embedding_files

    # Filter to only existing files
    existing_files = []
    for file_path in embedding_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            print(f"Warning: File not found: {file_path}")

    if not existing_files:
        print("❌ No embedding files found to process")
        return

    print(f"Found {len(existing_files)} embedding files to process:")
    for file_path in existing_files:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  - {file_path} ({file_size:.1f} MB)")

    # Create the CSV mapping
    success = create_label_iri_csv(existing_files, args.output)

    if success:
        print(f"\n🎉 Label-IRI mapping complete! Check {args.output}")
    else:
        print(f"\n❌ Failed to create label-IRI mapping")

if __name__ == "__main__":
    main()
