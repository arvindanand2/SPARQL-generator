"""Placeholder module for baseline embedding CSV export utilities."""

import csv
import re
input_file = 'process_embeddings.txt'
output_file = 'process_embeddings.csv'

with open(input_file, 'r', encoding='utf-8') as f:
    data = f.read()

# Split by double newlines to get each process block
blocks = re.split(r'\n\n+', data.strip())

rows = []
for block in blocks:
    label = re.search(r'Label: (.+)', block)
    also_called = re.search(r'Also called: (.+)', block)
    hierarchy = re.search(r'Hierarchy: (.+)', block)
    category = re.search(r'Category: (.+)', block)
    rows.append([
        label.group(1) if label else '',
        also_called.group(1) if also_called else '',
        hierarchy.group(1) if hierarchy else '',
        category.group(1) if category else ''
    ])

with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Label', 'Also Called', 'Hierarchy', 'Category'])
    writer.writerows(rows)

print(f"CSV file '{output_file}' created successfully.")

