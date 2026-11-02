"""Identify duplicate labels and aliases within the enhanced embeddings CSV."""

import csv
from collections import defaultdict


def main():
    """Scan the enhanced CSV export for duplicate label or alias values."""
    with open('process_embeddings_enhanced.csv', 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    label_counts = defaultdict(list)
    altlabel_counts = defaultdict(list)
    for idx, row in enumerate(reader, start=2):  # start=2 to account for header
        label = row['Label'].strip().lower()
        label_counts[label].append(idx)
        altlabels = [alt.strip().lower() for alt in row['Also Called'].split(',') if alt.strip()]
        for alt in altlabels:
            altlabel_counts[alt].append(idx)
    print('Duplicate Labels:')
    for label, rows in label_counts.items():
        if len(rows) > 1:
            print(f'  "{label}" found in rows: {rows}')
    print('\nDuplicate Also Called values:')
    for alt, rows in altlabel_counts.items():
        if len(rows) > 1:
            print(f'  "{alt}" found in rows: {rows}')
if __name__ == '__main__':
    main()

