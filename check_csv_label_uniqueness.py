"""Deduplicate alias terms in the enhanced embeddings CSV export."""

import csv
from collections import defaultdict


def main():
    """Remove alias values that duplicate labels or appear multiple times."""
    with open('process_embeddings_enhanced.csv', 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    # Collect all Label values (case-insensitive, trimmed)
    label_set = {row['Label'].strip().lower() for row in reader if row['Label'].strip()}
    # Collect all Also Called terms and count occurrences
    altlabel_counts = defaultdict(int)
    for row in reader:
        for alt in [a.strip().lower() for a in row['Also Called'].split(',') if a.strip()]:
            altlabel_counts[alt] += 1
    # Identify Also Called terms that are duplicated or match any Label
    duplicate_altlabels = {alt for alt, count in altlabel_counts.items() if count > 1 or alt in label_set}
    # Remove only the duplicated/matching Also Called terms from each row
    for row in reader:
        altlabel_list = [alt.strip() for alt in row['Also Called'].split(',') if alt.strip()]
        filtered_altlabels = [alt for alt in altlabel_list if alt.lower() not in duplicate_altlabels]
        row['Also Called'] = ', '.join(filtered_altlabels)
    with open('process_embeddings_enhanced.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=reader[0].keys())
        writer.writeheader()
        writer.writerows(reader)
if __name__ == '__main__':
    main()
