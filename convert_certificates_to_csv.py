"""Flatten certificate taxonomy JSON into a CSV-friendly structure."""

import json
import csv


def traverse_certificates(cert, hierarchy_path, category, rows):
    """Depth-first traversal that builds CSV rows for certificate hierarchies.

    Parameters
    ----------
    cert : dict
        Current certificate node.
    hierarchy_path : list[str]
        Sequence capturing ancestor names.
    category : str
        Top-level category label to annotate rows with.
    rows : list[dict]
        Mutable list collecting CSV row dictionaries.
    """
    name = cert.get('name', '').strip()
    altlabels = cert.get('altLabels', [])
    hierarchy = ' > '.join(hierarchy_path + [name])
    rows.append({
        'Label': name,
        'Also Called': ', '.join([a.strip() for a in altlabels if a.strip()]),
        'Hierarchy': hierarchy,
        'Category': category
    })
    for child in cert.get('children', []):
        traverse_certificates(child, hierarchy_path + [name], category, rows)

def main():
    """Load certificate JSON and write ``certificates_enhanced.csv``."""
    with open('certificates.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for cert in data.get('certificates', []):
        traverse_certificates(cert, [], cert.get('name', '').strip(), rows)
    with open('certificates_enhanced.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Label', 'Also Called', 'Hierarchy', 'Category'])
        writer.writeheader()
        writer.writerows(rows)
if __name__ == '__main__':
    main()

