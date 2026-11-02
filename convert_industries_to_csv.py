"""Translate industry taxonomy JSON into rows for CSV exports."""

import json
import csv


def traverse_industries(ind, hierarchy_path, category, rows):
    """Depth-first traversal that collects industry hierarchy rows.

    Parameters
    ----------
    ind : dict
        Current industry node being visited.
    hierarchy_path : list[str]
        Breadcrumb trail for the current location within the tree.
    category : str
        Top-level industry category name.
    rows : list[dict]
        Mutable sequence being populated with CSV rows.
    """
    name = ind.get('name', '').strip()
    altlabels = ind.get('altLabels', [])
    hierarchy = ' > '.join(hierarchy_path + [name])
    rows.append({
        'Label': name,
        'Also Called': ', '.join([a.strip() for a in altlabels if a.strip()]),
        'Hierarchy': hierarchy,
        'Category': category
    })
    for child in ind.get('children', []):
        traverse_industries(child, hierarchy_path + [name], category, rows)

def main():
    """Emit ``industries_enhanced.csv`` from the hierarchical JSON export."""
    with open('industries.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for ind in data.get('industries', []):
        traverse_industries(ind, [], ind.get('name', '').strip(), rows)
    with open('industries_enhanced.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Label', 'Also Called', 'Hierarchy', 'Category'])
        writer.writeheader()
        writer.writerows(rows)
if __name__ == '__main__':
    main()

