"""Convert material capability taxonomy into a flattened CSV export."""

import json
import csv


def traverse_materials(mat, hierarchy_path, category, rows):
    """Depth-first traversal constructing CSV rows for material capabilities.

    Parameters
    ----------
    mat : dict
        Current material capability node.
    hierarchy_path : list[str]
        Breadcrumb list for the ancestral names.
    category : str
        Top-level material category name.
    rows : list[dict]
        Output accumulator for CSV row dictionaries.
    """
    name = mat.get('name', '').strip()
    altlabels = mat.get('altLabels', [])
    hierarchy = ' > '.join(hierarchy_path + [name])
    rows.append({
        'Label': name,
        'Also Called': ', '.join([a.strip() for a in altlabels if a.strip()]),
        'Hierarchy': hierarchy,
        'Category': category
    })
    for child in mat.get('children', []):
        traverse_materials(child, hierarchy_path + [name], category, rows)

def main():
    """Generate ``material_caps_enhanced.csv`` from the JSON taxonomy."""
    with open('material_caps.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for mat in data.get('material_caps', []):
        traverse_materials(mat, [], mat.get('name', '').strip(), rows)
    with open('material_caps_enhanced.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Label', 'Also Called', 'Hierarchy', 'Category'])
        writer.writeheader()
        writer.writerows(rows)
if __name__ == '__main__':
    main()

