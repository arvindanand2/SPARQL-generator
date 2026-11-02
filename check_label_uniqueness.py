"""Detect duplicate labels within hierarchical process capability JSON."""

import json
from collections import defaultdict


def collect_labels_and_altlabels(processes, label_counts, altlabel_counts, path=None):
    """Recursively traverse processes to populate label and alt-label registries.

    Parameters
    ----------
    processes : list[dict]
        Sequence of process nodes to examine.
    label_counts : dict[str, list[str]]
        Aggregate of label values to their hierarchy paths.
    altlabel_counts : dict[str, list[str]]
        Aggregate of alt-label values to their hierarchy paths.
    path : list[str] | None, optional
        Breadcrumb trail representing the current hierarchy.
    """
    if path is None:
        path = []
    for proc in processes:
        name = proc.get('name')
        altlabels = proc.get('altLabels', [])
        current_path = path + [name]
        if name:
            label_counts[name].append(' > '.join(current_path))
        for alt in altlabels:
            altlabel_counts[alt].append(' > '.join(current_path))
        for child in proc.get('children', []):
            collect_labels_and_altlabels([child], label_counts, altlabel_counts, current_path)

def main():
    """Load process capability JSON and report duplicate label usage."""
    with open('process_caps.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    processes = data.get('process_caps', [])
    label_counts = defaultdict(list)
    altlabel_counts = defaultdict(list)
    collect_labels_and_altlabels(processes, label_counts, altlabel_counts)
    print('Duplicate Labels:')
    for label, locations in label_counts.items():
        if len(locations) > 1:
            print(f'  "{label}" found at:')
            for loc in locations:
                print(f'    {loc}')
    print('\nDuplicate Also Called (altLabels):')
    for alt, locations in altlabel_counts.items():
        if alt and len(locations) > 1:
            print(f'  "{alt}" found at:')
            for loc in locations:
                print(f'    {loc}')
if __name__ == '__main__':
    main()

