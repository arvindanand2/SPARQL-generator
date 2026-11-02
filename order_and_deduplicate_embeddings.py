"""Reorder and deduplicate enhanced embedding blocks to match canonical order."""

import re
from collections import OrderedDict

# Helper to parse blocks from text

def parse_blocks(text, with_def=False):
    """Parse embedding text blocks into an ordered mapping keyed by label.

    Parameters
    ----------
    text : str
        Raw text containing repeated embedding sections.
    with_def : bool, default False
        Whether to extract ``Definition`` fields when present.

    Returns
    -------
    collections.OrderedDict
        Mapping of label to dictionary of extracted fields.
    """
    blocks = re.split(r'\n\n+', text.strip())
    result = OrderedDict()
    for block in blocks:
        label_match = re.search(r'Label: (.+)', block)
        if not label_match:
            continue
        label = label_match.group(1).strip()
        also_called = re.search(r'Also called: (.+)', block)
        hierarchy = re.search(r'Hierarchy: (.+)', block)
        category = re.search(r'Category: (.+)', block)
        definition = re.search(r'Definition: (.+)', block) if with_def else None
        result[label] = {
            'Label': label,
            'Also called': also_called.group(1).strip() if also_called else '',
            'Hierarchy': hierarchy.group(1).strip() if hierarchy else '',
            'Category': category.group(1).strip() if category else '',
            'Definition': definition.group(1).strip() if definition else ''
        }
    return result

# Read both files
with open('process_embeddings_enhanced.txt', 'r', encoding='utf-8') as f:
    enhanced_text = f.read()
with open('process_embeddings_enhanced_with_def.txt', 'r', encoding='utf-8') as f:
    enhanced_def_text = f.read()

# Parse blocks
enhanced_blocks = parse_blocks(enhanced_text, with_def=False)
enhanced_def_blocks = parse_blocks(enhanced_def_text, with_def=True)

# Remove duplicates in enhanced_def_blocks (keep first occurrence)
seen = set()
deduped_def_blocks = OrderedDict()
for label, block in enhanced_def_blocks.items():
    if label not in seen:
        deduped_def_blocks[label] = block
        seen.add(label)

# Reorder deduped_def_blocks to match enhanced_blocks order
ordered_def_blocks = OrderedDict()
missing_labels = []
for label in enhanced_blocks:
    if label in deduped_def_blocks:
        ordered_def_blocks[label] = deduped_def_blocks[label]
    else:
        missing_labels.append(label)

# Save ordered, deduplicated blocks to new file
with open('process_embeddings_enhanced_with_def_ordered.txt', 'w', encoding='utf-8') as f:
    for block in ordered_def_blocks.values():
        f.write(f"Label: {block['Label']}\n")
        f.write(f"Also called: {block['Also called']}\n")
        f.write(f"Hierarchy: {block['Hierarchy']}\n")
        f.write(f"Category: {block['Category']}\n")
        if block['Definition']:
            f.write(f"Definition: {block['Definition']}\n")
        f.write("\n")

# Save missing labels' blocks from enhanced_blocks to missing_labels.txt
with open('missing_labels.txt', 'w', encoding='utf-8') as f:
    for label in missing_labels:
        block = enhanced_blocks[label]
        f.write(f"Label: {block['Label']}\n")
        f.write(f"Also called: {block['Also called']}\n")
        f.write(f"Hierarchy: {block['Hierarchy']}\n")
        f.write(f"Category: {block['Category']}\n")
        f.write("\n")

print(f"Ordered and deduplicated file saved as process_embeddings_enhanced_with_def_ordered.txt.")
print(f"Missing labels saved in missing_labels.txt.")

