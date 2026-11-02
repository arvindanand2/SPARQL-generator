"""Helpers to reorder enhanced embedding files to match the canonical sequence."""

import re

def extract_labels_and_entries(filepath):
    """Extract labels and their complete entries from embeddings file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split entries by the separator or by "Label:" at the beginning of lines
    entries = []
    current_entry = ""
    lines = content.split('\n')

    for line in lines:
        if line.startswith('Label:'):
            if current_entry.strip():
                entries.append(current_entry.strip())
            current_entry = line
        elif line.startswith('================================================================================'):
            continue  # Skip separator lines
        else:
            if current_entry:
                current_entry += '\n' + line

    # Add the last entry
    if current_entry.strip():
        entries.append(current_entry.strip())

    # Extract labels and create label->entry mapping
    label_to_entry = {}
    labels_order = []

    for entry in entries:
        match = re.search(r'^Label:\s*(.+)$', entry, re.MULTILINE)
        if match:
            label = match.group(1).strip()
            labels_order.append(label)
            label_to_entry[label] = entry

    return labels_order, label_to_entry

def reorder_enhanced_file():
    """Reorder process_embeddings_enhanced.txt to match process_embeddings.txt order"""

    # Extract order from original file
    print("Reading original process_embeddings.txt...")
    original_labels, _ = extract_labels_and_entries('process_embeddings.txt')

    # Extract entries from enhanced file
    print("Reading process_embeddings_enhanced.txt...")
    enhanced_labels, enhanced_entries = extract_labels_and_entries('process_embeddings_enhanced.txt')

    print(f"Original file has {len(original_labels)} entries")
    print(f"Enhanced file has {len(enhanced_labels)} entries")

    # Check for missing or extra labels
    original_set = set(original_labels)
    enhanced_set = set(enhanced_entries.keys())

    missing_in_enhanced = original_set - enhanced_set
    extra_in_enhanced = enhanced_set - original_set

    if missing_in_enhanced:
        print(f"Warning: {len(missing_in_enhanced)} labels missing in enhanced file:")
        for label in list(missing_in_enhanced)[:5]:  # Show first 5
            print(f"  - {label}")
        if len(missing_in_enhanced) > 5:
            print(f"  ... and {len(missing_in_enhanced) - 5} more")

    if extra_in_enhanced:
        print(f"Warning: {len(extra_in_enhanced)} extra labels in enhanced file:")
        for label in list(extra_in_enhanced)[:5]:  # Show first 5
            print(f"  - {label}")
        if len(extra_in_enhanced) > 5:
            print(f"  ... and {len(extra_in_enhanced) - 5} more")

    # Reorder enhanced entries to match original order
    reordered_entries = []
    found_count = 0

    for label in original_labels:
        if label in enhanced_entries:
            reordered_entries.append(enhanced_entries[label])
            found_count += 1
        else:
            print(f"Missing entry for label: {label}")

    print(f"Successfully matched {found_count} entries")

    # Write reordered file
    separator = "=" * 80
    output_content = f"\n{separator}\n\n".join(reordered_entries)

    # Backup original enhanced file
    import shutil
    shutil.copy('process_embeddings_enhanced.txt', 'process_embeddings_enhanced_backup.txt')
    print("Created backup: process_embeddings_enhanced_backup.txt")

    # Write reordered content
    with open('process_embeddings_enhanced.txt', 'w', encoding='utf-8') as f:
        f.write(output_content)

    print("Successfully reordered process_embeddings_enhanced.txt to match process_embeddings.txt")

    # Verify the reordering
    new_labels, _ = extract_labels_and_entries('process_embeddings_enhanced.txt')

    matches = 0
    for i, (orig, new) in enumerate(zip(original_labels, new_labels)):
        if orig == new:
            matches += 1
        else:
            print(f"Mismatch at position {i}: '{orig}' vs '{new}'")
            break

    print(f"Verification: {matches}/{len(original_labels)} labels in correct order")

    return matches == len(original_labels)

if __name__ == "__main__":
    print("Starting reordering process...")
    success = reorder_enhanced_file()
    if success:
        print("✅ Reordering completed successfully!")
    else:
        print("❌ Reordering completed with some issues - check the output above")
