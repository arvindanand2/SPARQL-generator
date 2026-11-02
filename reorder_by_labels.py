"""Reorder enhanced embedding files to match the canonical label order."""


def extract_labels_in_order(filepath):
    """Extract all labels from a file in their original order"""
    labels = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('Label:'):
                label = line.replace('Label:', '').strip()
                labels.append(label)
    return labels

def extract_entries_with_labels(filepath):
    """Extract complete entries mapped by their labels"""
    entries = {}
    current_entry = ""
    current_label = None

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip('\n')
        if line.startswith('Label:'):
            # Save previous entry
            if current_label and current_entry.strip():
                entries[current_label] = current_entry.strip()

            # Start new entry
            current_label = line.replace('Label:', '').strip()
            current_entry = line
        elif line.startswith('================================================================================'):
            continue  # Skip separator lines
        else:
            if current_entry:
                current_entry += '\n' + line

    # Save the last entry
    if current_label and current_entry.strip():
        entries[current_label] = current_entry.strip()

    return entries

def reorder_enhanced_to_match_original():
    """Reorder enhanced file to match original file's label order"""
    print("Extracting label order from process_embeddings.txt...")
    original_order = extract_labels_in_order('process_embeddings.txt')

    print("Extracting entries from process_embeddings_enhanced.txt...")
    enhanced_entries = extract_entries_with_labels('process_embeddings_enhanced.txt')

    print(f"Original file has {len(original_order)} labels")
    print(f"Enhanced file has {len(enhanced_entries)} entries")

    # Create backup
    import shutil
    try:
        shutil.copy('process_embeddings_enhanced.txt', 'process_embeddings_enhanced_backup.txt')
        print("✅ Backup created: process_embeddings_enhanced_backup.txt")
    except Exception as e:
        print(f"⚠️ Could not create backup: {e}")

    # Reorder entries to match original sequence
    reordered_entries = []
    matched_count = 0
    missing_labels = []

    for label in original_order:
        if label in enhanced_entries:
            reordered_entries.append(enhanced_entries[label])
            matched_count += 1
        else:
            missing_labels.append(label)

    # Write reordered file
    if reordered_entries:
        separator = "=" * 80
        output_content = f"\n{separator}\n\n".join(reordered_entries)

        with open('process_embeddings_enhanced.txt', 'w', encoding='utf-8') as f:
            f.write(output_content)

        print(f"✅ Reordered {len(reordered_entries)} entries to match original sequence")

    # Verify the reordering
    new_order = extract_labels_in_order('process_embeddings_enhanced.txt')

    print(f"\n📊 REORDERING RESULTS:")
    print(f"Matched and reordered: {matched_count} entries")
    print(f"Missing from enhanced: {len(missing_labels)} labels")

    if missing_labels and len(missing_labels) <= 10:
        print("Missing labels:")
        for label in missing_labels:
            print(f"  - {label}")
    elif len(missing_labels) > 10:
        print(f"First 10 missing labels:")
        for label in missing_labels[:10]:
            print(f"  - {label}")
        print(f"  ... and {len(missing_labels) - 10} more")

    # Verify order alignment
    order_matches = 0
    enhanced_present = [label for label in original_order if label in enhanced_entries]

    for i, (orig, new) in enumerate(zip(enhanced_present, new_order)):
        if orig == new:
            order_matches += 1
        else:
            print(f"❌ Order mismatch at position {i+1}: expected '{orig}', got '{new}'")
            break

    if order_matches == len(new_order):
        print(f"✅ Perfect order alignment: {order_matches}/{len(new_order)} labels in correct sequence")
    else:
        print(f"⚠️ Partial alignment: {order_matches}/{len(new_order)} labels in correct order")

    return matched_count, len(missing_labels)

if __name__ == "__main__":
    matched, missing = reorder_enhanced_to_match_original()
    print(f"\n🎯 FINAL RESULT: {matched} entries reordered, {missing} labels missing from enhanced file")
