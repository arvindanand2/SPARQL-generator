"""Compare original and enhanced embedding exports up to the shear forming marker."""


def compare_files_to_shear_forming():
    """Extract labels from original up to Shear Forming and compare with enhanced file"""

    # Read original file and extract labels up to Shear Forming
    original_labels = []
    with open('process_embeddings.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith('Label:'):
                label = line.replace('Label:', '').strip()
                original_labels.append((i+1, label))
                # Stop when we reach Shear Forming
                if 'Shear Forming' in label:
                    break

    # Read enhanced file and extract all labels
    enhanced_labels = set()
    with open('process_embeddings_enhanced.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('Label:'):
                label = line.replace('Label:', '').strip()
                enhanced_labels.add(label)

    # Compare and find missing
    missing_labels = []
    present_labels = []

    print("📋 ALL LABELS UP TO 'SHEAR FORMING' COMPARISON:")
    print("="*80)

    for line_num, label in original_labels:
        if label in enhanced_labels:
            status = "✅ PRESENT"
            present_labels.append(label)
        else:
            status = "❌ MISSING"
            missing_labels.append(label)

        print(f"{line_num:3d}. {status:10} | {label}")

    # Summary
    print("\n" + "="*80)
    print(f"📊 SUMMARY:")
    print(f"Total labels checked: {len(original_labels)}")
    print(f"Present in enhanced: {len(present_labels)}")
    print(f"Missing from enhanced: {len(missing_labels)}")
    print(f"Coverage: {len(present_labels)}/{len(original_labels)} ({len(present_labels)/len(original_labels)*100:.1f}%)")

    if missing_labels:
        print(f"\n❌ MISSING LABELS ({len(missing_labels)}):")
        print("-"*60)
        for i, label in enumerate(missing_labels, 1):
            print(f"{i:2d}. {label}")
    else:
        print("\n🎉 ALL LABELS ARE PRESENT IN ENHANCED FILE!")

    return missing_labels

if __name__ == "__main__":
    missing = compare_files_to_shear_forming()
