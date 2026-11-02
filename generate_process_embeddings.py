"""Generate human-readable embedding text blocks from process capability JSON."""

import json

def generate_embedding_text(process_caps_file, output_file):
    """
    Generate embedding text for process capabilities in the specified format.

    Format:
    Label: [Process Name]
    Also called: [Alternative Labels]
    Hierarchy: [Parent > Child > ... > Current Process]
    Category: [Top-level Category]
    """

    with open(process_caps_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    embeddings = []

    def traverse_process(process, hierarchy_path=[], category=""):
        """Recursively traverse the process hierarchy and generate embeddings"""

        current_name = process['name']
        alt_labels = process.get('altLabels', [])

        # Build hierarchy path
        full_hierarchy = hierarchy_path + [current_name]
        hierarchy_str = " > ".join(full_hierarchy)

        # Generate also called section
        also_called = ", ".join(alt_labels) if alt_labels else "None"

        # Create embedding text
        embedding_text = f"""Label: {current_name}
Also called: {also_called}
Hierarchy: {hierarchy_str}
Category: {category}"""

        embeddings.append(embedding_text)

        # Recursively process children
        for child in process.get('children', []):
            traverse_process(child, full_hierarchy, category)

    # Process each top-level category
    for process in data['process_caps']:
        category_name = process['name']
        traverse_process(process, [], category_name)

    # Write embeddings to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n" + "="*80 + "\n\n".join(embeddings))

    print(f"Generated {len(embeddings)} embedding texts")
    print(f"Saved to: {output_file}")

    return embeddings

if __name__ == "__main__":
    # Generate embeddings
    embeddings = generate_embedding_text(
        'process_caps.json',
        'process_embeddings.txt'
    )

    # Print first few examples
    print("\nFirst 3 examples:")
    print("="*80)
    for i, embedding in enumerate(embeddings[:3]):
        print(f"\nExample {i+1}:")
        print(embedding)
        print("-" * 40)
