"""Debug variant of process embedding generation with verbose logging."""

import json
import sys
import traceback

def generate_embedding_text(process_caps_file, output_file):
    """
    Generate embedding text for process capabilities in the specified format.
    """
    try:
        print(f"Reading input file: {process_caps_file}")
        with open(process_caps_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Loaded data with {len(data.get('process_caps', []))} top-level processes")

        embeddings = []

        def traverse_process(process, hierarchy_path=[], category=""):
            """Recursively traverse the process hierarchy and generate embeddings"""
            try:
                current_name = process['name']
                alt_labels = process.get('altLabels', [])

                # Build hierarchy path
                full_hierarchy = hierarchy_path + [current_name]
                hierarchy_str = " > ".join(full_hierarchy)

                # Generate also called section
                also_called = ", ".join(alt_labels) if alt_labels else "None"

                # Generate notes based on process type and keywords
                notes = generate_notes(current_name, alt_labels, hierarchy_path)

                # Create embedding text
                embedding_text = f"""Label: {current_name}
Also called: {also_called}
Hierarchy: {hierarchy_str}
Category: {category}
Notes: {notes}"""

                embeddings.append(embedding_text)

                # Recursively process children
                for child in process.get('children', []):
                    traverse_process(child, full_hierarchy, category)

            except Exception as e:
                print(f"Error processing process {process.get('name', 'Unknown')}: {e}")

        def generate_notes(name, alt_labels, hierarchy):
            """Generate relevant keywords and notes for the process"""
            notes = []

            # Add the process name words as keywords
            name_words = name.lower().replace('-', ' ').split()
            notes.extend(name_words)

            # Add alternative label words
            for alt in alt_labels:
                alt_words = alt.lower().replace('-', ' ').split()
                notes.extend(alt_words)

            # Add hierarchy-based keywords
            for parent in hierarchy:
                parent_words = parent.lower().replace('-', ' ').split()
                notes.extend(parent_words)

            # Add specific manufacturing/process keywords based on context
            manufacturing_keywords = {
                'welding': ['welding', 'joining', 'fusion', 'metal', 'heat', 'fabrication'],
                'machining': ['cutting', 'material removal', 'subtractive', 'tool', 'precision'],
                'casting': ['molten', 'mold', 'metal', 'liquid', 'solidification'],
                'forming': ['deformation', 'shaping', 'plastic', 'pressure', 'bending'],
                'grinding': ['abrasive', 'surface', 'finishing', 'precision', 'smooth'],
                'drilling': ['hole', 'boring', 'cutting', 'machining', 'tool'],
                'milling': ['cutting', 'machining', 'surface', 'tool', 'removal'],
                'turning': ['lathe', 'rotating', 'cutting', 'cylindrical', 'machining'],
                'forging': ['hammering', 'pressing', 'hot', 'shaping', 'metal'],
                'coating': ['surface', 'protection', 'layer', 'finishing', 'treatment'],
                'additive': ['3d printing', 'building', 'layer', 'manufacturing', 'material deposition'],
                'brazing': ['joining', 'filler metal', 'heat', 'bonding', 'assembly'],
                'soldering': ['joining', 'low temperature', 'electronics', 'bonding', 'assembly']
            }

            # Check for relevant keywords based on process name and hierarchy
            name_lower = name.lower()
            hierarchy_lower = ' '.join(hierarchy).lower()
            all_text = (name_lower + ' ' + hierarchy_lower).lower()

            for keyword, related_terms in manufacturing_keywords.items():
                if keyword in all_text:
                    notes.extend(related_terms)

            # Remove duplicates and empty strings, then join
            unique_notes = list(dict.fromkeys([note for note in notes if note.strip()]))
            return ", ".join(unique_notes)

        # Process each top-level category
        for process in data['process_caps']:
            category_name = process['name']
            print(f"Processing category: {category_name}")
            traverse_process(process, [], category_name)

        print(f"Generated {len(embeddings)} embedding texts")

        # Write embeddings to file
        print(f"Writing to output file: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n\n".join(embeddings))

        print(f"Saved to: {output_file}")
        return embeddings

    except Exception as e:
        print(f"Error in generate_embedding_text: {e}")
        traceback.print_exc()
        return []

if __name__ == "__main__":
    try:
        print("Starting process embeddings generation...")

        # Generate embeddings
        embeddings = generate_embedding_text(
            'process_caps.json',
            'process_embeddings.txt'
        )

        if embeddings:
            # Print first few examples
            print("\nFirst 3 examples:")
            print("="*80)
            for i, embedding in enumerate(embeddings[:3]):
                print(f"\nExample {i+1}:")
                print(embedding)
                print("-" * 40)
        else:
            print("No embeddings generated!")

    except Exception as e:
        print(f"Script error: {e}")
        traceback.print_exc()
