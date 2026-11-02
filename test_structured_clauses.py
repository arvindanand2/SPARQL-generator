#!/usr/bin/env python3
"""
Extract subjects and objects from structured clauses and find their corresponding classes/IRIs
"""

import json
from enhanced_csv_embedder import EnhancedCSVEmbedder

def extract_subjects_and_objects(clauses_data):
    """Extract subject and object hints from structured clauses"""

    subjects = set()
    objects = set()

    print("🔍 Extracting subjects and objects from clauses...")

    for i, clause in enumerate(clauses_data.get('clauses', [])):
        subject_hint = clause.get('subject_hint', '')
        object_hint = clause.get('object_hint', '')
        relation_hint = clause.get('relation_hint', '')
        surface = clause.get('surface', '')

        print(f"\n  📋 Clause {i+1}: '{surface}'")

        if subject_hint:
            subjects.add(subject_hint)
            print(f"    ➜ Subject: {subject_hint}")

        if object_hint:
            objects.add(object_hint)
            print(f"    ➜ Object: {object_hint}")

        if relation_hint:
            print(f"    ➜ Relation: {relation_hint}")

    return list(subjects), list(objects)

def find_classes_for_entities(embedder, entities, entity_type="Entity"):
    """Find classes/IRIs for extracted subjects and objects"""

    print(f"\n🔎 Finding classes for {len(entities)} {entity_type.lower()}s...")

    results = {}

    for entity in entities:
        print(f"\n  🎯 Searching for: '{entity}'")

        # Search for the entity in the embeddings
        search_results = embedder.search_closest_labels(entity, top_k=5)

        if search_results:
            best_match = search_results[0]
            results[entity] = {
                'best_match': {
                    'label': best_match.label,
                    'iri': best_match.iri,
                    'score': best_match.score,
                    'category': best_match.category,
                    'definition': best_match.definition
                },
                'alternatives': [
                    {
                        'label': result.label,
                        'iri': result.iri,
                        'score': result.score,
                        'category': result.category
                    }
                    for result in search_results[1:3]  # Top 3 alternatives
                ]
            }

            print(f"    ✅ Best match: {best_match.label} (Score: {best_match.score:.4f})")
            print(f"    🔗 IRI: {best_match.iri}")
        else:
            results[entity] = {'best_match': None, 'alternatives': []}
            print(f"    ❌ No matches found")

    return results

def test_clause_extraction():
    """Main function to extract and find classes for subjects and objects"""

    # Your structured clause data
    clauses_data = {
        "clauses": [
            {
                "type": "spo",
                "surface": "find gloves",
                "subject_hint": "company",
                "relation_hint": "produce",
                "object_hint": "gloves",
                "confidence": 0.78
            },
            {
                "type": "spo",
                "surface": "which are made for medical grade",
                "subject_hint": "company",
                "relation_hint": "made for",
                "object_hint": "medical grade",
                "confidence": 0.85
            },
            {
                "type": "spo",
                "surface": "in Savannah Georgia",
                "subject_hint": "company",
                "relation_hint": "located in",
                "object_hint": "Savannah Georgia",
                "confidence": 0.92
            },
            {
                "type": "filter",
                "surface": "which has more than 250 employees",
                "subject_hint": "company",
                "relation_hint": "employee count",
                "object_hint": None,
                "confidence": 0.82
            },
            {
                "type": "spo",
                "surface": "iso 9000 ceritified",
                "subject_hint": "company",
                "relation_hint": "have certification",
                "object_hint": "ISO 9000",
                "confidence": 0.86
            },
            {
                "type": "spo",
                "surface": "manufacture chips",
                "subject_hint": "company",
                "relation_hint": "manufacture",
                "object_hint": "chips",
                "confidence": 0.8
            }
        ]
    }

    print("🚀 Extracting Classes for Subjects and Objects")
    print("=" * 60)

    # Initialize the embedder
    embedder = EnhancedCSVEmbedder()
    embedder.load_iri_mapping("label_iri_mappings.csv")

    # Load embeddings from cache
    print("\nLoading embeddings...")
    csv_files = {
        "industries": "industries_enhanced.csv",
        "material_caps": "material_caps_enhanced.csv",
        "certificates": "certificates_enhanced_fixed.csv",
        "process_embeddings": "process_embeddings_enhanced_with_def.csv"
    }
    embedder.embed_csv_files(csv_files, force_recompute=False)

    print("\n" + "=" * 60)

    # Extract subjects and objects
    subjects, objects = extract_subjects_and_objects(clauses_data)

    print(f"\n📊 Extraction Summary:")
    print(f"   • Unique subjects: {subjects}")
    print(f"   • Unique objects: {objects}")

    # Find classes for subjects
    print("\n" + "=" * 60)
    subject_classes = find_classes_for_entities(embedder, subjects, "Subject")

    # Find classes for objects
    print("\n" + "=" * 60)
    object_classes = find_classes_for_entities(embedder, objects, "Object")

    # Display final results
    print("\n" + "=" * 60)
    print("🎯 FINAL RESULTS - SUBJECT CLASSES")
    print("=" * 60)

    for subject, result in subject_classes.items():
        print(f"\n📌 Subject: '{subject}'")
        if result['best_match']:
            best = result['best_match']
            print(f"   ✅ Best Class: {best['label']}")
            print(f"   🔗 IRI: {best['iri']}")
            print(f"   📊 Score: {best['score']:.4f}")
            print(f"   🏷️  Category: {best['category']}")
            if best['definition']:
                definition = best['definition'][:100] + "..." if len(best['definition']) > 100 else best['definition']
                print(f"   📝 Definition: {definition}")
        else:
            print("   ❌ No class found")

    print("\n" + "=" * 60)
    print("🎯 FINAL RESULTS - OBJECT CLASSES")
    print("=" * 60)

    for obj, result in object_classes.items():
        print(f"\n📌 Object: '{obj}'")
        if result['best_match']:
            best = result['best_match']
            print(f"   ✅ Best Class: {best['label']}")
            print(f"   🔗 IRI: {best['iri']}")
            print(f"   📊 Score: {best['score']:.4f}")
            print(f"   🏷️  Category: {best['category']}")
            if best['definition']:
                definition = best['definition'][:100] + "..." if len(best['definition']) > 100 else best['definition']
                print(f"   📝 Definition: {definition}")
        else:
            print("   ❌ No class found")

    # Return structured results
    return {
        'subjects': subjects,
        'objects': objects,
        'subject_classes': subject_classes,
        'object_classes': object_classes
    }

if __name__ == "__main__":
    results = test_clause_extraction()
