#!/usr/bin/env python3
"""
Enhanced CSV Embedding Pipeline

This script:
1. Embeds industry_enhanced, material_caps_enhanced, certificates_enhanced, and process_embeddings_enhanced_with_def CSV files
2. Creates a search system to find closest label matches using embeddings
3. Maps found labels to their IRIs using label_iri_mapping.csv

Features:
- Batch embedding with OpenAI
- Cosine similarity search
- IRI mapping integration
- Progress tracking and error handling
- Results caching for efficiency
"""

import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
from dataclasses import dataclass
import pickle
from datetime import datetime

# Optional .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

# Import permanent storage
from permanent_embedding_storage import PermanentEmbeddingStorage


@dataclass
class EmbeddingResult:
    """Result of embedding search"""
    label: str
    score: float
    iri: str
    category: str
    hierarchy: str
    definition: str
    source_file: str


class EnhancedCSVEmbedder:
    """Manage CSV-driven embedding corpora and perform similarity search with IRI mapping."""

    def __init__(self, openai_model: str = "text-embedding-3-large", cache_dir: str = "embedding_cache"):
        """Initialize OpenAI client, cache directory, and in-memory stores."""
        self.openai_model = openai_model
        self.client = OpenAI()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Storage for embeddings and metadata
        self.embeddings = {}
        self.metadata = {}
        self.iri_mapping = {}

        print(f"Initialized Enhanced CSV Embedder with model: {openai_model}")

    def load_iri_mapping(self, iri_mapping_file: str = "label_iri_mappings.csv"):
        """Load the label to IRI mapping from CSV.

        Parameters
        ----------
        iri_mapping_file : str
            Path to a CSV containing ``label`` and ``iri`` columns.
        """
        try:
            df = pd.read_csv(iri_mapping_file, encoding='utf-8')
            for _, row in df.iterrows():
                label = row['label'].strip().lower()
                self.iri_mapping[label] = {
                    'iri': row['iri'],
                    'type': row.get('type', ''),
                    'id': row.get('id', ''),
                    'source_file': row.get('source_file', '')
                }
            print(f"Loaded {len(self.iri_mapping)} label-to-IRI mappings")
        except Exception as e:
            print(f"Warning: Could not load IRI mapping from {iri_mapping_file}: {e}")
            self.iri_mapping = {}

    def embed_text_batch(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """Generate embeddings for multiple texts with batching.

        Parameters
        ----------
        texts : list[str]
            Input strings to embed.
        batch_size : int, default 100
            Number of texts submitted per OpenAI request.

        Returns
        -------
        list[numpy.ndarray]
            Dense embeddings for each text, in order.
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"  Embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

            try:
                response = self.client.embeddings.create(
                    model=self.openai_model,
                    input=batch
                )

                batch_embeddings = [np.array(item.embedding) for item in response.data]
                embeddings.extend(batch_embeddings)

            except Exception as e:
                print(f"Error in batch embedding: {e}")
                # Add zero vectors as fallback
                embeddings.extend([np.zeros(1536) for _ in batch])

        return embeddings

    def process_csv_file(self, csv_file: str, source_name: str) -> Tuple[List[str], List[Dict]]:
        """Process a CSV file and extract texts for embedding and metadata.

        Parameters
        ----------
        csv_file : str
            Path to the CSV file to read.
        source_name : str
            Logical identifier for the corpus (e.g., ``"industries"``).

        Returns
        -------
        tuple[list[str], list[dict]]
            Prepared embedding texts and their metadata dictionaries.
        """
        print(f"\nProcessing {csv_file} ({source_name})")

        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            print(f"  Loaded {len(df)} rows")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            return [], []

        texts_to_embed = []
        metadata_list = []

        for _, row in df.iterrows():
            # Create embedding text from Label + Definition
            label = str(row.get('Label', '')).strip()
            definition = str(row.get('Definition', '')).strip()

            if not label:
                continue

            # Combine label and definition for better embeddings
            if definition and definition != 'nan':
                embed_text = f"{label}: {definition}"
            else:
                embed_text = label

            texts_to_embed.append(embed_text)

            metadata_list.append({
                'label': label,
                'category': str(row.get('Category', '')).strip(),
                'hierarchy': str(row.get('Hierarchy', '')).strip(),
                'definition': definition,
                'also_called': str(row.get('Also Called', '')).strip(),
                'source_file': source_name,
                'embed_text': embed_text
            })

        print(f"  Prepared {len(texts_to_embed)} items for embedding")
        return texts_to_embed, metadata_list

    def embed_csv_files(self, csv_files: Dict[str, str], force_recompute: bool = False):
        """Embed all CSV files and cache results."""
        # Use model-specific cache file name for reusability
        model_suffix = self.openai_model.replace('/', '_').replace(':', '_')
        cache_file = self.cache_dir / f"enhanced_csv_embeddings_{model_suffix}.pkl"

        # Try to load from cache if it exists and force_recompute is False
        if cache_file.exists() and not force_recompute:
            print(f"Loading embeddings from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                    # Verify the model matches (should always match due to filename, but double-check)
                    cached_model = cached_data.get('model', self.openai_model)
                    if cached_model == self.openai_model:
                        self.embeddings = cached_data['embeddings']
                        self.metadata = cached_data['metadata']
                        print(f"✅ Loaded {sum(len(emb) for emb in self.embeddings.values())} cached embeddings (model: {cached_model})")
                        return
                    else:
                        print(f"⚠️  Model mismatch in cache file, recomputing...")
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing embeddings...")

        print(f"Computing embeddings for all CSV files with model: {self.openai_model}...")

        for source_name, csv_file in csv_files.items():
            if not Path(csv_file).exists():
                print(f"Warning: File not found: {csv_file}")
                continue

            # Process CSV file
            texts, metadata = self.process_csv_file(csv_file, source_name)

            if not texts:
                print(f"  No texts to embed for {source_name}")
                continue

            # Generate embeddings
            print(f"  Generating embeddings for {len(texts)} items...")
            embeddings = self.embed_text_batch(texts)

            # Store results
            self.embeddings[source_name] = np.array(embeddings)
            self.metadata[source_name] = metadata

            print(f"  ✅ Completed {source_name}: {len(embeddings)} embeddings")

        # Save to cache
        try:
            cache_data = {
                'embeddings': self.embeddings,
                'metadata': self.metadata,
                'timestamp': datetime.now().isoformat(),
                'model': self.openai_model
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"\n💾 Cached embeddings to: {cache_file}")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def find_best_iri_match(self, query_label: str, threshold: float = 0.7) -> Optional[str]:
        """Find the best IRI given a label using fast heuristics before embeddings."""
        if not self.iri_mapping:
            return None

        query_lower = query_label.strip().lower()

        # Try exact match first (fastest)
        exact_match = self.iri_mapping.get(query_lower)
        if exact_match:
            return exact_match.get('iri')

        # Generate capability-style variants for better matching
        capability_variants = []

        # Variant 1: Remove spaces and add "Capability"
        no_spaces = query_lower.replace(' ', '').replace('-', '').replace('_', '')
        capability_variants.append(no_spaces + 'capability')

        # Variant 2: CamelCase with Capability suffix
        words = query_lower.replace('-', ' ').replace('_', ' ').split()
        if words:
            camel_case = ''.join(word.capitalize() for word in words)
            capability_variants.append((camel_case + 'Capability').lower())

        # Variant 3: Just the capability suffix
        capability_variants.append(no_spaces)

        # Try capability variants
        for variant in capability_variants:
            exact_match = self.iri_mapping.get(variant)
            if exact_match:
                return exact_match.get('iri')

        # Try partial matches with different strategies (much faster than embeddings)
        best_score = 0.0
        best_iri = None

        for mapped_label, iri_info in self.iri_mapping.items():
            # Strategy 1: Check if query is contained in mapped label
            if query_lower in mapped_label:
                score = len(query_lower) / len(mapped_label)  # Longer matches get higher scores
                if score > best_score:
                    best_score = score
                    best_iri = iri_info.get('iri')

            # Strategy 2: Check if mapped label is contained in query
            elif mapped_label in query_lower:
                score = len(mapped_label) / len(query_lower)
                if score > best_score:
                    best_score = score
                    best_iri = iri_info.get('iri')

            # Strategy 3: Check capability-style matching
            elif 'capability' in mapped_label and no_spaces in mapped_label:
                # Give high score for capability matches
                score = 0.8
                if score > best_score:
                    best_score = score
                    best_iri = iri_info.get('iri')

            # Strategy 4: Simple word overlap scoring
            else:
                query_words = set(query_lower.split())
                mapped_words = set(mapped_label.split())
                if query_words and mapped_words:
                    overlap = len(query_words & mapped_words)
                    total_words = len(query_words | mapped_words)
                    if total_words > 0:
                        score = overlap / total_words
                        if score >= threshold and score > best_score:
                            best_score = score
                            best_iri = iri_info.get('iri')

        return best_iri if best_score >= 0.3 else None  # Minimum threshold for any match

    def search_closest_labels(self, query: str, top_k: int = 5,
                            source_filter: Optional[List[str]] = None) -> List[EmbeddingResult]:
        """Search for closest labels using embedding similarity."""
        if not self.embeddings:
            print("No embeddings loaded. Please run embed_csv_files() first.")
            return []

        # Generate query embedding
        print(f"Searching for: '{query}'")
        try:
            response = self.client.embeddings.create(
                model=self.openai_model,
                input=[query]
            )
            query_embedding = np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []

        # Normalize query embedding
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)

        all_results = []

        # Search across all sources (or filtered sources)
        sources_to_search = source_filter if source_filter else list(self.embeddings.keys())

        for source_name in sources_to_search:
            if source_name not in self.embeddings:
                continue

            embeddings = self.embeddings[source_name]
            metadata = self.metadata[source_name]

            if len(embeddings) == 0:
                continue

            # Normalize embeddings
            embedding_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

            # Calculate cosine similarities
            similarities = np.dot(embedding_norms, query_norm)

            # Get results for this source
            for i, score in enumerate(similarities):
                meta = metadata[i]
                label = meta['label']

                # Use semantic IRI matching instead of exact matching
                iri = self.find_best_iri_match(label, threshold=0.7)
                if not iri:
                    iri = f'No IRI found for: {label}'

                result = EmbeddingResult(
                    label=label,
                    score=float(score),
                    iri=iri,
                    category=meta['category'],
                    hierarchy=meta['hierarchy'],
                    definition=meta['definition'],
                    source_file=source_name
                )
                all_results.append(result)

        # Sort by score and return top_k
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

    def search_and_display_results(self, query: str, top_k: int = 5,
                                 source_filter: Optional[List[str]] = None, generic_display: bool = True):
        """Search and display results in a formatted way with optional generic display."""
        results = self.search_closest_labels(query, top_k, source_filter)

        print(f"\n🔍 Top {len(results)} results for: '{query}'")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            # Use generic label display if requested
            display_label = result.label
            if generic_display:
                display_label = self.normalize_label_for_display(result.label, result.iri)

            print(f"\n{i}. 📌 {display_label} (Score: {result.score:.4f})")
            print(f"   🏷️  Category: {result.category}")
            print(f"   🔗 IRI: {result.iri}")

            # Show generic process type extracted from IRI
            if generic_display and result.iri and "No IRI found" not in result.iri:
                process_type = self.extract_process_type_from_iri(result.iri)
                if process_type and process_type != display_label:
                    print(f"   🔬 Process Type: {process_type}")

            print(f"   📂 Source: {result.source_file}")
            if result.hierarchy:
                print(f"   📊 Hierarchy: {result.hierarchy}")
            if result.definition and result.definition != 'nan':
                definition_preview = result.definition[:150] + "..." if len(result.definition) > 150 else result.definition
                print(f"   📝 Definition: {definition_preview}")

        return results

    def normalize_label_for_display(self, label: str, iri: str = "") -> str:
        """Convert capability-style labels to more generic, user-friendly format."""
        # Remove common suffixes to make labels more generic
        suffixes_to_remove = [
            "Capability", "ProcessingCapability", "MachiningCapability",
            "ManufacturingCapability", "IndustryCapability", "QualityCapability",
            "CertificateCapability", "TechnologyCapability"
        ]

        normalized = label

        # Remove capability suffixes (case insensitive)
        for suffix in suffixes_to_remove:
            if normalized.lower().endswith(suffix.lower()):
                normalized = normalized[:-len(suffix)].strip()
                break

        # Convert from CamelCase to readable format if needed
        if normalized and not any(c.islower() for c in normalized):
            # If all caps or CamelCase, convert to readable format
            import re
            # Add spaces before capital letters
            spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', normalized)
            # Add spaces before numbers
            spaced = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', spaced)
            # Clean up multiple spaces
            spaced = re.sub(r'\s+', ' ', spaced).strip()

            if spaced != normalized:
                normalized = spaced.title()

        return normalized if normalized else label

    def extract_process_type_from_iri(self, iri: str) -> str:
        """Extract a more generic process type from IRI."""
        if not iri or "No IRI found" in iri:
            return ""

        # Extract the last part of the IRI (the actual class name)
        try:
            class_name = iri.split('/')[-1]
            # Remove common suffixes
            for suffix in ["Capability", "ProcessingCapability", "MachiningCapability"]:
                if class_name.endswith(suffix):
                    class_name = class_name[:-len(suffix)]
                    break

            # Convert CamelCase to readable
            import re
            readable = re.sub(r'([a-z])([A-Z])', r'\1 \2', class_name)
            return readable.title() if readable else class_name
        except:
            return ""

    def process_structured_clauses(self, clauses_data: Dict) -> List[EmbeddingResult]:
        """Process structured clauses and convert them to search queries."""
        clauses = clauses_data.get('clauses', [])
        warnings = clauses_data.get('warnings', [])

        print("🔍 Processing structured clauses...")
        print(f"Found {len(clauses)} clauses with {len(warnings)} warnings")

        # Extract search terms from clauses
        search_terms = []
        filters = []
        location_info = {}

        for i, clause in enumerate(clauses):
            clause_type = clause.get('type', '')
            surface = clause.get('surface', '')
            subject_hint = clause.get('subject_hint', '')
            relation_hint = clause.get('relation_hint', '')
            object_hint = clause.get('object_hint', '')
            confidence = clause.get('confidence', 0.0)

            print(f"\n  📋 Clause {i+1}: '{surface}' (confidence: {confidence:.2f})")

            if clause_type == 'spo':  # Subject-Predicate-Object
                if relation_hint in ['produce', 'manufacture']:
                    search_terms.append(object_hint)
                    print(f"    ➜ Product: {object_hint}")
                elif relation_hint == 'made for':
                    search_terms.append(f"{object_hint} grade")
                    print(f"    ➜ Grade: {object_hint}")
                elif relation_hint == 'located in':
                    location_parts = object_hint.split()
                    if len(location_parts) >= 2:
                        location_info['city'] = location_parts[0]
                        location_info['state'] = ' '.join(location_parts[1:])
                    print(f"    ➜ Location: {object_hint}")
                elif relation_hint == 'have certification':
                    search_terms.append(f"{object_hint} certification")
                    print(f"    ➜ Certification: {object_hint}")

            elif clause_type == 'filter':  # Numerical or categorical filter
                filter_info = clause.get('filter', {})
                if filter_info.get('op') == 'gt':
                    filters.append({
                        'type': 'employees',
                        'operator': 'greater_than',
                        'value': filter_info.get('value'),
                        'unit': filter_info.get('unit', '')
                    })
                    print(f"    ➜ Filter: >{filter_info.get('value')} {filter_info.get('unit', '')}")

        # Combine search terms into comprehensive queries
        combined_query = ' '.join(search_terms)
        print(f"\n🔎 Combined search query: '{combined_query}'")

        if location_info:
            print(f"🌍 Location filters: {location_info}")
        if filters:
            print(f"📊 Numerical filters: {filters}")

        # Perform the search
        results = self.search_closest_labels(combined_query, top_k=10)

        # Apply post-processing filters (location, employee count, etc.)
        filtered_results = self.apply_clause_filters(results, location_info, filters)

        return filtered_results

    def apply_clause_filters(self, results: List[EmbeddingResult], location_info: Dict, filters: List[Dict]) -> List[EmbeddingResult]:
        """Apply additional filters based on structured clauses."""
        filtered_results = []

        print(f"\n🔧 Applying filters to {len(results)} results...")

        for result in results:
            include_result = True

            # Location filtering (basic implementation)
            if location_info and result.hierarchy:
                city = location_info.get('city', '').lower()
                state = location_info.get('state', '').lower()
                hierarchy_lower = result.hierarchy.lower()

                if city and city not in hierarchy_lower:
                    # For demo purposes, we'll be lenient with location matching
                    # In production, you'd have proper geographic data
                    pass

            # Employee count filtering (would need additional data source)
            for filter_item in filters:
                if filter_item['type'] == 'employees':
                    # This would require additional company data
                    # For now, we'll include all results
                    pass

            if include_result:
                filtered_results.append(result)

        print(f"✅ Filtered down to {len(filtered_results)} results")
        return filtered_results

    def search_with_structured_clauses(self, clauses_data: Dict, display_results: bool = True) -> List[EmbeddingResult]:
        """Main entry point for structured clause searching."""
        results = self.process_structured_clauses(clauses_data)

        if display_results:
            print(f"\n🎯 Results for structured query:")
            print("=" * 80)

            for i, result in enumerate(results[:10], 1):  # Show top 10
                display_label = self.normalize_label_for_display(result.label, result.iri)

                print(f"\n{i}. 📌 {display_label} (Score: {result.score:.4f})")
                print(f"   🏷️  Category: {result.category}")
                print(f"   🔗 IRI: {result.iri}")
                print(f"   📂 Source: {result.source_file}")

                if result.hierarchy:
                    print(f"   📊 Hierarchy: {result.hierarchy}")
                if result.definition and result.definition != 'nan':
                    definition_preview = result.definition[:150] + "..." if len(result.definition) > 150 else result.definition
                    print(f"   📝 Definition: {definition_preview}")

        return results

    # ...existing code...
def main():
    """CLI entrypoint for embedding generation and interactive search."""
    parser = argparse.ArgumentParser(description="Enhanced CSV Embedding Pipeline")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings for CSV files")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--sources", nargs="+", help="Filter by source files")
    parser.add_argument("--force", action="store_true", help="Force recompute embeddings")
    parser.add_argument("--model", default="text-embedding-3-large", help="OpenAI embedding model")

    args = parser.parse_args()

    # Initialize embedder
    embedder = EnhancedCSVEmbedder(openai_model=args.model)

    # Load IRI mapping
    embedder.load_iri_mapping("label_iri_mappings.csv")

    # Define CSV files to process
    csv_files = {
        "industries": "industries_enhanced.csv",
        "material_caps": "material_caps_enhanced.csv",
        "certificates": "certificates_enhanced_fixed.csv",
        "process_embeddings": "process_embeddings_enhanced_with_def.csv"
    }

    # Generate embeddings if requested or if cache doesn't exist
    model_suffix = args.model.replace('/', '_').replace(':', '_')
    cache_file = Path(f"embedding_cache/enhanced_csv_embeddings_{model_suffix}.pkl")

    if args.embed or not cache_file.exists():
        embedder.embed_csv_files(csv_files, force_recompute=args.force)
    else:
        embedder.embed_csv_files(csv_files, force_recompute=False)

    # Perform search if query provided
    if args.search:
        results = embedder.search_and_display_results(
            args.search,
            top_k=args.top_k,
            source_filter=args.sources
        )

    # Interactive mode if no search query provided
    elif not args.embed:
        print("\n🎯 Interactive Search Mode")
        print("Enter search queries (or 'quit' to exit):")

        while True:
            try:
                query = input("\n🔍 Search: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if not query:
                    continue

                results = embedder.search_and_display_results(query, top_k=args.top_k)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break


if __name__ == "__main__":
    main()
