#!/usr/bin/env python3
"""
Permanent Embedding Storage System

This script creates a robust permanent storage system for embeddings with:
- Multiple storage formats (pickle, numpy, JSON metadata)
- Backup and versioning
- Easy loading and validation
- Export capabilities for other systems
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse
import shutil
import hashlib

class PermanentEmbeddingStorage:
    """Manage versioned embeddings with multi-format persistence and exports."""

    def __init__(self, storage_dir: str = "permanent_embeddings"):
        """Initialize permanent embedding storage."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.storage_dir / "embeddings").mkdir(exist_ok=True)
        (self.storage_dir / "metadata").mkdir(exist_ok=True)
        (self.storage_dir / "backups").mkdir(exist_ok=True)
        (self.storage_dir / "exports").mkdir(exist_ok=True)

        print(f"Initialized permanent storage at: {self.storage_dir}")

    def save_embeddings_permanently(self, embeddings: Dict[str, np.ndarray],
                                  metadata: Dict[str, List[Dict]],
                                  iri_mapping: Dict[str, Dict],
                                  model_name: str = "text-embedding-3-large"):
        """Save embeddings in multiple permanent formats with versioning.

        Parameters
        ----------
        embeddings : dict[str, numpy.ndarray]
            Named embedding matrices keyed by data source.
        metadata : dict[str, list[dict]]
            Metadata entries aligned with each embedding row.
        iri_mapping : dict[str, dict]
            Mapping from normalized labels to IRI metadata.
        model_name : str, default "text-embedding-3-large"
            Identifier recorded alongside the stored version.
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = self.storage_dir / f"version_{timestamp}"
        version_dir.mkdir(exist_ok=True)

        print(f"Saving embeddings permanently to version: {timestamp}")

        # 1. Save embeddings as numpy arrays (efficient, preserves precision)
        embeddings_dir = version_dir / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)

        for source_name, embedding_array in embeddings.items():
            embedding_file = embeddings_dir / f"{source_name}_embeddings.npy"
            np.save(embedding_file, embedding_array)
            print(f"  ✅ Saved {source_name} embeddings: {embedding_array.shape}")

        # 2. Save metadata as JSON (human-readable, portable)
        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved metadata for {len(metadata)} sources")

        # 3. Save IRI mapping
        iri_file = version_dir / "iri_mapping.json"
        with open(iri_file, 'w', encoding='utf-8') as f:
            json.dump(iri_mapping, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Saved {len(iri_mapping)} IRI mappings")

        # 4. Save complete pickle backup (compatibility)
        pickle_file = version_dir / "complete_embeddings.pkl"
        complete_data = {
            'embeddings': embeddings,
            'metadata': metadata,
            'iri_mapping': iri_mapping,
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'version': timestamp
        }
        with open(pickle_file, 'wb') as f:
            pickle.dump(complete_data, f)
        print(f"  ✅ Saved complete pickle backup")

        # 5. Save configuration and stats
        stats = {
            'version': timestamp,
            'model': model_name,
            'creation_date': datetime.now().isoformat(),
            'sources': {
                source: {
                    'embedding_count': len(emb),
                    'embedding_dimension': emb.shape[1] if len(emb.shape) > 1 else 0,
                    'metadata_count': len(metadata.get(source, [])),
                    'file_size_mb': round(emb.nbytes / (1024*1024), 2)
                }
                for source, emb in embeddings.items()
            },
            'total_embeddings': sum(len(emb) for emb in embeddings.values()),
            'total_size_mb': round(sum(emb.nbytes for emb in embeddings.values()) / (1024*1024), 2)
        }

        stats_file = version_dir / "embedding_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✅ Saved embedding statistics")

        # 6. Create a "latest" symlink/copy for easy access
        latest_dir = self.storage_dir / "latest"
        if latest_dir.exists():
            if latest_dir.is_symlink():
                latest_dir.unlink()
            else:
                shutil.rmtree(latest_dir)

        # Copy instead of symlink for Windows compatibility
        shutil.copytree(version_dir, latest_dir)
        print(f"  ✅ Updated 'latest' version pointer")

        # 7. Create index file
        self._update_index(timestamp, stats)

        print(f"\n🎉 Successfully saved embeddings permanently!")
        print(f"   Version: {timestamp}")
        print(f"   Location: {version_dir}")
        print(f"   Total embeddings: {stats['total_embeddings']}")
        print(f"   Total size: {stats['total_size_mb']} MB")

        return version_dir

    def _update_index(self, version: str, stats: Dict):
        """Update the index of all saved versions."""
        index_file = self.storage_dir / "index.json"

        # Load existing index
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        else:
            index = {'versions': []}

        # Add new version
        version_info = {
            'version': version,
            'timestamp': stats['creation_date'],
            'model': stats['model'],
            'total_embeddings': stats['total_embeddings'],
            'total_size_mb': stats['total_size_mb'],
            'sources': list(stats['sources'].keys())
        }

        # Remove any existing entry with same version
        index['versions'] = [v for v in index['versions'] if v['version'] != version]
        index['versions'].append(version_info)

        # Sort by timestamp (newest first)
        index['versions'].sort(key=lambda x: x['timestamp'], reverse=True)

        # Update index metadata
        index['last_updated'] = datetime.now().isoformat()
        index['latest_version'] = version

        # Save index
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)

    def load_embeddings(self, version: str = "latest") -> Dict[str, Any]:
        """Load embeddings from permanent storage."""
        if version == "latest":
            load_dir = self.storage_dir / "latest"
        else:
            load_dir = self.storage_dir / f"version_{version}"

        if not load_dir.exists():
            raise FileNotFoundError(f"Version {version} not found in {self.storage_dir}")

        print(f"Loading embeddings from version: {version}")

        # Load embeddings
        embeddings = {}
        embeddings_dir = load_dir / "embeddings"

        for npy_file in embeddings_dir.glob("*_embeddings.npy"):
            source_name = npy_file.stem.replace("_embeddings", "")
            embeddings[source_name] = np.load(npy_file)
            print(f"  ✅ Loaded {source_name}: {embeddings[source_name].shape}")

        # Load metadata
        metadata_file = load_dir / "metadata.json"
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Load IRI mapping
        iri_file = load_dir / "iri_mapping.json"
        with open(iri_file, 'r', encoding='utf-8') as f:
            iri_mapping = json.load(f)

        # Load stats
        stats_file = load_dir / "embedding_stats.json"
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        print(f"  ✅ Loaded {len(metadata)} metadata sources")
        print(f"  ✅ Loaded {len(iri_mapping)} IRI mappings")

        return {
            'embeddings': embeddings,
            'metadata': metadata,
            'iri_mapping': iri_mapping,
            'stats': stats
        }

    def list_versions(self) -> List[Dict]:
        """List all available embedding versions"""
        index_file = self.storage_dir / "index.json"

        if not index_file.exists():
            return []

        with open(index_file, 'r', encoding='utf-8') as f:
            index = json.load(f)

        return index.get('versions', [])

    def export_to_formats(self, version: str = "latest", formats: List[str] = None):
        """Export embeddings to various formats for other systems."""
        if formats is None:
            formats = ['csv', 'json', 'parquet']

        data = self.load_embeddings(version)
        embeddings = data['embeddings']
        metadata = data['metadata']

        export_dir = self.storage_dir / "exports" / f"export_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        export_dir.mkdir(parents=True, exist_ok=True)

        print(f"Exporting embeddings to: {export_dir}")

        for source_name in embeddings.keys():
            source_embeddings = embeddings[source_name]
            source_metadata = metadata[source_name]

            # Prepare combined data
            export_data = []
            for i, meta in enumerate(source_metadata):
                row = meta.copy()
                # Add embedding as array
                row['embedding'] = source_embeddings[i].tolist()
                export_data.append(row)

            df = pd.DataFrame(export_data)

            # Export to requested formats
            if 'csv' in formats:
                # For CSV, save embeddings separately due to size
                csv_file = export_dir / f"{source_name}_metadata.csv"
                df_no_embedding = df.drop('embedding', axis=1)
                df_no_embedding.to_csv(csv_file, index=False, encoding='utf-8')

                # Save embeddings as separate numpy file
                emb_file = export_dir / f"{source_name}_embeddings.npy"
                np.save(emb_file, source_embeddings)
                print(f"  ✅ CSV: {csv_file} + {emb_file}")

            if 'json' in formats:
                json_file = export_dir / f"{source_name}_complete.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                print(f"  ✅ JSON: {json_file}")

            if 'parquet' in formats:
                try:
                    parquet_file = export_dir / f"{source_name}_complete.parquet"
                    df.to_parquet(parquet_file, index=False)
                    print(f"  ✅ Parquet: {parquet_file}")
                except ImportError:
                    print(f"  ⚠️  Parquet export requires pyarrow: pip install pyarrow")

        return export_dir

    def backup_version(self, version: str = "latest", backup_name: str = None):
        """Create a backup of a specific version"""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if version == "latest":
            source_dir = self.storage_dir / "latest"
        else:
            source_dir = self.storage_dir / f"version_{version}"

        backup_dir = self.storage_dir / "backups" / backup_name
        shutil.copytree(source_dir, backup_dir)

        print(f"✅ Created backup: {backup_dir}")
        return backup_dir


def main():
    """Command-line interface for saving, loading, listing, and exporting embeddings."""
    parser = argparse.ArgumentParser(description="Permanent Embedding Storage Manager")
    parser.add_argument("--save", action="store_true", help="Save current embeddings permanently")
    parser.add_argument("--list", action="store_true", help="List all saved versions")
    parser.add_argument("--load", type=str, help="Load specific version (or 'latest')")
    parser.add_argument("--export", type=str, help="Export version to other formats")
    parser.add_argument("--backup", type=str, help="Create backup of version")
    parser.add_argument("--formats", nargs="+", default=['csv', 'json'], help="Export formats")
    parser.add_argument("--storage-dir", default="permanent_embeddings", help="Storage directory")

    args = parser.parse_args()

    storage = PermanentEmbeddingStorage(args.storage_dir)

    if args.save:
        # Load current embeddings from cache
        cache_file = Path("embedding_cache/enhanced_csv_embeddings.pkl")
        if not cache_file.exists():
            print("❌ No cached embeddings found. Please run embedding generation first.")
            return

        print("Loading current embeddings from cache...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        embeddings = cached_data['embeddings']
        metadata = cached_data['metadata']
        model = cached_data.get('model', 'text-embedding-3-large')

        # Load IRI mapping
        iri_mapping = {}
        iri_file = Path("label_iri_mappings.csv")
        if iri_file.exists():
            df = pd.read_csv(iri_file, encoding='utf-8')
            for _, row in df.iterrows():
                label = row['label'].strip().lower()
                iri_mapping[label] = {
                    'iri': row['iri'],
                    'type': row.get('type', ''),
                    'id': row.get('id', ''),
                    'source_file': row.get('source_file', '')
                }

        storage.save_embeddings_permanently(embeddings, metadata, iri_mapping, model)

    elif args.list:
        versions = storage.list_versions()
        if not versions:
            print("No saved embedding versions found.")
        else:
            print(f"\n📋 Available embedding versions ({len(versions)}):")
            print("=" * 80)
            for i, version in enumerate(versions, 1):
                print(f"{i}. Version: {version['version']}")
                print(f"   Date: {version['timestamp']}")
                print(f"   Model: {version['model']}")
                print(f"   Embeddings: {version['total_embeddings']}")
                print(f"   Size: {version['total_size_mb']} MB")
                print(f"   Sources: {', '.join(version['sources'])}")
                if i < len(versions):
                    print()

    elif args.load:
        try:
            data = storage.load_embeddings(args.load)
            print(f"✅ Successfully loaded version: {args.load}")
            stats = data['stats']
            print(f"   Total embeddings: {stats['total_embeddings']}")
            print(f"   Sources: {', '.join(stats['sources'].keys())}")
        except FileNotFoundError as e:
            print(f"❌ {e}")

    elif args.export:
        try:
            export_dir = storage.export_to_formats(args.export, args.formats)
            print(f"✅ Successfully exported to: {export_dir}")
        except FileNotFoundError as e:
            print(f"❌ {e}")

    elif args.backup:
        try:
            backup_dir = storage.backup_version(args.backup)
            print(f"✅ Successfully created backup: {backup_dir}")
        except FileNotFoundError as e:
            print(f"❌ {e}")

    else:
        print("Please specify an action: --save, --list, --load, --export, or --backup")
        print("Use --help for more information.")


if __name__ == "__main__":
    main()
