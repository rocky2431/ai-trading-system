#!/usr/bin/env python3
"""Initialize Qdrant vector database for IQFMP factor knowledge base.

This script:
1. Creates required collections with proper vector configurations
2. Sets up indexes for efficient similarity search
3. Configures payload filtering for factor metadata

Usage:
    python scripts/init_qdrant.py
    python scripts/init_qdrant.py --recreate  # Drop and recreate collections
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("Warning: qdrant-client not installed. Install with: pip install qdrant-client")


# Qdrant configuration
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)

# Collection configurations
COLLECTIONS = {
    "factors": {
        "description": "Factor knowledge base - stores factor embeddings for deduplication",
        "vector_size": 768,  # text-embedding-ada-002 or similar
        "distance": models.Distance.COSINE,
        "payload_schema": {
            "factor_name": models.PayloadSchemaType.KEYWORD,
            "factor_family": models.PayloadSchemaType.KEYWORD,
            "expression": models.PayloadSchemaType.TEXT,
            "sharpe_ratio": models.PayloadSchemaType.FLOAT,
            "ic_mean": models.PayloadSchemaType.FLOAT,
            "created_at": models.PayloadSchemaType.DATETIME,
            "status": models.PayloadSchemaType.KEYWORD,
        },
    },
    "research_trials": {
        "description": "Research trial embeddings - for similar trial lookup",
        "vector_size": 768,
        "distance": models.Distance.COSINE,
        "payload_schema": {
            "trial_id": models.PayloadSchemaType.KEYWORD,
            "factor_name": models.PayloadSchemaType.KEYWORD,
            "factor_family": models.PayloadSchemaType.KEYWORD,
            "sharpe_ratio": models.PayloadSchemaType.FLOAT,
            "passed_threshold": models.PayloadSchemaType.BOOL,
            "created_at": models.PayloadSchemaType.DATETIME,
        },
    },
    "factor_clusters": {
        "description": "Factor cluster centroids - for clustering and categorization",
        "vector_size": 768,
        "distance": models.Distance.COSINE,
        "payload_schema": {
            "cluster_id": models.PayloadSchemaType.KEYWORD,
            "cluster_name": models.PayloadSchemaType.KEYWORD,
            "factor_count": models.PayloadSchemaType.INTEGER,
            "avg_sharpe": models.PayloadSchemaType.FLOAT,
            "updated_at": models.PayloadSchemaType.DATETIME,
        },
    },
}


def get_client() -> "QdrantClient":
    """Create Qdrant client."""
    if QDRANT_API_KEY:
        return QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
        )
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def check_connection(client: "QdrantClient") -> bool:
    """Check if Qdrant is accessible."""
    try:
        client.get_collections()
        return True
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        return False


def create_collection(
    client: "QdrantClient",
    name: str,
    config: dict,
    recreate: bool = False,
) -> bool:
    """Create a Qdrant collection with the specified configuration."""
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(c.name == name for c in collections)

        if exists:
            if recreate:
                print(f"  Dropping existing collection: {name}")
                client.delete_collection(name)
            else:
                print(f"  Collection already exists: {name} (use --recreate to reset)")
                return True

        # Create collection
        print(f"  Creating collection: {name}")
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=config["vector_size"],
                distance=config["distance"],
            ),
        )

        # Create payload indexes for filtering
        print(f"  Creating payload indexes for: {name}")
        for field_name, field_type in config["payload_schema"].items():
            try:
                client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=field_type,
                )
            except Exception as e:
                print(f"    Warning: Failed to create index for {field_name}: {e}")

        print(f"  Collection {name} created successfully!")
        return True

    except Exception as e:
        print(f"  Error creating collection {name}: {e}")
        return False


def init_qdrant(recreate: bool = False):
    """Initialize all Qdrant collections."""
    if not QDRANT_AVAILABLE:
        print("ERROR: qdrant-client not installed")
        sys.exit(1)

    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    client = get_client()

    if not check_connection(client):
        print("\nERROR: Cannot connect to Qdrant. Please ensure:")
        print("  1. Qdrant is running (docker-compose up -d qdrant)")
        print("  2. Host/port are correct (QDRANT_HOST, QDRANT_PORT)")
        sys.exit(1)

    print("\n=== Initializing Qdrant Collections ===\n")

    success = True
    for name, config in COLLECTIONS.items():
        print(f"\n{config['description']}")
        if not create_collection(client, name, config, recreate):
            success = False

    print("\n=== Qdrant Initialization Summary ===")
    collections = client.get_collections().collections
    for c in collections:
        info = client.get_collection(c.name)
        print(f"  {c.name}: {info.points_count} vectors, {info.vectors_count} total")

    if success:
        print("\n=== Qdrant initialization complete! ===")
    else:
        print("\n=== Qdrant initialization completed with warnings ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize Qdrant vector database")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate all collections",
    )
    args = parser.parse_args()

    if args.recreate:
        print("WARNING: This will drop all existing collections!")
        confirm = input("Are you sure? (yes/no): ")
        if confirm.lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    init_qdrant(recreate=args.recreate)
