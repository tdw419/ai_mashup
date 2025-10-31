# engine/stores/lancedb_store.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Tuple, Dict, Any, Optional
import os
import lancedb
import pyarrow as pa
import numpy as np

# For manual embedding if automatic fails
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

@dataclass
class Hit:
    text: str
    ts: datetime
    score: float
    meta: dict

class LanceHistoryStore:
    def __init__(self, uri: str = "./.lancedb", table_name: str = "history",
                 embed_model: str = "all-MiniLM-L6-v2"):
        self.uri = uri
        self.table_name = table_name
        self.embed_model_name = embed_model
        self.embedder = None
        self.db = None
        self.tbl = None

        self._init_db()

    def _init_db(self):
        """Initialize database and table with proper embedding setup"""
        try:
            # Connect to database
            self.db = lancedb.connect(self.uri)

            # Initialize embedder
            self._init_embedder()

            # Create or open table
            if self.table_name in self.db.table_names():
                self.tbl = self.db.open_table(self.table_name)
                print(f"✓ Opened existing table: {self.table_name}")
            else:
                self._create_table()

        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            # Fallback to creating the table if it doesn't exist
            if "does not exist" in str(e):
                self._create_table()
            else:
                raise

    def _init_embedder(self):
        """Initialize the embedding model"""
        if HAS_SENTENCE_TRANSFORMERS:
            self.embedder = SentenceTransformer(self.embed_model_name)
            print(f"✓ Initialized SentenceTransformer embedder: {self.embed_model_name}")
        else:
            raise ImportError("sentence-transformers is required for embedding. Please install it with: pip install sentence-transformers")

    def _create_table(self):
        """Create a new table with proper schema"""
        try:
            # Define schema
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("ts", pa.float64()),
                pa.field("meta", pa.string()),  # Store as JSON string
                pa.field("vector", pa.list_(pa.float32(), 384))  # Fixed dimension for MiniLM
            ])

            self.tbl = self.db.create_table(self.table_name, schema=schema)
            print(f"✓ Created new table: {self.table_name}")
        except Exception as e:
            print(f"❌ Failed to create table: {e}")
            raise

    def _embed_text(self, text: str) -> List[float]:
        """Embed text using the configured embedder"""
        if hasattr(self.embedder, 'embed'):
            # LanceDB embedder
            return self.embedder.embed(text)
        elif hasattr(self.embedder, 'encode'):
            # Direct SentenceTransformer
            embedding = self.embedder.encode(text)
            return embedding.tolist()
        else:
            raise AttributeError("Embedder has no embed or encode method")

    def add(self, rows: Iterable[Tuple[str, str, datetime, Dict[str, Any]]]):
        """Add rows to the table with proper embedding"""
        data = []
        for _id, text, ts, meta in rows:
            # Convert datetime to timestamp
            if isinstance(ts, datetime):
                timestamp = ts.replace(tzinfo=timezone.utc).timestamp()
            else:
                timestamp = float(ts)

            # Generate embedding
            try:
                vector = self._embed_text(text)
            except Exception as e:
                print(f"⚠️ Embedding failed for text '{text[:50]}...': {e}")
                continue

            # Prepare data
            data.append({
                "id": _id,
                "text": text,
                "ts": timestamp,
                "meta": str(meta),  # Convert dict to string
                "vector": vector
            })

        if data:
            try:
                self.tbl.add(data)
                print(f"✓ Added {len(data)} documents to {self.table_name}")
            except Exception as e:
                print(f"❌ Failed to add documents: {e}")
                raise

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search the table with automatic embedding"""
        try:
            # Embed the query
            query_vector = self._embed_text(query)

            # Perform search
            results = self.tbl.search(query_vector).limit(top_k).to_list()

            # Convert to expected format
            hits = []
            for result in results:
                # Parse meta back from string
                try:
                    import ast
                    meta = ast.literal_eval(result["meta"])
                except:
                    meta = {"raw": result["meta"]}

                # Convert timestamp back to datetime
                ts = datetime.fromtimestamp(result["ts"], tz=timezone.utc)

                # LanceDB returns _distance, convert to similarity score
                distance = result.get("_distance", 1.0)
                similarity = 1.0 / (1.0 + distance)  # Simple conversion

                hits.append((result["text"], similarity, meta))

            return hits

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    def get_table_info(self) -> Dict[str, Any]:
        """Get information about the table"""
        if not self.tbl:
            return {"error": "Table not initialized"}

        try:
            count = self.tbl.count_rows()
            schema = self.tbl.schema
            return {
                "table_name": self.table_name,
                "row_count": count,
                "schema": str(schema),
                "embedder": self.embed_model_name
            }
        except Exception as e:
            return {"error": str(e)}

# Alternative simplified version if above doesn't work
class SimpleLanceStore:
    """Simplified version that manually handles everything"""

    def __init__(self, uri: str = "./.lancedb", table_name: str = "history"):
        self.uri = uri
        self.table_name = table_name
        self.db = lancedb.connect(uri)

        # Manual embedder
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError("Install: pip install sentence-transformers")

        # Create table if needed
        if table_name not in self.db.table_names():
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), 384)),
                pa.field("meta", pa.string())
            ])
            self.tbl = self.db.create_table(table_name, schema=schema, data=[])
        else:
            self.tbl = self.db.open_table(table_name)

    def add(self, rows):
        data = []
        for _id, text, ts, meta in rows:
            vector = self.embedder.encode(text).tolist()
            data.append({
                "id": _id,
                "text": text,
                "vector": vector,
                "meta": str(meta)
            })
        if data:
            self.tbl.add(data)

    def search(self, query, top_k=5):
        query_vec = self.embedder.encode(query).tolist()
        results = self.tbl.search(query_vec).limit(top_k).to_list()

        hits = []
        for r in results:
            try:
                import ast
                meta = ast.literal_eval(r["meta"])
            except:
                meta = {"raw": r["meta"]}

            distance = r.get("_distance", 1.0)
            score = 1.0 / (1.0 + distance)
            hits.append((r["text"], score, meta))

        return hits
