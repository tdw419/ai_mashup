#!/usr/bin/env python3
# scripts/test_lancedb.py
from engine.stores.lancedb_store import LanceHistoryStore
from datetime import datetime, timezone

def test_basic_operations():
    print("🧪 Testing LanceDB integration...")

    try:
        # Create store
        store = LanceHistoryStore(uri="./.lancedb_test", table_name="test_data")

        # Add test data
        test_rows = [
            ("doc1", "How to implement a recursive AI system with builder-first approach",
             datetime.now(timezone.utc), {"type": "test", "topic": "AI"}),
            ("doc2", "LanceDB vector database configuration and embedding setup",
             datetime.now(timezone.utc), {"type": "test", "topic": "database"}),
            ("doc3", "Week 1 deliverables for pixel architecture implementation",
             datetime.now(timezone.utc), {"type": "test", "topic": "architecture"})
        ]

        store.add(test_rows)
        print("✅ Successfully added test data")

        # Test search
        results = store.search("AI system implementation", top_k=2)
        print(f"✅ Search returned {len(results)} results")

        for i, (text, score, meta) in enumerate(results):
            print(f"  {i+1}. Score: {score:.3f}, Text: {text[:60]}...")

        # Cleanup (optional)
        # store.db.drop_table("test_data")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_operations()
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Tests failed")
