#!/usr/bin/env python3
# scripts/index_history.py
import argparse, os, glob, hashlib, json
from datetime import datetime, timezone
from engine.stores.lancedb_store import LanceHistoryStore

def sniff_ts(path: str) -> datetime:
    try:
        ts = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        return ts
    except Exception:
        return datetime.now(timezone.utc)

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def iter_docs(input_glob: str):
    for p in glob.glob(input_glob, recursive=True):
        if os.path.isdir(p):
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext in (".md", ".txt", ".json"):
            text = load_text(p)
            # If JSON, try to pull 'content' field
            if ext == ".json":
                try:
                    obj = json.loads(text)
                    if isinstance(obj, dict) and "content" in obj:
                        text = obj["content"]
                except Exception:
                    pass
            h = hashlib.sha1(p.encode()).hexdigest()
            ts = sniff_ts(p)
            meta = {"path": p, "source": "history_file", "timestamp": ts.isoformat()}
            yield (h, text, ts, meta)

def main():
    ap = argparse.ArgumentParser(description="Index past chats / notes into LanceDB.")
    ap.add_argument("--glob", default="history/**/*.*", help="Glob of files to index")
    ap.add_argument("--db", default="./.lancedb", help="LanceDB uri dir")
    ap.add_argument("--table", default="history", help="Table name")
    args = ap.parse_args()

    print(f"📚 Indexing files from {args.glob}...")

    try:
        store = LanceHistoryStore(uri=args.db, table_name=args.table)

        # Get some info about the table
        info = store.get_table_info()
        print(f"📊 Table info: {info}")

        # Index documents
        docs = list(iter_docs(args.glob))
        print(f"📄 Found {len(docs)} documents to index")

        store.add(docs)
        print(f"✅ Successfully indexed {len(docs)} documents into {args.db}:{args.table}")

    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        raise

if __name__ == "__main__":
    main()
