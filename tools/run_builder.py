#!/usr/bin/env python3
# tools/run_builder.py
import argparse, yaml, os
from engine.llm import LLM
from engine.stores.lancedb_store import LanceHistoryStore
from engine.workflow import BuilderFirstWorkflow

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="Run Builder-First workflow once.")
    ap.add_argument("--prompt", required=True, help="User prompt/task")
    ap.add_argument("--cfg", default="config/builder_mode.yaml")
    ap.add_argument("--db", default="./.lancedb")
    ap.add_argument("--table", default="history")
    ap.add_argument("--out", default=None, help="Optional path to save output markdown")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    llm = LLM()
    vecdb = LanceHistoryStore(uri=args.db, table_name=args.table)

    wf = BuilderFirstWorkflow(llm=llm, vecdb=vecdb, cfg=cfg)
    final, report = wf.run(args.prompt)

    print(final)
    print("\n---\nGate Report:")
    print(f"problems={report.problems} solutions={report.solutions} ratio={report.ratio:.2f} "
          f"actionability={report.actionability:.2f} specificity={report.specificity:.2f} "
          f"vague_ratio={report.vague_ratio:.2f} passed={report.passed}")
    if report.notes:
        print("notes:", "; ".join(report.notes))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(final)
        print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
