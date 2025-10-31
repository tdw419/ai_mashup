#!/usr/bin/env python3
"""
Audit Markdown with the Builder Gate. Exits nonzero on failure.

Usage:
  python tools/gate_check.py --file docs/decisions/foo.md
  # or let pre-commit pass filenames
"""
import argparse, sys, pathlib
from engine.builder_enforcer import audit

def check_text(txt: str, name: str) -> int:
    report = audit(txt)
    if report.passed:
        print(f"✓ PASS {name} :: ratio={report.ratio:.2f} action={report.actionability:.2f} spec={report.specificity:.2f} vague={report.vague_ratio:.2f}")
        return 0
    print(f"✗ FAIL {name}")
    print(f"  ratio={report.ratio:.2f} action={report.actionability:.2f} spec={report.specificity:.2f} vague={report.vague_ratio:.2f}")
    if report.notes:
        for n in report.notes:
            print("  -", n)
    return 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="Path to a markdown file to audit")
    ap.add_argument("files", nargs="*", help="Files (from pre-commit)")
    args = ap.parse_args()

    paths = []
    if args.file:
        paths.append(args.file)
    if args.files:
        paths.extend(args.files)

    if not paths:
        print("No files provided to gate_check.", file=sys.stderr)
        return 0  # be lenient when nothing to check

    exit_code = 0
    for p in paths:
        path = pathlib.Path(p)
        if not path.exists() or not path.is_file():
            continue
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Error reading {p}: {e}", file=sys.stderr)
            exit_code = 2
            continue
        exit_code |= check_text(txt, p)

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
