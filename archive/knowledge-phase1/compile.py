#!/usr/bin/env python
"""
Knowledge Compiler -- quick stats and integrity check on the unified knowledge base.

Usage:
    python knowledge/compile.py
"""
import json
from pathlib import Path
from collections import Counter

KNOWLEDGE_DIR = Path(__file__).parent
ENTRIES_DIR = KNOWLEDGE_DIR / "entries"


def main():
    # Load entries
    entries = []
    for f in sorted(ENTRIES_DIR.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            entries.append(json.load(fh))

    # Load constraints
    constraints_path = KNOWLEDGE_DIR / "constraints.json"
    with open(constraints_path, "r", encoding="utf-8") as fh:
        constraints = json.load(fh)

    # Stats
    types = Counter(e.get("type", "unknown") for e in entries)
    statuses = Counter(e.get("status", "unknown") for e in entries)
    sessions = sorted(set(e.get("session", 0) for e in entries if e.get("session")))
    tags_all = Counter()
    for e in entries:
        for t in e.get("tags", []):
            tags_all[t] += 1

    ss_constraints = [c for c in constraints if not c["id"].startswith("fc")]
    fc_constraints = [c for c in constraints if c["id"].startswith("fc")]

    print("=== Knowledge Base Stats ===")
    print(f"Entries: {len(entries)}")
    print(f"  By type: {dict(types)}")
    print(f"  By status: {dict(statuses)}")
    print(f"  Sessions: {min(sessions)}-{max(sessions)} ({len(sessions)} total)")
    print(f"Constraints: {len(constraints)} total")
    print(f"  Singularity Search (c001-c051): {len(ss_constraints)}")
    print(f"  FoldCore (fc001-fc015): {len(fc_constraints)}")
    print(f"Top tags: {dict(tags_all.most_common(10))}")

    # Integrity checks
    ids = [e["id"] for e in entries]
    dupes = [i for i in ids if ids.count(i) > 1]
    if dupes:
        print(f"\nWARNING: Duplicate entry IDs: {set(dupes)}")
    else:
        print(f"\nIntegrity: OK (no duplicate IDs)")


if __name__ == "__main__":
    main()
