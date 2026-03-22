"""
embed_csi.py — CSI corpus embedding pipeline for FluxCore Step 10.

Reads combined_train.jsonl, embeds text → 384-dim unit vectors using
sentence-transformers/all-MiniLM-L6-v2, samples 2000 records evenly
across divisions, outputs:
  - data/csi_embedded.json       — 2000 records with {division, label, vec}
  - data/csi_division_centers.json — mean unit vector per division

Usage:
  python3 scripts/embed_csi.py

Install deps if needed:
  pip3 install sentence-transformers numpy
"""

import json
import math
import sys
import os

# Redirect HF cache to /tmp to avoid permission issues on shared systems
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(SCRIPT_DIR)
DATA_IN      = "/mnt/b/M/avir/data/csi-corpus/combined_train.jsonl"
DATA_OUT     = os.path.join(BASE_DIR, "data", "csi_embedded.json")
CENTERS_OUT  = os.path.join(BASE_DIR, "data", "csi_division_centers.json")

TARGET_TOTAL = 2000
MODEL_NAME   = "all-MiniLM-L6-v2"

# ── dependency check ───────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip3 install sentence-transformers numpy")
    sys.exit(1)

# ── load all records grouped by division ──────────────────────────────────
print(f"Reading {DATA_IN} ...")
by_division = defaultdict(list)
with open(DATA_IN) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        by_division[rec["division"]].append(rec)

divisions = sorted(by_division.keys())
print(f"Found {sum(len(v) for v in by_division.values())} records across {len(divisions)} divisions")

# ── sample evenly across divisions ────────────────────────────────────────
per_div = math.ceil(TARGET_TOTAL / len(divisions))
sampled = []
for div in divisions:
    recs = by_division[div]
    # take up to per_div, spread evenly through the list
    if len(recs) <= per_div:
        sampled.extend(recs)
    else:
        step = len(recs) / per_div
        sampled.extend(recs[int(i * step)] for i in range(per_div))

# trim to exactly TARGET_TOTAL
sampled = sampled[:TARGET_TOTAL]
print(f"Sampled {len(sampled)} records ({per_div} per division)")

# ── embed ──────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)

texts = [r["text"] for r in sampled]
print(f"Embedding {len(texts)} texts (batch_size=64) ...")
embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True,
                          show_progress_bar=True)
# embeddings shape: (N, 384), already L2-normalized by normalize_embeddings=True

print(f"Embedding shape: {embeddings.shape}")

# ── write csi_embedded.json ────────────────────────────────────────────────
print(f"Writing {DATA_OUT} ...")
os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)
records_out = []
for i, rec in enumerate(sampled):
    records_out.append({
        "division": rec["division"],
        "label":    rec.get("label", ""),
        "vec":      embeddings[i].tolist()
    })

with open(DATA_OUT, "w") as f:
    json.dump(records_out, f)
print(f"Wrote {len(records_out)} records to {DATA_OUT}")

# ── compute division centers ───────────────────────────────────────────────
print("Computing division centers ...")
div_vecs = defaultdict(list)
for rec in records_out:
    div_vecs[rec["division"]].append(rec["vec"])

centers = {}
for div, vecs in div_vecs.items():
    arr = np.array(vecs)           # (n, 384)
    mean = arr.mean(axis=0)        # (384,)
    norm = np.linalg.norm(mean) + 1e-12
    centers[div] = (mean / norm).tolist()

with open(CENTERS_OUT, "w") as f:
    json.dump(centers, f)
print(f"Wrote {len(centers)} division centers to {CENTERS_OUT}")

# ── summary ───────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────")
print(f"  Total records embedded : {len(records_out)}")
print(f"  Divisions              : {len(centers)}")
print(f"  Vector dimension       : {len(records_out[0]['vec'])}")
print(f"  Output                 : {DATA_OUT}")
print(f"  Centers                : {CENTERS_OUT}")
div_counts = defaultdict(int)
for r in records_out:
    div_counts[r["division"]] += 1
for div in sorted(div_counts):
    print(f"    div {div:>3s}: {div_counts[div]} records")
print("────────────────────────────────────────────────────────")
