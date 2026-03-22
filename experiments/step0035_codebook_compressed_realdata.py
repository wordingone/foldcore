#!/usr/bin/env python3
"""
Step 35: CSI real-data compressed kernel test.
Tests CompressedKernel on 1920 CSI embeddings with n=64 and n=8 cells.
"""

import sys
import json
import math
import random

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from rk import mcosine, frob
from fluxcore_compressed import CompressedKernel


def load_data():
    with open('B:/M/avir/research/fluxcore/data/csi_embedded.json') as f:
        records = json.load(f)
    with open('B:/M/avir/research/fluxcore/data/csi_division_centers.json') as f:
        centers = json.load(f)
    return records, centers


def project_vector(P, r, k):
    """Project r in R^d -> R^(k x k) via P @ r reshaped."""
    d = len(r)
    flat = [sum(P[i][j] * r[j] for j in range(d)) for i in range(k * k)]
    return [flat[i * k:(i + 1) * k] for i in range(k)]


def run_experiment(records, centers, n, k=4, d=384, seed=42, proj_seed=999):
    print(f"\n{'='*70}")
    print(f"  RUN: n={n} cells, k={k}, d={d}")
    print(f"{'='*70}\n")

    kernel = CompressedKernel(n=n, k=k, d=d, seed=seed, proj_seed=proj_seed)

    # Feed all 1920 records
    total = len(records)
    for idx, rec in enumerate(records):
        r = rec['vec']
        dMs = kernel.step(r=r)

        if (idx + 1) % 200 == 0:
            me = kernel.mean_ef_dist()
            ma = kernel.mean_autonomy()
            energy = kernel.mean_energy(dMs)
            print(f"  [{idx+1:4d}/{total}]  ef_dist={me:.4f}  autonomy={ma:.4f}  energy={energy:.6f}")

    # Final stats
    print(f"\n  Final after {total} steps:")
    print(f"    mean_ef_dist  = {kernel.mean_ef_dist():.4f}")
    print(f"    mean_autonomy = {kernel.mean_autonomy():.4f}")
    print(f"    step_count    = {kernel.step_count}")

    # --- Post-processing analysis ---
    print(f"\n  --- Alignment Analysis ---\n")

    # Build projection matrix (same as kernel's internal one)
    P = kernel.P

    # Project all division centers
    proj_centers = {}
    for div_name, center_vec in centers.items():
        proj_centers[div_name] = project_vector(P, center_vec, k)

    # For each cell, find best division alignment
    div_names = sorted(centers.keys())
    best_div = []
    best_align = []

    for i in range(n):
        cell_M = kernel.cells[i]
        top_div = "N/A"
        top_val = 0.0
        for dname in div_names:
            a = mcosine(cell_M, proj_centers[dname])
            if a != a:  # NaN check
                continue
            if abs(a) > abs(top_val):
                top_val = a
                top_div = dname
        best_div.append(top_div)
        best_align.append(top_val)

    # Print per-cell table
    threshold = 0.3
    valid_count = 0
    covered_divs = set()

    print(f"  {'Cell':>4s}  {'Best Div':>8s}  {'Alignment':>10s}  {'Valid?':>6s}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*6}")
    for i in range(n):
        valid = abs(best_align[i]) > threshold
        if valid:
            valid_count += 1
            covered_divs.add(best_div[i])
        tag = "YES" if valid else "no"
        print(f"  {i:4d}  {best_div[i]:>8s}  {best_align[i]:>10.4f}  {tag:>6s}")

    score = valid_count / n
    coverage = len(covered_divs)

    print(f"\n  --- Summary ---")
    print(f"    Valid cells: {valid_count}/{n} = {score*100:.1f}%")
    print(f"    Division coverage: {coverage}/33 divisions")
    print(f"    (Fold baseline: 357/359 = 99.4%)")

    # Show which divisions are covered
    print(f"\n    Covered divisions ({coverage}): {sorted(covered_divs)}")
    missing = set(div_names) - covered_divs
    if missing:
        print(f"    Missing divisions ({len(missing)}): {sorted(missing)}")

    return score, coverage


def main():
    records, centers = load_data()
    print(f"Loaded {len(records)} records, {len(centers)} division centers")

    # Sort records by division (they should already be sorted, but ensure it)
    records.sort(key=lambda r: r['division'])

    # Run A: n=64
    score_64, cov_64 = run_experiment(records, centers, n=64)

    # Run B: n=8
    score_8, cov_8 = run_experiment(records, centers, n=8)

    # Final comparison
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"  n=64: score={score_64*100:.1f}% ({int(score_64*64)}/{64}), coverage={cov_64}/33")
    print(f"  n=8:  score={score_8*100:.1f}% ({int(score_8*8)}/{8}), coverage={cov_8}/33")
    print(f"  Fold baseline: 99.4% (357/359)")


if __name__ == '__main__':
    main()
