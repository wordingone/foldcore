#!/usr/bin/env python3
"""
Step 41: Cell splitting for coverage recovery.
One variable from Step 38: add split_thresh=0.3, split_eps=0.1.
Same: top-k=5 coupling, mu-2sigma spawning, tau=0.3, CSI corpus, seed=42.
"""

import sys
import json
import math

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from rk import mcosine, frob
from fluxcore_compressed_v5 import CompressedKernel


def load_data():
    with open('B:/M/avir/research/fluxcore/data/csi_embedded.json') as f:
        records = json.load(f)
    with open('B:/M/avir/research/fluxcore/data/csi_division_centers.json') as f:
        centers = json.load(f)
    return records, centers


def project_vector(P, r, k):
    d = len(r)
    flat = [sum(P[i][j] * r[j] for j in range(d)) for i in range(k * k)]
    return [flat[i * k:(i + 1) * k] for i in range(k)]


def run_csi(records, centers, split_thresh=0.3, split_eps=0.1):
    print(f"\n{'='*70}")
    print(f"  RUN: split_thresh={split_thresh}, split_eps={split_eps}, k_couple=5")
    print(f"{'='*70}\n")

    kernel = CompressedKernel(n=8, k=4, d=384, seed=42, proj_seed=999,
                              tau=0.3, spawning=True, max_cells=500,
                              k_couple=5, split_thresh=split_thresh,
                              split_eps=split_eps, split_min_age=50)

    total = len(records)
    last_dMs = []
    for idx, rec in enumerate(records):
        r = rec['vec']
        last_dMs = kernel.step(r=r)

        if (idx + 1) % 200 == 0:
            me = kernel.mean_ef_dist()
            ma = kernel.mean_autonomy()
            energy = kernel.mean_energy(last_dMs)
            nc = len(kernel.cells)
            print(f"  [{idx+1:4d}/{total}]  cells={nc:3d}  split={kernel.total_split:2d}  "
                  f"ef={me:.4f}  auton={ma:.4f}  energy={energy:.6f}")

    nc_final = len(kernel.cells)
    print(f"\n  Final after {total} steps:")
    print(f"    cells         = {nc_final}")
    print(f"    total_spawned = {kernel.total_spawned}")
    print(f"    total_split   = {kernel.total_split}")
    print(f"    mean_ef_dist  = {kernel.mean_ef_dist():.4f}")
    print(f"    mean_autonomy = {kernel.mean_autonomy():.4f}")

    # Alignment
    P = kernel.P
    div_names = sorted(centers.keys())
    proj_centers = {d: project_vector(P, v, 4) for d, v in centers.items()}

    best_div, best_align = [], []
    for cell in kernel.cells:
        top_div, top_val = 'N/A', 0.0
        for dn in div_names:
            a = mcosine(cell, proj_centers[dn])
            if a != a:
                continue
            if abs(a) > abs(top_val):
                top_val, top_div = a, dn
        best_div.append(top_div)
        best_align.append(top_val)

    threshold = 0.3
    valid_count = 0
    covered_divs = set()
    for i in range(nc_final):
        valid = abs(best_align[i]) > threshold
        if valid:
            valid_count += 1
            covered_divs.add(best_div[i])

    score = valid_count / nc_final if nc_final > 0 else 0
    coverage = len(covered_divs)

    # Print top 20
    sorted_idx = sorted(range(nc_final), key=lambda i: abs(best_align[i]), reverse=True)
    print(f"\n  Top 20 cells:")
    print(f"  {'Cell':>4}  {'Div':>4}  {'Align':>8}")
    for _, i in enumerate(sorted_idx[:20]):
        print(f"  {i:4d}  {best_div[i]:>4}  {best_align[i]:>8.4f}")

    print(f"\n  --- Summary ---")
    print(f"    Cells: {nc_final}  Spawned: {kernel.total_spawned}  Split: {kernel.total_split}")
    print(f"    Valid: {valid_count}/{nc_final} = {score*100:.1f}%")
    print(f"    Division coverage: {coverage}/33")
    print(f"    (Step 38 baseline: 54 cells, 19/33 coverage, 0 splits)")
    print(f"\n    Covered: {sorted(covered_divs)}")
    missing = set(div_names) - covered_divs
    if missing:
        print(f"    Missing: {sorted(missing)}")

    return kernel, coverage, valid_count, nc_final


def generation_survival_test(kernel):
    print(f"\n{'='*70}")
    print(f"  GENERATION SURVIVAL TEST")
    print(f"{'='*70}\n")
    print(f"  Cells: {len(kernel.cells)}, running 5000 steps r=None...\n")

    dMs = []
    for step in range(5000):
        dMs = kernel.step(r=None)
        if (step + 1) % 1000 == 0:
            energy = kernel.mean_energy(dMs)
            print(f"  [step {step+1:5d}]  cells={len(kernel.cells):3d}  energy={energy:.8f}")

    final_energy = kernel.mean_energy(dMs)
    survived = final_energy > 0.001
    print(f"\n  Final: cells={len(kernel.cells)}, energy={final_energy:.8f}")
    print(f"  SURVIVAL: {'YES' if survived else 'NO'}")
    return survived, final_energy


def main():
    records, centers = load_data()
    print(f"Loaded {len(records)} records, {len(centers)} division centers")
    records.sort(key=lambda r: r['division'])

    kernel, coverage, valid_count, nc_final = run_csi(records, centers)
    survived, final_energy = generation_survival_test(kernel)

    print(f"\n{'='*70}")
    print(f"  STEP 41 FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Splitting (thresh=0.3): {nc_final} cells, {valid_count} valid, {coverage}/33 coverage")
    print(f"  Generation: {'PASS' if survived else 'FAIL'} (energy={final_energy:.8f})")
    print(f"  Step 38 (no split):  54 cells, 53 valid, 19/33, energy=0.116")
    delta = coverage - 19
    sign = "+" if delta >= 0 else ""
    print(f"  Delta vs Step 38: {sign}{delta}/33 divisions")


if __name__ == '__main__':
    main()
