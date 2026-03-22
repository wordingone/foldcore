#!/usr/bin/env python3
"""
Step 42: Centroid-seeded initialization — diagnostic.
Pre-seed kernel with 33 projected division centroids (one per CSI division).
Run full corpus. Report: divisions maintained vs collapsed.
Tells us if problem is DISCOVERY (spawning) or MAINTENANCE (coupling).
"""

import sys
import json
import math

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from rk import mcosine, frob, mrand
from fluxcore_compressed_v2 import CompressedKernel


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


def main():
    records, centers = load_data()
    print(f"Loaded {len(records)} records, {len(centers)} division centers")
    records.sort(key=lambda r: r['division'])

    k, d = 4, 384
    div_names = sorted(centers.keys())
    n_divs = len(div_names)

    # Build a kernel just to get the fixed projection P
    import random, math as _math
    proj_seed = 999
    random.seed(proj_seed)
    scale = 1.0 / _math.sqrt(d)
    P = [[random.gauss(0, scale) for _ in range(d)] for _ in range(k * k)]

    # Project all 33 centroids into k×k space
    proj_centroids = {dn: project_vector(P, centers[dn], k) for dn in div_names}

    print(f"\nProjected {n_divs} division centroids into R^({k}x{k})")

    # Build kernel with n=1 placeholder, then replace cells with centroids
    kernel = CompressedKernel(n=1, k=k, d=d, seed=42, proj_seed=proj_seed,
                              tau=0.3, spawning=False, k_couple=5)
    # Replace cells with the 33 projected centroids
    kernel.cells = [proj_centroids[dn] for dn in div_names]
    kernel.cell_surprise_ema = [0.1] * n_divs
    kernel.cell_age = [0] * n_divs
    kernel.n_min = n_divs  # never prune below 33
    kernel.total_spawned = 0

    print(f"Kernel initialized: {len(kernel.cells)} cells (one per division)")
    print(f"Verifying initial alignment...")

    # Initial alignment check
    covered_init = set()
    for i, dn in enumerate(div_names):
        a = mcosine(kernel.cells[i], proj_centroids[dn])
        if abs(a) > 0.3:
            covered_init.add(dn)
    print(f"Initial coverage (before corpus): {len(covered_init)}/33")

    print(f"\n{'='*70}")
    print(f"  RUNNING CSI CORPUS (spawning=False, 33 centroid cells)")
    print(f"{'='*70}\n")

    total = len(records)
    last_dMs = []
    for idx, rec in enumerate(records):
        last_dMs = kernel.step(r=rec['vec'])

        if (idx + 1) % 200 == 0:
            me = kernel.mean_ef_dist()
            ma = kernel.mean_autonomy()
            energy = kernel.mean_energy(last_dMs)
            nc = len(kernel.cells)
            print(f"  [{idx+1:4d}/{total}]  cells={nc:2d}  ef={me:.4f}  auton={ma:.4f}  energy={energy:.6f}")

    print(f"\n  Final: cells={len(kernel.cells)}, ef={kernel.mean_ef_dist():.4f}")

    # Final alignment
    print(f"\n{'='*70}")
    print(f"  ALIGNMENT AFTER CORPUS")
    print(f"{'='*70}\n")

    print(f"  {'Cell':>4}  {'Div':>4}  {'Align':>8}  {'SeedDiv':>8}  {'SameSeed?':>10}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*10}")

    covered_final = set()
    same_as_seed = 0
    for i in range(len(kernel.cells)):
        cell = kernel.cells[i]
        top_div, top_val = 'N/A', 0.0
        for dn in div_names:
            a = mcosine(cell, proj_centroids[dn])
            if a != a:
                continue
            if abs(a) > abs(top_val):
                top_val, top_div = a, dn
        valid = abs(top_val) > 0.3
        if valid:
            covered_final.add(top_div)
        seed_div = div_names[i] if i < len(div_names) else '?'
        same = (top_div == seed_div)
        if same:
            same_as_seed += 1
        print(f"  {i:4d}  {top_div:>4}  {top_val:>8.4f}  {seed_div:>8}  {'YES' if same else 'DRIFT':>10}")

    print(f"\n  --- Summary ---")
    print(f"    Initial coverage: {len(covered_init)}/33")
    print(f"    Final coverage:   {len(covered_final)}/33")
    print(f"    Cells that kept seed alignment: {same_as_seed}/{n_divs}")
    print(f"    Delta: {len(covered_final) - len(covered_init):+d}/33")
    print(f"\n    Covered: {sorted(covered_final)}")
    missing = set(div_names) - covered_final
    if missing:
        print(f"    Missing: {sorted(missing)}")

    # Diagnosis
    print(f"\n{'='*70}")
    print(f"  DIAGNOSIS")
    print(f"{'='*70}")
    delta = len(covered_final) - len(covered_init)
    if delta >= -2:
        print(f"  Coverage maintained ({len(covered_final)}/33 vs {len(covered_init)}/33 init)")
        print(f"  -> Problem is DISCOVERY: spawning can't find proximate divisions")
        print(f"  -> Fix: better spawning (not architecture change)")
    else:
        print(f"  Coverage dropped ({len(covered_init)} -> {len(covered_final)})")
        print(f"  -> Problem is MAINTENANCE: coupling collapses proximate cells")
        print(f"  -> Fix: architecture change (repulsion, local coupling, or larger k)")

    # Generation survival
    print(f"\n{'='*70}")
    print(f"  GENERATION SURVIVAL (3000 steps)")
    print(f"{'='*70}\n")
    dMs = []
    for step in range(3000):
        dMs = kernel.step(r=None)
        if (step + 1) % 1000 == 0:
            energy = kernel.mean_energy(dMs)
            print(f"  [step {step+1:5d}]  cells={len(kernel.cells)}  energy={energy:.8f}")
    final_energy = kernel.mean_energy(dMs)
    survived = final_energy > 0.001
    print(f"\n  Final energy: {final_energy:.8f}")
    print(f"  SURVIVAL: {'YES' if survived else 'NO'}")

    print(f"\n{'='*70}")
    print(f"  STEP 42 FINAL")
    print(f"{'='*70}")
    print(f"  Centroid-seeded: {len(covered_final)}/33 final coverage")
    print(f"  Generation: {'PASS' if survived else 'FAIL'} (energy={final_energy:.8f})")
    print(f"  Step 38 (cold-start): 19/33 coverage")
    print(f"  Diagnosis: {'DISCOVERY' if delta >= -2 else 'MAINTENANCE'} problem")


if __name__ == '__main__':
    main()
