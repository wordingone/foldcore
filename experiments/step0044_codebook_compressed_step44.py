#!/usr/bin/env python3
"""
Step 44: Coupling suppression diagnostic.
One variable from Step 42: add cos_suppress_thresh=0.8.
When cos(M_i, M_j) > 0.8, suppress coupling (w_ij = 0).

Setup: centroid-seeded 33 cells + mu-2sigma spawning.
Tells us: does coupling suppression prevent the maintenance collapse?

Step 42 baseline (no suppression): 33/33 -> 7/33 final coverage.
"""

import sys
import json
import math

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from rk import mcosine, frob, mrand
from fluxcore_compressed_v7 import CompressedKernel


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
    cos_suppress_thresh = 0.8
    div_names = sorted(centers.keys())
    n_divs = len(div_names)

    # Use same fixed projection as Step 42 (proj_seed=999)
    import random, math as _math
    proj_seed = 999
    random.seed(proj_seed)
    scale = 1.0 / _math.sqrt(d)
    P = [[random.gauss(0, scale) for _ in range(d)] for _ in range(k * k)]

    proj_centroids = {dn: project_vector(P, centers[dn], k) for dn in div_names}

    print(f"\nProjected {n_divs} division centroids into R^({k}x{k})")
    print(f"cos_suppress_thresh = {cos_suppress_thresh}")

    # Build kernel with centroid seeding
    kernel = CompressedKernel(n=1, k=k, d=d, seed=42, proj_seed=proj_seed,
                              tau=0.3, spawning=True, k_couple=5,
                              max_cells=500,
                              cos_suppress_thresh=cos_suppress_thresh)
    kernel.cells = [proj_centroids[dn] for dn in div_names]
    kernel.cell_surprise_ema = [0.1] * n_divs
    kernel.cell_age = [0] * n_divs
    kernel.n_min = n_divs
    kernel.total_spawned = 0

    print(f"Kernel initialized: {len(kernel.cells)} cells (one per division)")

    # Initial alignment check
    covered_init = set()
    for i, dn in enumerate(div_names):
        a = mcosine(kernel.cells[i], proj_centroids[dn])
        if abs(a) > 0.3:
            covered_init.add(dn)
    print(f"Initial coverage (before corpus): {len(covered_init)}/33")

    print(f"\n{'='*70}")
    print(f"  RUNNING CSI CORPUS (spawning=True, cos_suppress={cos_suppress_thresh})")
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

    print(f"\n  Final: cells={len(kernel.cells)}, spawned={kernel.total_spawned}, ef={kernel.mean_ef_dist():.4f}")

    # Final alignment
    print(f"\n{'='*70}")
    print(f"  ALIGNMENT AFTER CORPUS")
    print(f"{'='*70}\n")

    print(f"  {'Cell':>4}  {'Div':>4}  {'Align':>8}  {'SeedDiv':>8}  {'SameSeed?':>10}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*10}")

    covered_final = set()
    same_as_seed = 0
    n_cells = len(kernel.cells)
    for i in range(n_cells):
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
        seed_div = div_names[i] if i < len(div_names) else '(spawned)'
        same = (top_div == seed_div)
        if same and i < len(div_names):
            same_as_seed += 1
        print(f"  {i:4d}  {top_div:>4}  {top_val:>8.4f}  {seed_div:>8}  {'YES' if same else 'DRIFT':>10}")

    print(f"\n  --- Summary ---")
    print(f"    Initial coverage: {len(covered_init)}/33")
    print(f"    Final coverage:   {len(covered_final)}/33")
    print(f"    Seeded cells keeping alignment: {same_as_seed}/{n_divs}")
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
    step42_final = 7
    if len(covered_final) >= len(covered_init) - 2:
        print(f"  Coverage maintained ({len(covered_final)}/33 vs {len(covered_init)}/33 init)")
        print(f"  -> Coupling suppression FIXED the maintenance problem!")
    elif len(covered_final) > step42_final:
        print(f"  Coverage partially maintained ({len(covered_final)}/33 vs Step 42: {step42_final}/33)")
        print(f"  -> Coupling suppression PARTIALLY helps. More tuning needed.")
    else:
        print(f"  Coverage still collapsed ({len(covered_final)}/33, Step 42 baseline: {step42_final}/33)")
        print(f"  -> Coupling suppression NOT SUFFICIENT. Perception term also collapses cells.")

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
    print(f"  STEP 44 FINAL")
    print(f"{'='*70}")
    print(f"  cos_suppress_thresh: {cos_suppress_thresh}")
    print(f"  Centroid-seeded + suppression: {len(covered_final)}/33 final coverage")
    print(f"  Step 42 (seeded, no suppression): {step42_final}/33")
    print(f"  Step 38 (cold-start, no suppression): 19/33")
    print(f"  Generation: {'PASS' if survived else 'FAIL'} (energy={final_energy:.8f})")
    delta_vs_42 = len(covered_final) - step42_final
    print(f"  Delta vs Step 42: {delta_vs_42:+d}/33")


if __name__ == '__main__':
    main()
