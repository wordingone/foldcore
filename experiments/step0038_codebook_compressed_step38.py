#!/usr/bin/env python3
"""
Step 38: Sparse top-k=5 coupling — coverage test.
One variable change from Step 36: replace all-to-all coupling with top-5 coupling.
Everything else identical: mu-2sigma spawning, tau=0.3, CSI corpus, seed=42.
"""

import sys
import json
import math

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from rk import mcosine, frob
from fluxcore_compressed_v4 import CompressedKernel


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


def run_csi(records, centers, k_couple=5, n=8, tau=0.3, k=4, d=384):
    print(f"\n{'='*70}")
    print(f"  RUN: n_start={n}, tau={tau}, k_couple={k_couple}, spawning=mu-2sigma")
    print(f"{'='*70}\n")

    kernel = CompressedKernel(n=n, k=k, d=d, seed=42, proj_seed=999,
                              tau=tau, spawning=True, max_cells=500,
                              k_couple=k_couple)

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
            print(f"  [{idx+1:4d}/{total}]  cells={nc:3d}  ef_dist={me:.4f}  autonomy={ma:.4f}  energy={energy:.6f}")

    nc_final = len(kernel.cells)
    print(f"\n  Final after {total} steps:")
    print(f"    cells         = {nc_final}")
    print(f"    total_spawned = {kernel.total_spawned}")
    print(f"    mean_ef_dist  = {kernel.mean_ef_dist():.4f}")
    print(f"    mean_autonomy = {kernel.mean_autonomy():.4f}")

    # Alignment analysis
    P = kernel.P
    div_names = sorted(centers.keys())
    proj_centers = {d: project_vector(P, v, k) for d, v in centers.items()}

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

    if nc_final <= 80:
        print(f"\n  {'Cell':>4s}  {'Best Div':>8s}  {'Alignment':>10s}  {'Valid?':>6s}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*6}")
        for i in range(nc_final):
            valid = abs(best_align[i]) > threshold
            if valid:
                valid_count += 1
                covered_divs.add(best_div[i])
            tag = "YES" if valid else "no"
            print(f"  {i:4d}  {best_div[i]:>8s}  {best_align[i]:>10.4f}  {tag:>6s}")
    else:
        for i in range(nc_final):
            valid = abs(best_align[i]) > threshold
            if valid:
                valid_count += 1
                covered_divs.add(best_div[i])
        sorted_idx = sorted(range(nc_final), key=lambda i: abs(best_align[i]), reverse=True)
        print(f"\n  Top 20 cells by alignment:")
        print(f"  {'Cell':>4s}  {'Best Div':>8s}  {'Alignment':>10s}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*10}")
        for _, i in enumerate(sorted_idx[:20]):
            print(f"  {i:4d}  {best_div[i]:>8s}  {best_align[i]:>10.4f}")

    score = valid_count / nc_final if nc_final > 0 else 0
    coverage = len(covered_divs)

    print(f"\n  --- Summary ---")
    print(f"    Total cells:    {nc_final}")
    print(f"    Total spawned:  {kernel.total_spawned}")
    print(f"    Valid cells:    {valid_count}/{nc_final} = {score*100:.1f}%")
    print(f"    Division coverage: {coverage}/33 divisions")
    print(f"    (Step 36 all-to-all: 20/33 | Step 37 cos<0.5: 13/33)")
    print(f"\n    Covered: {sorted(covered_divs)}")
    missing = set(div_names) - covered_divs
    if missing:
        print(f"    Missing: {sorted(missing)}")

    return kernel, coverage, valid_count, nc_final


def generation_survival_test(kernel, label):
    print(f"\n{'='*70}")
    print(f"  GENERATION SURVIVAL TEST ({label})")
    print(f"{'='*70}\n")

    nc_start = len(kernel.cells)
    print(f"  Starting cells: {nc_start}")
    print(f"  Running 10,000 steps with r=None...\n")

    dMs = []
    for step in range(10000):
        dMs = kernel.step(r=None)
        if (step + 1) % 1000 == 0:
            nc = len(kernel.cells)
            me = kernel.mean_ef_dist()
            energy = kernel.mean_energy(dMs)
            print(f"  [step {step+1:5d}]  cells={nc:3d}  ef_dist={me:.4f}  energy={energy:.8f}")

    final_energy = kernel.mean_energy(dMs)
    nc_end = len(kernel.cells)
    survived = final_energy > 0.001
    print(f"\n  Final: cells={nc_end}, energy={final_energy:.8f}")
    print(f"  SURVIVAL: {'YES' if survived else 'NO'} (threshold=0.001)")
    return survived, final_energy, nc_end


def main():
    records, centers = load_data()
    print(f"Loaded {len(records)} records, {len(centers)} division centers")
    records.sort(key=lambda r: r['division'])

    kernel, coverage, valid_count, nc_final = run_csi(records, centers, k_couple=5)

    survived, final_energy, nc_end = generation_survival_test(kernel, "k_couple=5")

    print(f"\n{'='*70}")
    print(f"  STEP 38 FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Sparse coupling (k=5): {nc_final} cells, {valid_count} valid, {coverage}/33 coverage")
    print(f"  Generation survival: {'PASS' if survived else 'FAIL'} (energy={final_energy:.8f}, cells={nc_end})")
    print(f"  Step 36 (all-to-all): 61 cells, 60 valid, 20/33 coverage")
    print(f"  Step 37 (cos<0.5):   100 cells, ?, 13/33 coverage")
    delta = coverage - 20
    sign = "+" if delta >= 0 else ""
    print(f"  Delta vs Step 36: {sign}{delta}/33 divisions")


if __name__ == '__main__':
    main()
