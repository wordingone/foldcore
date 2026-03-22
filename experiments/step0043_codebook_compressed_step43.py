#!/usr/bin/env python3
"""
Step 43: k=10 experiment — testing whether higher-dimensional cells (100 params vs 16)
reduce coupling collapse and improve CSI division coverage.

Hypothesis: k=4 (16 params) is too lossy for 384-dim space. k=10 (100 params)
gives cells enough representational room to stay distinct under coupling pressure.

Config identical to Step 38 EXCEPT k=4 -> k=10:
  k=10, d=384, n=8, seed=42, proj_seed=999, tau=0.3, spawning=True, max_cells=500, k_couple=5
"""

import sys
import json
import math
import time

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


def run_csi(records, centers, k_val, label):
    print(f"\n{'='*70}")
    print(f"  RUN: k={k_val}, n=8, tau=0.3, k_couple=5, spawning=mu-2sigma")
    print(f"  {label}")
    print(f"{'='*70}\n")

    t0 = time.time()
    kernel = CompressedKernel(n=8, k=k_val, d=384, seed=42, proj_seed=999,
                              tau=0.3, spawning=True, max_cells=500,
                              k_couple=5)

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
            elapsed = time.time() - t0
            print(f"  [{idx+1:4d}/{total}]  cells={nc:3d}  ef_dist={me:.4f}  autonomy={ma:.4f}  energy={energy:.6f}  elapsed={elapsed:.1f}s")

    nc_final = len(kernel.cells)
    elapsed = time.time() - t0
    print(f"\n  Final after {total} steps ({elapsed:.1f}s):")
    print(f"    cells         = {nc_final}")
    print(f"    total_spawned = {kernel.total_spawned}")
    print(f"    mean_ef_dist  = {kernel.mean_ef_dist():.4f}")
    print(f"    mean_autonomy = {kernel.mean_autonomy():.4f}")

    # Alignment analysis
    P = kernel.P
    div_names = sorted(centers.keys())
    proj_centers = {d: project_vector(P, v, k_val) for d, v in centers.items()}

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
    print(f"\n    Covered: {sorted(covered_divs)}")
    missing = set(div_names) - covered_divs
    if missing:
        print(f"    Missing: {sorted(missing)}")

    return kernel, coverage, valid_count, nc_final


def generation_survival_test(kernel, label, steps=5000):
    print(f"\n{'='*70}")
    print(f"  GENERATION SURVIVAL TEST ({label})")
    print(f"{'='*70}\n")

    nc_start = len(kernel.cells)
    print(f"  Starting cells: {nc_start}")
    print(f"  Running {steps} steps with r=None...\n")

    t0 = time.time()
    dMs = []
    for step in range(steps):
        dMs = kernel.step(r=None)
        if (step + 1) % 1000 == 0:
            nc = len(kernel.cells)
            me = kernel.mean_ef_dist()
            energy = kernel.mean_energy(dMs)
            elapsed = time.time() - t0
            print(f"  [step {step+1:5d}]  cells={nc:3d}  ef_dist={me:.4f}  energy={energy:.8f}  elapsed={elapsed:.1f}s")

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

    # --- k=10 experiment ---
    kernel_10, cov_10, valid_10, nc_10 = run_csi(records, centers, k_val=10, label="Step 43: k=10 hypothesis test")
    surv_10, energy_10, nc_end_10 = generation_survival_test(kernel_10, "k=10", steps=5000)

    print(f"\n{'='*70}")
    print(f"  STEP 43 RESULTS — k=10")
    print(f"{'='*70}")
    print(f"  k=10: {nc_10} cells, {valid_10} valid, {cov_10}/33 coverage")
    print(f"  Generation survival: {'PASS' if surv_10 else 'FAIL'} (energy={energy_10:.8f}, cells={nc_end_10})")
    print(f"  Step 38 baseline (k=4): 54 cells, 19/33 coverage, energy=0.116")
    delta_10 = cov_10 - 19
    sign_10 = "+" if delta_10 >= 0 else ""
    print(f"  Delta vs Step 38: {sign_10}{delta_10}/33 divisions")

    # --- If k=10 improves significantly (>= +3), also test k=16 ---
    if cov_10 >= 22:
        print(f"\n  k=10 improved by {delta_10} divisions — testing k=16...")

        # Reload fresh data
        records2, centers2 = load_data()
        records2.sort(key=lambda r: r['division'])

        kernel_16, cov_16, valid_16, nc_16 = run_csi(records2, centers2, k_val=16, label="Step 43b: k=16 extended test")
        surv_16, energy_16, nc_end_16 = generation_survival_test(kernel_16, "k=16", steps=5000)

        print(f"\n{'='*70}")
        print(f"  STEP 43 FINAL COMPARISON")
        print(f"{'='*70}")
        print(f"  Step 38 (k=4):  54 cells, 19/33 coverage, energy=0.116")
        print(f"  Step 43 (k=10): {nc_10} cells, {cov_10}/33 coverage, energy={energy_10:.8f}")
        print(f"  Step 43 (k=16): {nc_16} cells, {cov_16}/33 coverage, energy={energy_16:.8f}")
        print(f"  Delta k=10 vs baseline: {sign_10}{delta_10}")
        delta_16 = cov_16 - 19
        sign_16 = "+" if delta_16 >= 0 else ""
        print(f"  Delta k=16 vs baseline: {sign_16}{delta_16}")
    else:
        print(f"\n{'='*70}")
        print(f"  STEP 43 FINAL")
        print(f"{'='*70}")
        print(f"  k=10 did NOT improve significantly ({sign_10}{delta_10}). Skipping k=16.")
        print(f"  Step 38 (k=4):  54 cells, 19/33 coverage, energy=0.116")
        print(f"  Step 43 (k=10): {nc_10} cells, {cov_10}/33 coverage, energy={energy_10:.8f}")


if __name__ == '__main__':
    main()
