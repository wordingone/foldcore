#!/usr/bin/env python3
"""
Step 36: CSI corpus test with dynamic cell spawning.
Tests CompressedKernel v2 (spawning) on 1920 CSI embeddings.
Run A: tau=0.3 (default), n_start=8, spawning=True
Run B: tau=0.1 (sharper coupling), n_start=8, spawning=True
Generation survival test on best run.
"""

import sys
import json
import math
import random

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from rk import mcosine, frob, mzero
from fluxcore_compressed_v2 import CompressedKernel


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


def run_experiment(records, centers, n, tau, k=4, d=384, seed=42, proj_seed=999):
    print(f"\n{'='*70}")
    print(f"  RUN: n_start={n}, tau={tau}, k={k}, spawning=True")
    print(f"{'='*70}\n")

    kernel = CompressedKernel(n=n, k=k, d=d, seed=seed, proj_seed=proj_seed,
                              tau=tau, spawning=True, max_cells=500)

    # Feed all 1920 records
    total = len(records)
    for idx, rec in enumerate(records):
        r = rec['vec']
        dMs = kernel.step(r=r)

        if (idx + 1) % 200 == 0:
            me = kernel.mean_ef_dist()
            ma = kernel.mean_autonomy()
            energy = kernel.mean_energy(dMs)
            nc = len(kernel.cells)
            print(f"  [{idx+1:4d}/{total}]  cells={nc:3d}  ef_dist={me:.4f}  autonomy={ma:.4f}  energy={energy:.6f}")

    nc_final = len(kernel.cells)

    # Final stats
    print(f"\n  Final after {total} steps:")
    print(f"    cells         = {nc_final}")
    print(f"    total_spawned = {kernel.total_spawned}")
    print(f"    mean_ef_dist  = {kernel.mean_ef_dist():.4f}")
    print(f"    mean_autonomy = {kernel.mean_autonomy():.4f}")
    print(f"    step_count    = {kernel.step_count}")

    # --- Alignment Analysis ---
    print(f"\n  --- Alignment Analysis ---\n")

    P = kernel.P

    # Project all division centers
    proj_centers = {}
    for div_name, center_vec in centers.items():
        proj_centers[div_name] = project_vector(P, center_vec, k)

    div_names = sorted(centers.keys())
    best_div = []
    best_align = []

    for i in range(nc_final):
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

    threshold = 0.3
    valid_count = 0
    covered_divs = set()

    # Print per-cell table (compact for large cell counts)
    if nc_final <= 50:
        print(f"  {'Cell':>4s}  {'Best Div':>8s}  {'Alignment':>10s}  {'Valid?':>6s}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*6}")
        for i in range(nc_final):
            valid = abs(best_align[i]) > threshold
            if valid:
                valid_count += 1
                covered_divs.add(best_div[i])
            tag = "YES" if valid else "no"
            print(f"  {i:4d}  {best_div[i]:>8s}  {best_align[i]:>10.4f}  {tag:>6s}")
    else:
        # Compact: just count
        for i in range(nc_final):
            valid = abs(best_align[i]) > threshold
            if valid:
                valid_count += 1
                covered_divs.add(best_div[i])
        # Show top 20 and bottom 20
        sorted_idx = sorted(range(nc_final), key=lambda i: abs(best_align[i]), reverse=True)
        print(f"  Top 20 cells by alignment:")
        print(f"  {'Cell':>4s}  {'Best Div':>8s}  {'Alignment':>10s}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*10}")
        for rank, i in enumerate(sorted_idx[:20]):
            print(f"  {i:4d}  {best_div[i]:>8s}  {best_align[i]:>10.4f}")

    score = valid_count / nc_final if nc_final > 0 else 0
    coverage = len(covered_divs)

    print(f"\n  --- Summary ---")
    print(f"    Total cells:    {nc_final}")
    print(f"    Total spawned:  {kernel.total_spawned}")
    print(f"    Valid cells:    {valid_count}/{nc_final} = {score*100:.1f}%")
    print(f"    Division coverage: {coverage}/33 divisions")
    print(f"    (Step 35 n=8 baseline: 7/33 divisions)")

    print(f"\n    Covered divisions ({coverage}): {sorted(covered_divs)}")
    missing = set(div_names) - covered_divs
    if missing:
        print(f"    Missing divisions ({len(missing)}): {sorted(missing)}")

    return kernel, score, coverage, valid_count, nc_final


def generation_survival_test(kernel, label):
    """Disconnect input, run 10000 steps, check energy survival."""
    print(f"\n{'='*70}")
    print(f"  GENERATION SURVIVAL TEST ({label})")
    print(f"{'='*70}\n")

    nc_start = len(kernel.cells)
    print(f"  Starting cells: {nc_start}")
    print(f"  Running 10,000 steps with r=None...\n")

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

    # Run A: tau=0.3 (default)
    kernel_a, score_a, cov_a, valid_a, nc_a = run_experiment(
        records, centers, n=8, tau=0.3)

    # Run B: tau=0.1 (sharper coupling)
    kernel_b, score_b, cov_b, valid_b, nc_b = run_experiment(
        records, centers, n=8, tau=0.1)

    # Final comparison
    print(f"\n{'='*70}")
    print(f"  COMPARISON: tau=0.3 vs tau=0.1")
    print(f"{'='*70}")
    print(f"  tau=0.3: cells={nc_a}, valid={valid_a}/{nc_a} ({score_a*100:.1f}%), coverage={cov_a}/33")
    print(f"  tau=0.1: cells={nc_b}, valid={valid_b}/{nc_b} ({score_b*100:.1f}%), coverage={cov_b}/33")
    print(f"  Step 35 baseline (n=8, no spawning): 7/33 divisions")

    # Pick best run for survival test
    if cov_a >= cov_b:
        best_label = "tau=0.3"
        best_kernel = kernel_a
    else:
        best_label = "tau=0.1"
        best_kernel = kernel_b

    print(f"\n  Best run: {best_label} (coverage={max(cov_a, cov_b)}/33)")

    survived, final_energy, nc_end = generation_survival_test(best_kernel, best_label)

    # Final summary
    print(f"\n{'='*70}")
    print(f"  STEP 36 FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Run A (tau=0.3): {nc_a} cells, {valid_a} valid, {cov_a}/33 coverage")
    print(f"  Run B (tau=0.1): {nc_b} cells, {valid_b} valid, {cov_b}/33 coverage")
    print(f"  Generation survival ({best_label}): {'PASS' if survived else 'FAIL'} (energy={final_energy:.8f}, cells={nc_end})")
    print(f"  Step 35 baseline: 7/33 divisions (fixed 8 cells)")


if __name__ == '__main__':
    main()
