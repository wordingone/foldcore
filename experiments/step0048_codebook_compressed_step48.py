#!/usr/bin/env python3
"""
Step 48: Gaussian-weighted perception (distance-decayed, nonzero for all cells).
lr_i_eff = lr_i * exp(-||M_i - R||_F^2 / (2 * sigma^2))
sigma = self-calibrating EMA of mean cell-to-input distance.

Synthesis: Step 46 (zero perception) -> meaningless eigenforms. Steps 38/47
(flat perception) -> dominant convergence. Gaussian: weak but nonzero for
distant cells -> meaningful eigenforms, no flat convergence.

Baselines:
  Step 38: 19/33, energy=0.116 (Pareto A)
  Step 45: 21/33, energy=0.024 (Pareto B)
  Step 46: 7/33,  energy=0.124
  Step 47: 19/33, energy=0.097
"""

import sys
import json
import math
import time

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from rk import mcosine, frob
from fluxcore_compressed_v10 import CompressedKernel


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


def run_experiment(records, centers, perc_sigma, div_names):
    k, d = 4, 384
    label = f"sigma={perc_sigma:.2f}" if perc_sigma is not None else "sigma=self-calib"

    kernel = CompressedKernel(
        n=8, k=k, d=d, seed=42, proj_seed=999,
        tau=0.3, spawning=True, max_cells=500, k_couple=5,
        perc_sigma=perc_sigma
    )

    print(f"\n{'='*70}")
    print(f"  RUN: {label}")
    print(f"{'='*70}\n")

    t0 = time.time()
    total = len(records)
    last_dMs = []
    for idx, rec in enumerate(records):
        last_dMs = kernel.step(r=rec['vec'])
        if (idx + 1) % 400 == 0:
            me = kernel.mean_ef_dist()
            ma = kernel.mean_autonomy()
            energy = kernel.mean_energy(last_dMs)
            nc = len(kernel.cells)
            elapsed = time.time() - t0
            sigma_val = kernel._mean_dist_ema if perc_sigma is None else perc_sigma
            print(f"  [{idx+1:4d}/{total}]  cells={nc:3d}  ef={me:.4f}  auton={ma:.4f}  energy={energy:.6f}  sigma_eff={sigma_val:.3f}  elapsed={elapsed:.1f}s")

    nc_final = len(kernel.cells)
    sigma_final = kernel._mean_dist_ema if perc_sigma is None else perc_sigma
    print(f"\n  Final: cells={nc_final}, spawned={kernel.total_spawned}, sigma_eff={sigma_final:.4f}")

    # Alignment
    P = kernel.P
    proj_centers = {dn: project_vector(P, centers[dn], k) for dn in div_names}
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

    valid_count = sum(1 for a in best_align if abs(a) > 0.3)
    covered_divs = set(best_div[i] for i in range(nc_final) if abs(best_align[i]) > 0.3)

    print(f"  Coverage: {len(covered_divs)}/33  Valid: {valid_count}/{nc_final}")
    print(f"  Covered: {sorted(covered_divs)}")

    # Generation (3000 steps)
    dMs = []
    for step in range(3000):
        dMs = kernel.step(r=None)
    final_energy = kernel.mean_energy(dMs)
    survived = final_energy > 0.001
    print(f"  Gen energy (3000 steps): {final_energy:.6f}  SURVIVAL: {'YES' if survived else 'NO'}")

    return len(covered_divs), final_energy, sigma_final


def main():
    records, centers = load_data()
    print(f"Loaded {len(records)} records, {len(centers)} division centers")
    records.sort(key=lambda r: r['division'])
    div_names = sorted(centers.keys())

    print(f"\nStep 48: Gaussian-weighted perception")
    print(f"Baselines: Step38=19/33 e=0.116 | Step45=21/33 e=0.024 | Step46=7/33 e=0.124")

    results = []

    # Primary: self-calibrating sigma
    cov, energy, sigma = run_experiment(records, centers, None, div_names)
    results.append(('self-calib', cov, energy, sigma))

    # If primary ≥ 20/33, also test scaled variants
    if cov >= 20:
        cov2, energy2, s2 = run_experiment(records, centers, sigma * 0.5, div_names)
        results.append((f'sigma*0.5={sigma*0.5:.3f}', cov2, energy2, s2))
        cov3, energy3, s3 = run_experiment(records, centers, sigma * 2.0, div_names)
        results.append((f'sigma*2.0={sigma*2.0:.3f}', cov3, energy3, s3))

    print(f"\n{'='*70}")
    print(f"  STEP 48 SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':<20}  {'Coverage':>10}  {'Gen energy':>12}  {'Delta vs 38':>12}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*12}  {'-'*12}")
    print(f"  {'Step38 baseline':<20}  {'19/33':>10}  {'0.116':>12}  {'+0':>12}")
    print(f"  {'Step45 suppress':<20}  {'21/33':>10}  {'0.024':>12}  {'+2':>12}")
    for (label, cov, energy, sigma) in results:
        delta = f"{cov-19:+d}/33"
        print(f"  {label:<20}  {cov:>9}/33  {energy:>12.6f}  {delta:>12}")

    best_cov = max(r[1] for r in results)
    best_energy = max(r[2] for r in results)
    if best_cov > 21 and best_energy > 0.05:
        print(f"\n  -> BREAKTHROUGH: coverage > 21/33 AND generation > 0.05. Gaussian weighting is the fix.")
    elif best_cov > 19 and best_energy > 0.05:
        print(f"\n  -> IMPROVEMENT: coverage improved, generation preserved. Better than all previous.")
    elif best_cov > 19:
        print(f"\n  -> Partial: coverage improved, generation still weakened.")
    else:
        print(f"\n  -> No improvement. Coverage/generation tradeoff confirmed fundamental.")


if __name__ == '__main__':
    main()
