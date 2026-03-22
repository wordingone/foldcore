#!/usr/bin/env python3
"""
Step 52: Data-seeded codebook init (one variable from Step 50).

Change from Step 50: codebook[i] = normalize(records[i]['vec']) for i in range(n).
Everything else identical to Step 50: winner-take-all codebook update,
mu-2sigma spawning on codebook cosines, winner matrix perception.

Baselines:
  Step 38: 19/33, energy=0.116  (Pareto A, v2)
  Step 48: 21/33, energy=0.063  (Pareto B, v10)
  Step 50: 20/33, energy=0.047  (v12, winner codebook, random init)
"""

import sys
import json
import math
import time

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from rk import mcosine, frob
from fluxcore_compressed_v14 import CompressedKernel, _vec_cosine


def load_data():
    with open('B:/M/avir/research/fluxcore/data/csi_embedded.json') as f:
        records = json.load(f)
    with open('B:/M/avir/research/fluxcore/data/csi_division_centers.json') as f:
        centers = json.load(f)
    return records, centers


def main():
    records, centers = load_data()
    print(f"Loaded {len(records)} records, {len(centers)} division centers")
    records.sort(key=lambda r: r['division'])
    div_names = sorted(centers.keys())

    k, d, n = 4, 384, 8

    # Seed codebook from first n records
    init_cb = [records[i]['vec'] for i in range(n)]

    kernel = CompressedKernel(
        n=n, k=k, d=d, seed=42, proj_seed=999,
        tau=0.3, spawning=True, max_cells=500, k_couple=5,
        lr_codebook=0.1, init_codebook=init_cb
    )

    print(f"\nStep 52: Data-seeded codebook init (v14)")
    print(f"  Codebook seeded from first {n} records (normalized)")
    print(f"  Winner-take-all codebook + winner matrix perception (Step 50 otherwise)")
    print(f"  Baselines: Step38=19/33 e=0.116 | Step48=21/33 e=0.063 | Step50=20/33 e=0.047\n")
    print(f"{'='*70}")
    print(f"  RUNNING CSI CORPUS")
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
            print(f"  [{idx+1:4d}/{total}]  cells={nc:3d}  ef={me:.4f}  auton={ma:.4f}  energy={energy:.6f}  elapsed={elapsed:.1f}s")

    nc_final = len(kernel.cells)
    print(f"\n  Final: cells={nc_final}, spawned={kernel.total_spawned}")

    # Coverage: codebook alignment to division centers
    best_div, best_align = [], []
    for i in range(nc_final):
        v_i = kernel.codebook[i]
        top_div, top_val = 'N/A', 0.0
        for dn in div_names:
            sim = _vec_cosine(v_i, centers[dn])
            if abs(sim) > abs(top_val):
                top_val, top_div = sim, dn
        best_div.append(top_div)
        best_align.append(top_val)

    threshold = 0.3
    valid_count = sum(1 for a in best_align if abs(a) > threshold)
    covered_divs = set(best_div[i] for i in range(nc_final) if abs(best_align[i]) > threshold)
    missing = set(div_names) - covered_divs

    print(f"\n  Codebook coverage: {len(covered_divs)}/33  Valid cells: {valid_count}/{nc_final}")
    print(f"  Covered: {sorted(covered_divs)}")
    if missing:
        print(f"  Missing: {sorted(missing)}")

    # Matrix coverage for comparison
    P = kernel.P
    def project_vec(r):
        flat = [sum(P[i][j] * r[j] for j in range(d)) for i in range(k * k)]
        return [flat[i * k:(i + 1) * k] for i in range(k)]
    proj_centers = {dn: project_vec(centers[dn]) for dn in div_names}
    mat_covered = set()
    for cell in kernel.cells:
        top_div, top_val = 'N/A', 0.0
        for dn in div_names:
            a = mcosine(cell, proj_centers[dn])
            if a != a:
                continue
            if abs(a) > abs(top_val):
                top_val, top_div = a, dn
        if abs(top_val) > threshold:
            mat_covered.add(top_div)
    print(f"  Matrix coverage (comparison): {len(mat_covered)}/33")

    # Generation (3000 steps)
    print(f"\n  Running generation (3000 steps, r=None)...")
    dMs = []
    for step in range(3000):
        dMs = kernel.step(r=None)
        if (step + 1) % 1000 == 0:
            energy = kernel.mean_energy(dMs)
            print(f"  [gen {step+1:4d}]  cells={len(kernel.cells):3d}  energy={energy:.6f}")
    final_energy = kernel.mean_energy(dMs)
    survived = final_energy > 0.001
    print(f"  Gen energy (3000 steps): {final_energy:.6f}  SURVIVAL: {'YES' if survived else 'NO'}")

    cov = len(covered_divs)
    delta_38 = cov - 19
    delta_48 = cov - 21
    print(f"\n{'='*70}")
    print(f"  STEP 52 SUMMARY")
    print(f"{'='*70}")
    print(f"  Step 38 baseline:  19/33, energy=0.116  (Pareto A)")
    print(f"  Step 48 Pareto B:  21/33, energy=0.063")
    print(f"  Step 50 (rnd init):20/33, energy=0.047")
    print(f"  Step 52 (v14):     {cov}/33, energy={final_energy:.6f}")
    print(f"  Delta vs Step 38:  {delta_38:+d}/33")
    print(f"  Delta vs Step 48:  {delta_48:+d}/33")
    print(f"  Delta vs Step 50:  {cov-20:+d}/33")

    if cov >= 30 and final_energy > 0.05:
        print(f"\n  -> BREAKTHROUGH: {cov}/33 AND generation > 0.05. Dual-rep + seeded init is the fix.")
    elif cov >= 30:
        print(f"\n  -> HIGH COVERAGE: {cov}/33. Generation={final_energy:.6f}.")
    elif cov > 21 and final_energy > 0.05:
        print(f"\n  -> NEW PARETO: {cov}/33, energy={final_energy:.6f}. Seeded init helps.")
    elif cov > 21:
        print(f"\n  -> Coverage improvement: {cov}/33 (energy={final_energy:.6f}).")
    elif cov == 21:
        print(f"\n  -> Matched Pareto B ceiling. Init helps but not breakthrough.")
    elif cov > 20:
        print(f"\n  -> Marginal improvement over Step 50 ({cov} vs 20).")
    else:
        print(f"\n  -> No improvement vs Step 50.")


if __name__ == '__main__':
    main()
