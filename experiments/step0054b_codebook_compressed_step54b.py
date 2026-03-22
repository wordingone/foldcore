#!/usr/bin/env python3
"""
Step 54b: Fold-faithful codebook transplant into dual-rep (v16).

Three changes from v15:
1. Fixed spawn threshold = 0.5 (fold's proven mechanism for 33/33)
2. Additive codebook update: v_w = normalize(v_w + 0.015 * r)
3. Merge: |cos(vi, vj)| > 0.95 -> fuse into lower-index cell

Matrix dynamics unchanged. LVQ negative push kept.

Baselines:
  Step 38: 19/33, energy=0.116  (Pareto A)
  Step 48: 21/33, energy=0.063  (Pareto B)
  Step 53: 20/33, energy=0.091  (v15 LVQ)
  Fold:    33/33, ?.???          (hand-tuned, threshold=0.5)
"""

import sys
import json
import math
import time

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from rk import mcosine, frob
from fluxcore_compressed_v16 import CompressedKernel, _vec_cosine


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

    # Uniform stride seeding (from v15)
    stride = len(records) // n
    seed_indices = [i * stride for i in range(n)]
    init_cb = [records[i]['vec'] for i in seed_indices]
    print(f"  Seed indices: {seed_indices}")
    print(f"  Seed divisions: {[records[i]['division'] for i in seed_indices]}")

    kernel = CompressedKernel(
        n=n, k=k, d=d, seed=42, proj_seed=999,
        tau=0.3, spawning=True, max_cells=500, k_couple=5,
        lr_codebook=0.015, init_codebook=init_cb,
        neg_lr=0.01, k_neg=3,
        spawn_thresh=0.5, merge_thresh=0.95
    )

    print(f"\nStep 54b: Fold-faithful codebook (v16)")
    print(f"  Fixed spawn=0.5 | lr_cb=0.015 | merge>0.95 | LVQ neg push k=3")
    print(f"  Baselines: Step38=19/33 e=0.116 | Step48=21/33 e=0.063 | Step53=20/33 e=0.091\n")
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
            print(f"  [{idx+1:4d}/{total}]  cells={nc:3d}  ef={me:.4f}  auton={ma:.4f}  energy={energy:.6f}  spawned={kernel.total_spawned}  merged={kernel.total_merged}  elapsed={elapsed:.1f}s")

    nc_final = len(kernel.cells)
    print(f"\n  Final: cells={nc_final}, spawned={kernel.total_spawned}, merged={kernel.total_merged}")

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
    print(f"\n{'='*70}")
    print(f"  STEP 54b SUMMARY")
    print(f"{'='*70}")
    print(f"  Step 38 (Pareto A):  19/33, energy=0.116")
    print(f"  Step 48 (Pareto B):  21/33, energy=0.063")
    print(f"  Step 53 (v15 LVQ):   20/33, energy=0.091")
    print(f"  Step 54b (v16):      {cov}/33, energy={final_energy:.6f}")
    print(f"  Total spawned: {kernel.total_spawned}, merged: {kernel.total_merged}, final: {nc_final}")
    print(f"  Delta vs Step 38:    {cov-19:+d}/33")
    print(f"  Delta vs Step 48:    {cov-21:+d}/33")

    if cov >= 30 and final_energy > 0.05:
        print(f"\n  -> BREAKTHROUGH: {cov}/33 AND generation > 0.05.")
    elif cov >= 30:
        print(f"\n  -> HIGH COVERAGE: {cov}/33. Generation={final_energy:.6f}.")
    elif cov > 21 and final_energy > 0.05:
        print(f"\n  -> NEW PARETO: {cov}/33, energy={final_energy:.6f}. Fold codebook works.")
    elif cov > 21:
        print(f"\n  -> Coverage past 21/33 ceiling: {cov}/33.")
    elif cov == 21:
        print(f"\n  -> Matched Pareto B. Generation: {final_energy:.6f}.")
    elif cov > 19:
        print(f"\n  -> Marginal improvement ({cov}/33).")
    else:
        print(f"\n  -> No improvement vs baseline.")


if __name__ == '__main__':
    main()
