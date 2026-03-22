#!/usr/bin/env python3
"""
Step 55: Many-to-few dual-rep (v17).

Architecture: separate codebook (fold's exact memory system, 300+ cheap vectors)
routes into fixed 8-cell matrix layer (RK dynamics, unchanged from v2).

Changes from v16:
- Codebook and matrix layers fully decoupled
- Matrix layer: fixed n_matrix=8, no spawning of matrices
- Codebook layer: pure fold memory (fixed 0.5 threshold, additive lr=0.015, incremental merge)
- Routing: cb winner -> assigned matrix cell gets perception (assignment at spawn time)
- No O(n^2) matrix operations

Baselines:
  Step 38: 19/33, energy=0.116  (Pareto A, v2)
  Step 48: 21/33, energy=0.063  (Pareto B, v10)
  Step 53: 20/33, energy=0.091  (v15 LVQ)
  Step 54b: STOPPED (v16 fold-faithful, O(n^2) matrix trap)
  Fold:    33/33                 (hand-tuned, threshold=0.5)
"""

import sys
import json
import math
import time

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from fluxcore_compressed_v17 import ManyToFewKernel, _vec_cosine


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

    kernel = ManyToFewKernel(
        n_matrix=8, k=4, d=384, seed=42, proj_seed=999,
        tau=0.3, k_couple=5,
        spawn_thresh=0.5, merge_thresh=0.95, lr_codebook=0.015
    )

    print(f"\nStep 55: Many-to-few dual-rep (v17)")
    print(f"  n_matrix=8 (fixed) | spawn=0.5 | lr_cb=0.015 | merge>0.95 (incremental)")
    print(f"  Baselines: Step38=19/33 e=0.116 | Step48=21/33 e=0.063 | Fold=33/33\n")
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
            cb_size = len(kernel.codebook)
            elapsed = time.time() - t0
            print(f"  [{idx+1:4d}/{total}]  cb={cb_size:3d}  ef={me:.4f}  auton={ma:.4f}  "
                  f"energy={energy:.6f}  spawned={kernel.total_spawned}  merged={kernel.total_merged}  elapsed={elapsed:.1f}s")

    cb_final = len(kernel.codebook)
    print(f"\n  Final: cb={cb_final}, spawned={kernel.total_spawned}, merged={kernel.total_merged}")

    # Coverage: codebook alignment to division centers
    threshold = 0.3
    covered_divs = set()
    for v in kernel.codebook:
        best_div, best_val = 'N/A', 0.0
        for dn in div_names:
            sim = _vec_cosine(v, centers[dn])
            if abs(sim) > abs(best_val):
                best_val, best_div = sim, dn
        if abs(best_val) > threshold:
            covered_divs.add(best_div)

    missing = set(div_names) - covered_divs
    print(f"\n  Codebook coverage: {len(covered_divs)}/33")
    print(f"  Covered: {sorted(covered_divs)}")
    if missing:
        print(f"  Missing: {sorted(missing)}")

    # Matrix cell assignment distribution
    from collections import Counter
    assign_counts = Counter(kernel.cb_assignment)
    print(f"\n  Matrix cell load: {dict(sorted(assign_counts.items()))}")

    # Generation (3000 steps)
    print(f"\n  Running generation (3000 steps, r=None)...")
    dMs = []
    for step in range(3000):
        dMs = kernel.step(r=None)
        if (step + 1) % 1000 == 0:
            energy = kernel.mean_energy(dMs)
            print(f"  [gen {step+1:4d}]  energy={energy:.6f}")
    final_energy = kernel.mean_energy(dMs)
    survived = final_energy > 0.001
    print(f"  Gen energy (3000 steps): {final_energy:.6f}  SURVIVAL: {'YES' if survived else 'NO'}")

    cov = len(covered_divs)
    elapsed_total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  STEP 55 SUMMARY")
    print(f"{'='*70}")
    print(f"  Step 38 (Pareto A):  19/33, energy=0.116")
    print(f"  Step 48 (Pareto B):  21/33, energy=0.063")
    print(f"  Step 53 (v15 LVQ):   20/33, energy=0.091")
    print(f"  Fold (hand-tuned):   33/33")
    print(f"  Step 55 (v17):       {cov}/33, energy={final_energy:.6f}")
    print(f"  Total spawned: {kernel.total_spawned}, merged: {kernel.total_merged}, final cb: {cb_final}")
    print(f"  Total elapsed: {elapsed_total:.1f}s")
    print(f"  Delta vs Step 38: {cov-19:+d}/33")
    print(f"  Delta vs Step 48: {cov-21:+d}/33")

    if cov >= 30 and final_energy > 0.05:
        print(f"\n  -> BREAKTHROUGH: {cov}/33 AND generation > 0.05.")
    elif cov >= 30:
        print(f"\n  -> HIGH COVERAGE: {cov}/33. Generation={final_energy:.6f}.")
    elif cov > 21 and final_energy > 0.05:
        print(f"\n  -> NEW PARETO: {cov}/33, energy={final_energy:.6f}.")
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
