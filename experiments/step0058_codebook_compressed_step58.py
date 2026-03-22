#!/usr/bin/env python3
"""
Step 58: Many-to-few n_matrix=16 sweep.

Same v17 architecture. Only change: n_matrix=16.
Hypothesis: more cells -> more coupling diversity -> stronger generation? Or diffused perception?

Baseline: Step 55 (n_matrix=8): 33/33, energy=0.081, 25s
"""

import sys
import json
import time
from collections import Counter

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
    n_matrix = 16
    records, centers = load_data()
    print(f"Loaded {len(records)} records, {len(centers)} division centers")
    records.sort(key=lambda r: r['division'])
    div_names = sorted(centers.keys())

    kernel = ManyToFewKernel(
        n_matrix=n_matrix, k=4, d=384, seed=42, proj_seed=999,
        tau=0.3, k_couple=5,
        spawn_thresh=0.5, merge_thresh=0.95, lr_codebook=0.015
    )

    print(f"\nStep 58: Many-to-few n_matrix={n_matrix}")
    print(f"  n_matrix={n_matrix} | spawn=0.5 | lr_cb=0.015 | merge>0.95")
    print(f"  Baseline: Step55 (n_matrix=8): 33/33, energy=0.081\n")
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
    if missing:
        print(f"  Missing: {sorted(missing)}")

    assign_counts = Counter(kernel.cb_assignment)
    print(f"  Matrix cell load: {dict(sorted(assign_counts.items()))}")

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
    print(f"  STEP 58 SUMMARY (n_matrix={n_matrix})")
    print(f"{'='*70}")
    print(f"  Step 55 (n_matrix=8):   33/33, energy=0.081")
    print(f"  Step 58 (n_matrix={n_matrix}): {cov}/33, energy={final_energy:.6f}")
    print(f"  Total spawned: {kernel.total_spawned}, merged: {kernel.total_merged}, final cb: {cb_final}")
    print(f"  Total elapsed: {elapsed_total:.1f}s")

    if cov == 33 and final_energy > 0.081:
        print(f"\n  -> IMPROVEMENT: 33/33 + stronger generation ({final_energy:.3f} > 0.081).")
    elif cov == 33:
        print(f"\n  -> MATCHED: 33/33, generation={final_energy:.6f}.")
    elif cov >= 30:
        print(f"\n  -> NEAR: {cov}/33, generation={final_energy:.6f}.")
    else:
        print(f"\n  -> REGRESSION: {cov}/33 (floor is 33/33).")


if __name__ == '__main__':
    main()
