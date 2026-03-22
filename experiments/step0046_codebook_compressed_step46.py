#!/usr/bin/env python3
"""
Step 46: Sparse perception routing (top-3 cells receive each input).
One variable from Step 38: add k_perceive=3.
Coupling unchanged — preserves inter-cell drive and generation energy.

Hypothesis: broadcasting perception to all cells pulls proximate divisions
toward dominant inputs. Routing to top-3 only lets non-winning cells
maintain their eigenform → stay differentiated.

Step 38: 19/33, energy=0.116
Step 45: 21/33, energy=0.024 (coupling suppression — weakens generation)
"""

import sys
import json
import math
import time

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from rk import mcosine, frob
from fluxcore_compressed_v8 import CompressedKernel


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
    k_perceive = 3
    div_names = sorted(centers.keys())

    kernel = CompressedKernel(
        n=8, k=k, d=d, seed=42, proj_seed=999,
        tau=0.3, spawning=True, max_cells=500, k_couple=5,
        k_perceive=k_perceive
    )

    print(f"Config: n=8, k={k}, d={d}, tau=0.3, k_couple=5, k_perceive={k_perceive}")
    print(f"(Step 38 identical except k_perceive={k_perceive})\n")

    print(f"{'='*70}")
    print(f"  RUNNING CSI CORPUS")
    print(f"{'='*70}\n")

    t0 = time.time()
    total = len(records)
    last_dMs = []
    for idx, rec in enumerate(records):
        last_dMs = kernel.step(r=rec['vec'])
        if (idx + 1) % 200 == 0:
            me = kernel.mean_ef_dist()
            ma = kernel.mean_autonomy()
            energy = kernel.mean_energy(last_dMs)
            nc = len(kernel.cells)
            elapsed = time.time() - t0
            print(f"  [{idx+1:4d}/{total}]  cells={nc:3d}  ef={me:.4f}  auton={ma:.4f}  energy={energy:.6f}  elapsed={elapsed:.1f}s")

    nc_final = len(kernel.cells)
    print(f"\n  Final: cells={nc_final}, spawned={kernel.total_spawned}, ef={kernel.mean_ef_dist():.4f}")

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

    threshold = 0.3
    valid_count = 0
    covered_divs = set()
    for i in range(nc_final):
        if abs(best_align[i]) > threshold:
            valid_count += 1
            covered_divs.add(best_div[i])

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Total cells:    {nc_final}")
    print(f"  Total spawned:  {kernel.total_spawned}")
    print(f"  Valid cells:    {valid_count}/{nc_final} = {valid_count/nc_final*100:.1f}%")
    print(f"  Division coverage: {len(covered_divs)}/33")
    print(f"  Step 38 baseline:  19/33 (energy=0.116)")
    print(f"  Step 45 (suppress): 21/33 (energy=0.024)")
    delta_38 = len(covered_divs) - 19
    delta_45 = len(covered_divs) - 21
    print(f"  Delta vs Step 38:  {delta_38:+d}/33")
    print(f"  Delta vs Step 45:  {delta_45:+d}/33")
    print(f"\n  Covered: {sorted(covered_divs)}")
    missing = set(div_names) - covered_divs
    if missing:
        print(f"  Missing: {sorted(missing)}")

    # Generation survival
    print(f"\n{'='*70}")
    print(f"  GENERATION SURVIVAL (5000 steps)")
    print(f"{'='*70}\n")
    dMs = []
    for step in range(5000):
        dMs = kernel.step(r=None)
        if (step + 1) % 1000 == 0:
            energy = kernel.mean_energy(dMs)
            print(f"  [step {step+1:5d}]  cells={len(kernel.cells):3d}  energy={energy:.8f}")
    final_energy = kernel.mean_energy(dMs)
    survived = final_energy > 0.001
    print(f"\n  Final energy: {final_energy:.8f}")
    print(f"  SURVIVAL: {'YES' if survived else 'NO'}")

    print(f"\n{'='*70}")
    print(f"  STEP 46 FINAL")
    print(f"{'='*70}")
    print(f"  k_perceive={k_perceive} (sparse perception): {len(covered_divs)}/33")
    print(f"  Step 38 (broadcast perception): 19/33, energy=0.116")
    print(f"  Step 45 (coupling suppress):    21/33, energy=0.024")
    print(f"  Generation: {'PASS' if survived else 'FAIL'} (energy={final_energy:.8f})")
    print(f"  Delta vs Step 38: {delta_38:+d}/33")
    if delta_38 > 0 and final_energy > 0.05:
        print(f"  -> IMPROVEMENT: more coverage AND strong generation. Sparse perception is the fix.")
    elif delta_38 > 0:
        print(f"  -> Coverage improved but generation weakened. Mixed result.")
    elif delta_38 == 0:
        print(f"  -> No improvement. Perception routing doesn't help coverage.")
    else:
        print(f"  -> Coverage worsened. Perception routing hurts.")


if __name__ == '__main__':
    main()
