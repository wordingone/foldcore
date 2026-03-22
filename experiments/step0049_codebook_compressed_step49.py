#!/usr/bin/env python3
"""
Step 49: Flipped coupling coefficient — (1-a_i) → a_i.

Original: dM_i = a_i*(Phi-M_i) + (1-a_i)*lr_i*(R-M_i) + (1-a_i)*Σw*(Psi-M_i)
Modified: dM_i = a_i*(Phi-M_i) + (1-a_i)*lr_i*(R-M_i) + a_i*Σw*(Psi-M_i)

Insight: coupling was STRONGEST during perception (low a_i) — wrong.
Flip makes coupling co-activate with eigenform drive.
- Perception (a_i≈0): only independent tracking → coverage
- Generation (a_i≈1): Phi + coupling both strong → generation preserved

Baselines:
  Step 38: 19/33, energy=0.116  (Pareto A)
  Step 48: 21/33, energy=0.063  (Pareto B)
"""

import sys
import json
import math
import time

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from rk import mcosine, frob
from fluxcore_compressed_v11 import CompressedKernel


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
    div_names = sorted(centers.keys())

    k, d = 4, 384

    kernel = CompressedKernel(
        n=8, k=k, d=d, seed=42, proj_seed=999,
        tau=0.3, spawning=True, max_cells=500, k_couple=5
    )

    print(f"\nStep 49: Flipped coupling coefficient (1-a_i) -> a_i")
    print(f"Config: Step 38 identical except coupling coefficient flipped")
    print(f"Baselines: Step38=19/33 e=0.116 | Step48=21/33 e=0.063\n")
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
    print(f"\n  Final: cells={nc_final}, spawned={kernel.total_spawned}, ef={kernel.mean_ef_dist():.4f}, auton={kernel.mean_autonomy():.4f}")

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
    missing = set(div_names) - covered_divs

    print(f"\n  Coverage: {len(covered_divs)}/33  Valid: {valid_count}/{nc_final}")
    print(f"  Covered: {sorted(covered_divs)}")
    if missing:
        print(f"  Missing: {sorted(missing)}")

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

    # Summary
    delta_38 = len(covered_divs) - 19
    delta_48 = len(covered_divs) - 21
    print(f"\n{'='*70}")
    print(f"  STEP 49 SUMMARY")
    print(f"{'='*70}")
    print(f"  Step 38 baseline:  19/33, energy=0.116")
    print(f"  Step 48 Pareto B:  21/33, energy=0.063")
    print(f"  Step 49 (v11):     {len(covered_divs)}/33, energy={final_energy:.6f}")
    print(f"  Delta vs Step 38:  {delta_38:+d}/33")
    print(f"  Delta vs Step 48:  {delta_48:+d}/33")

    if len(covered_divs) > 21 and final_energy > 0.05:
        print(f"\n  -> BREAKTHROUGH: coverage > 21/33 AND generation > 0.05. Flipped coupling is the fix.")
    elif len(covered_divs) > 21 and final_energy > 0.001:
        print(f"\n  -> COVERAGE IMPROVEMENT: {len(covered_divs)}/33 — new Pareto B candidate (generation={final_energy:.6f}).")
    elif len(covered_divs) == 21 and final_energy > 0.063:
        print(f"\n  -> PARETO IMPROVEMENT: same coverage, stronger generation ({final_energy:.6f} vs 0.063).")
    elif len(covered_divs) > 19:
        print(f"\n  -> Partial improvement. Coverage gains but energy tradeoff persists.")
    elif len(covered_divs) == 19 and final_energy > 0.116:
        print(f"\n  -> Same coverage, stronger generation. Minor Pareto A improvement.")
    else:
        print(f"\n  -> No improvement vs baselines.")


if __name__ == '__main__':
    main()
