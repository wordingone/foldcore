#!/usr/bin/env python3
"""
Step 31: RK Autoregression Experiment
Does the Reflexive Kernel sustain non-trivial dynamics without external signal?
"""

import sys
sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')

from rk import Kernel

def run_experiment(alpha, beta, seed=42, label=""):
    print(f"\n{'='*72}")
    print(f"  RK AUTOREGRESSION -- alpha={alpha}, beta={beta}, seed={seed}")
    if label:
        print(f"  {label}")
    print(f"{'='*72}\n")

    K = Kernel(n=8, k=4, alpha=alpha, beta=beta, seed=seed)

    # Phase 1: 2000 steps seeding (no signal)
    print("  Phase 1: Seeding (2000 steps, no signal)")
    for _ in range(2000):
        K.step()
    m = K.measure()
    print(f"  After seeding: ef={m['ef']:.4f}  au={m['au']:.4f}  E={m['E']:.6f}  nrm={m['nrm']:.4f}\n")

    # Phase 2: Log every 200 steps from 2000 to 12000
    print("  Phase 2: Autoregression monitoring (steps 2000-12000)")
    hdr = f"  {'step':>6}  {'mean_ef_dist':>12}  {'mean_autonomy':>14}  {'mean_energy':>12}  {'mean_norm':>10}  {'avg_offdiag_align':>18}"
    print(hdr)
    print(f"  {'-'*6}  {'-'*12}  {'-'*14}  {'-'*12}  {'-'*10}  {'-'*18}")

    measurements = []
    for i in range(50):
        for _ in range(200):
            K.step()

        # Compute avg absolute off-diagonal alignment
        A = K.alignment_matrix()
        n = K.n
        offdiag_vals = []
        for ii in range(n):
            for jj in range(n):
                if ii != jj:
                    offdiag_vals.append(abs(A[ii][jj]))
        avg_offdiag = sum(offdiag_vals) / len(offdiag_vals) if offdiag_vals else 0.0

        ef = K.mean_ef_dist()
        au = K.mean_autonomy()
        en = K.mean_energy()
        nm = K.mean_norm()

        measurements.append({
            'step': K.step_count,
            'ef': ef, 'au': au, 'en': en, 'nm': nm, 'align': avg_offdiag
        })

        print(f"  {K.step_count:>6}  {ef:>12.6f}  {au:>14.6f}  {en:>12.8f}  {nm:>10.4f}  {avg_offdiag:>18.6f}")

    # Classification
    final_energy = measurements[-1]['en']
    print(f"\n  Final energy at step {K.step_count}: {final_energy:.8f}")

    if final_energy < 0.0001:
        classification = "FIXED POINT (at noise floor)"
    elif final_energy > 0.001:
        classification = "SUSTAINED DYNAMICS"
    else:
        classification = "BORDERLINE"

    print(f"  Classification: {classification}\n")

    # Summary stats
    energies = [m['en'] for m in measurements]
    ef_dists = [m['ef'] for m in measurements]
    print(f"  Energy range: [{min(energies):.8f}, {max(energies):.8f}]")
    print(f"  EF dist range: [{min(ef_dists):.6f}, {max(ef_dists):.6f}]")
    print(f"  Energy mean: {sum(energies)/len(energies):.8f}")
    print(f"  Energy std: {(sum((e - sum(energies)/len(energies))**2 for e in energies)/len(energies))**0.5:.8f}")

    return classification, final_energy


def main():
    print("Step 31: RK Autoregression Experiment")
    print("Question: Does the kernel sustain non-trivial dynamics without external signal?")
    print(f"Noise scale = 0.01 (Gaussian, every step)")
    print(f"'Fixed point' = energy < 0.0001 (at noise floor)")
    print(f"'Sustained dynamics' = energy > 0.001 (above noise floor)")

    # Primary run
    cls1, e1 = run_experiment(alpha=1.2, beta=0.8, seed=42, label="PRIMARY RUN")

    results = [("alpha=1.2", cls1, e1)]

    # If fixed point, re-run with higher alpha
    if "FIXED POINT" in cls1:
        print("\n" + "#"*72)
        print("  Primary run reached fixed point. Testing higher alpha values...")
        print("#"*72)

        cls2, e2 = run_experiment(alpha=1.5, beta=0.8, seed=42, label="ELEVATED ALPHA")
        results.append(("alpha=1.5", cls2, e2))

        cls3, e3 = run_experiment(alpha=2.0, beta=0.8, seed=42, label="HIGH ALPHA")
        results.append(("alpha=2.0", cls3, e3))

    # Final summary
    print(f"\n{'='*72}")
    print("  FINAL SUMMARY")
    print(f"{'='*72}\n")

    for label, cls, energy in results:
        print(f"  {label:<15}  energy={energy:.8f}  -> {cls}")

    print(f"\n{'='*72}")
    print("  INTERPRETATION")
    print(f"{'='*72}\n")

    any_sustained = any("SUSTAINED" in r[1] for r in results)
    if any_sustained:
        print("  The RK sustains non-trivial dynamics without external signal.")
        print("  The supercritical bifurcation (alpha > 1) prevents collapse.")
        print("  Cells continue to evolve through self-application and coupling.")
        print("  This is AUTOPOIETIC behavior -- the system maintains itself.")
    elif any("BORDERLINE" in r[1] for r in results):
        print("  The RK shows borderline dynamics -- energy hovers near threshold.")
        print("  The system neither fully stabilizes nor sustains strong dynamics.")
        print("  This may indicate a critical regime near a phase transition.")
    else:
        print("  The RK converges to eigenforms and reaches a noise-floor fixed point.")
        print("  Without external signal, the system finds stable attractors and stops.")
        print("  The noise prevents true zero but dynamics are trivial.")
        if len(results) > 1:
            print("  Even elevated alpha values did not sustain dynamics.")

    print()


if __name__ == '__main__':
    main()
