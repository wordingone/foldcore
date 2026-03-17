#!/usr/bin/env python3
"""
Compare TemporalMinimal (0 unjustified) vs TemporalPrediction (5 unjustified).
Quick test: discrimination, stability, action diversity. <30s.
"""

import torch, numpy as np
from collections import Counter
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from temporal import TemporalMinimal, TemporalPrediction


def test_discrimination(cls, label, d=32, n_actions=4, n_seeds=5):
    """Run discrimination 5 seeds. Report min/max/mean."""
    print(f"\n{label} -- Discrimination (d={d}, {n_seeds} seeds):")
    all_dom = []
    all_distinct = []
    for seed in range(n_seeds):
        torch.manual_seed(seed * 1000 + 42)
        s = cls(d=d, n_actions=n_actions)
        centers = [torch.randn(d) * 3 for _ in range(4)]

        # Warmup
        for _ in range(200):
            for c in centers:
                s.step(c + 0.3 * torch.randn(d))

        # Test
        results = {i: [] for i in range(4)}
        for _ in range(100):
            for i, c in enumerate(centers):
                s.step(c + 0.3 * torch.randn(d))
                results[i].append(results[i])  # bug: should append action
        # Fix: re-run correctly
        results = {i: [] for i in range(4)}
        for _ in range(100):
            for i, c in enumerate(centers):
                a = s.step(c + 0.3 * torch.randn(d))
                results[i].append(a)

        fracs = []
        dom_actions = []
        for i in range(4):
            counts = Counter(results[i])
            dom = counts.most_common(1)[0]
            fracs.append(dom[1] / 100)
            dom_actions.append(dom[0])
        avg_dom = np.mean(fracs)
        n_dist = len(set(dom_actions))
        all_dom.append(avg_dom)
        all_distinct.append(n_dist)

    print(f"  Avg dominance: min={min(all_dom)*100:.1f}%  max={max(all_dom)*100:.1f}%  mean={np.mean(all_dom)*100:.1f}%")
    print(f"  Distinct actions: min={min(all_distinct)}  max={max(all_distinct)}  mean={np.mean(all_distinct):.1f}")
    return np.mean(all_dom), np.mean(all_distinct)


def test_stability(cls, label, d=32, n_steps=500):
    """Check W norm stability."""
    print(f"\n{label} -- W Stability (d={d}, {n_steps} steps):")
    torch.manual_seed(42)
    s = cls(d=d, n_actions=4)
    norms = []
    for step in range(n_steps):
        s.step(torch.randn(d))
        if (step + 1) % 100 == 0:
            n = s.W.norm().item()
            norms.append(n)
            print(f"  step {step+1}: ||W||={n:.1f}")
    exploded = any(n > 1e6 for n in norms)
    print(f"  Exploded: {exploded}")
    return not exploded


def test_action_diversity(cls, label, d=64, n_actions=4, n_steps=500):
    """Check action balance."""
    print(f"\n{label} -- Action Diversity:")
    torch.manual_seed(42)
    s = cls(d=d, n_actions=n_actions)
    actions = [s.step(torch.randn(d)) for _ in range(n_steps)]
    counts = Counter(actions)
    print(f"  Actions: {dict(sorted(counts.items()))}")
    balance = min(counts.values()) / max(counts.values()) if len(counts) > 1 else 0
    print(f"  Balance: {balance:.2f}")
    return balance


def main():
    print("=" * 60)
    print("Minimal (0 U) vs Standard (5 U) Comparison")
    print("=" * 60)

    for cls, label in [(TemporalMinimal, "MINIMAL (0 U)"),
                       (TemporalPrediction, "STANDARD (5 U)")]:
        stable = test_stability(cls, label)
        if not stable:
            print(f"  {label}: W EXPLODED. Substrate non-viable.")
            continue
        test_discrimination(cls, label)
        test_action_diversity(cls, label)

    print("\n" + "=" * 60)
    print("If Minimal works: R3 potentially PASS (0 unjustified)")
    print("If Minimal explodes: normalization is IRREDUCIBLE, not unjustified")
    print("Either way: one fewer question about the frozen frame.")


if __name__ == '__main__':
    main()
