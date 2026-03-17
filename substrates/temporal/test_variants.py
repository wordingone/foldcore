#!/usr/bin/env python3
"""Test each unjustified element independently. Which are irreducible?"""

import torch, numpy as np
from collections import Counter
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from temporal import TemporalPrediction


class Depth1(TemporalPrediction):
    """Remove chain depth 2. Use W@x directly (depth 1)."""
    def step(self, x):
        x = x.to(self.device).float()
        if self.prev is None:
            self.prev = x.clone()
            return 0
        pred = self.W @ self.prev
        err = x - pred
        self.pred_err = err.norm().item()
        denom = self.prev.dot(self.prev).clamp(min=1e-8)
        self.W += torch.outer(err, self.prev) / denom
        action = (self.W @ x).abs().argmax().item() % self.n_actions  # depth 1
        self.prev = x.clone()
        return action


class NoAbs(TemporalPrediction):
    """Remove abs() before argmax."""
    def step(self, x):
        x = x.to(self.device).float()
        if self.prev is None:
            self.prev = x.clone()
            return 0
        pred = self.W @ self.prev
        err = x - pred
        self.pred_err = err.norm().item()
        denom = self.prev.dot(self.prev).clamp(min=1e-8)
        self.W += torch.outer(err, self.prev) / denom
        p1 = self.W @ x
        p2 = self.W @ p1
        action = p2.argmax().item() % self.n_actions  # no abs
        self.prev = x.clone()
        return action


class Depth1NoAbs(TemporalPrediction):
    """Both: depth 1, no abs."""
    def step(self, x):
        x = x.to(self.device).float()
        if self.prev is None:
            self.prev = x.clone()
            return 0
        pred = self.W @ self.prev
        err = x - pred
        self.pred_err = err.norm().item()
        denom = self.prev.dot(self.prev).clamp(min=1e-8)
        self.W += torch.outer(err, self.prev) / denom
        action = (self.W @ x).argmax().item() % self.n_actions  # depth 1 + no abs
        self.prev = x.clone()
        return action


def test_disc(cls, label, n_seeds=5):
    print(f"\n{label}:")
    all_dom = []
    all_dist = []
    for seed in range(n_seeds):
        torch.manual_seed(seed * 1000 + 42)
        s = cls(d=32, n_actions=4)
        centers = [torch.randn(32) * 3 for _ in range(4)]
        for _ in range(200):
            for c in centers:
                s.step(c + 0.3 * torch.randn(32))
        results = {i: [] for i in range(4)}
        for _ in range(100):
            for i, c in enumerate(centers):
                a = s.step(c + 0.3 * torch.randn(32))
                results[i].append(a)
        fracs = []
        dom_actions = []
        for i in range(4):
            counts = Counter(results[i])
            dom = counts.most_common(1)[0]
            fracs.append(dom[1] / 100)
            dom_actions.append(dom[0])
        all_dom.append(np.mean(fracs))
        all_dist.append(len(set(dom_actions)))

    print(f"  Dominance: {np.mean(all_dom)*100:.1f}% (range {min(all_dom)*100:.0f}-{max(all_dom)*100:.0f})")
    print(f"  Distinct:  {np.mean(all_dist):.1f} (range {min(all_dist)}-{max(all_dist)})")

    # Action diversity
    torch.manual_seed(42)
    s = cls(d=64, n_actions=4)
    actions = [s.step(torch.randn(64)) for _ in range(500)]
    counts = Counter(actions)
    balance = min(counts.values()) / max(counts.values()) if len(counts) > 1 else 0
    print(f"  Balance:   {balance:.2f}  Actions: {dict(sorted(counts.items()))}")

    # W norm
    print(f"  ||W||:     {s.W.norm().item():.1f}")
    return np.mean(all_dom), np.mean(all_dist), balance


def main():
    print("=" * 60)
    print("Testing each unjustified element independently")
    print("=" * 60)
    print("If variant works: element is UNJUSTIFIED (U)")
    print("If variant fails: element is IRREDUCIBLE (I)")

    variants = [
        (TemporalPrediction, "BASELINE (5 U)"),
        (Depth1, "- chain_depth (depth 1)"),
        (NoAbs, "- abs (no abs before argmax)"),
        (Depth1NoAbs, "- both (depth 1, no abs)"),
    ]

    for cls, label in variants:
        dom, dist, bal = test_disc(cls, label)
        # Classify: working if dominance > 50% AND distinct > 1 AND balance > 0.1
        works = dom > 0.5 and dist > 1 and bal > 0.1
        print(f"  VERDICT: {'WORKS' if works else 'DEGRADED/DEAD'}")


if __name__ == '__main__':
    main()
