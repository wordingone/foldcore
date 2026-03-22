#!/usr/bin/env python3
"""
Step 293 -- AMR Fold: disagreement-driven spawn + distance-gated merge on a%b.

Spec. Placement test: does disagreement-driven AMR produce better
codebook topology than distance-driven spawning?

Architecture:
- For each training input x with label y:
  1. Compute cosine sims to all codebook vectors
  2. Top-K (K=5) vote
  3. SPAWN if votes DISAGREE (plurality < K)
  4. MERGE if votes AGREE (plurality == K) AND nearest same-class sim > theta_merge
  5. Update winner toward x (lr=0.015)
- Eval: top-K class vote over codebook

Baseline: Step 286 best LOO = 49% (thermometer + LOO-scored features).
Kill criterion: AMR LOO <= 49%.

Anti-inflation: placement test only. Human designed the mechanism.
"""

import math
import time
import numpy as np

# ─── Config ────────────────────────────────────────────────────────────────────

TRAIN_MAX   = 20
K_VOTE      = 5
THETA_MERGE = 0.85
LR          = 0.015
DIST_THRESH = 0.7     # for distance-based baseline comparison

# ─── Encoding: thermometer (best from Step 286-289 arc) ───────────────────────

def encode_thermometer(a, b, max_val=TRAIN_MAX):
    va = np.zeros(max_val, dtype=np.float32)
    va[:a] = 1.0
    vb = np.zeros(max_val, dtype=np.float32)
    vb[:b] = 1.0
    v = np.concatenate([va, vb])
    n = np.linalg.norm(v)
    return v / n if n > 1e-15 else v

# ─── Dataset ──────────────────────────────────────────────────────────────────

def build_dataset():
    X, Y = [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            X.append(encode_thermometer(a, b))
            Y.append(a % b)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int32)

# ─── AMR Fold ─────────────────────────────────────────────────────────────────

class AMRFold:
    def __init__(self, k=K_VOTE, theta_merge=THETA_MERGE, lr=LR):
        self.V      = []   # codebook vectors (normalized)
        self.labels = []   # codebook labels
        self.k      = k
        self.theta_merge = theta_merge
        self.lr     = lr
        self.n_spawn  = 0
        self.n_merge  = 0
        self.n_update = 0

    def _sims(self, x):
        if not self.V:
            return np.array([])
        return np.array(self.V) @ x  # cosine sims (all normalized)

    def _topk_vote(self, sims):
        """Returns (winner_idx, plurality_class, plurality_count)."""
        k_eff = min(self.k, len(sims))
        topk  = np.argpartition(sims, -k_eff)[-k_eff:]
        votes = {}
        for i in topk:
            c = self.labels[i]
            votes[c] = votes.get(c, 0) + 1
        plurality_class = max(votes, key=votes.get)
        plurality_count = votes[plurality_class]
        winner = topk[np.argmax(sims[topk])]
        return winner, plurality_class, plurality_count

    def step(self, x, y):
        if not self.V:
            # Codebook empty — always spawn first point
            self.V.append(x.copy())
            self.labels.append(y)
            self.n_spawn += 1
            return

        sims = self._sims(x)
        winner, plurality_class, plurality_count = self._topk_vote(sims)

        agrees = (plurality_count == self.k)

        if not agrees:
            # SPAWN: disagreement at this input
            self.V.append(x.copy())
            self.labels.append(y)
            self.n_spawn += 1
        else:
            # MAYBE MERGE: find nearest same-class neighbor of winner
            same_class = [i for i, lbl in enumerate(self.labels)
                          if lbl == self.labels[winner] and i != winner]
            if same_class and agrees:
                sc_sims = sims[same_class]
                best_sc = same_class[np.argmax(sc_sims)]
                if sims[best_sc] > self.theta_merge:
                    # MERGE: average + normalize
                    merged = self.V[winner] + self.V[best_sc]
                    n = np.linalg.norm(merged)
                    merged = merged / n if n > 1e-15 else merged
                    # Replace winner, delete best_sc
                    self.V[winner] = merged
                    del self.V[best_sc]
                    del self.labels[best_sc]
                    self.n_merge += 1
                    # Recompute winner after merge
                    sims = self._sims(x)
                    winner = int(np.argmax(sims))

        # UPDATE winner toward x
        v = self.V[winner]
        updated = v + self.lr * (x - v)
        n = np.linalg.norm(updated)
        self.V[winner] = updated / n if n > 1e-15 else updated
        self.n_update += 1

    def predict(self, x):
        if not self.V:
            return -1
        sims = self._sims(x)
        k_eff = min(self.k, len(sims))
        topk  = np.argpartition(sims, -k_eff)[-k_eff:]
        votes = {}
        for i in topk:
            c = self.labels[i]
            votes[c] = votes.get(c, 0) + 1
        return max(votes, key=votes.get)


class DistanceFold:
    """Baseline: distance-based spawn (theta=0.7), same update, top-K vote."""
    def __init__(self, k=K_VOTE, spawn_thresh=DIST_THRESH, lr=LR):
        self.V      = []
        self.labels = []
        self.k      = k
        self.spawn_thresh = spawn_thresh
        self.lr     = lr
        self.n_spawn  = 0

    def step(self, x, y):
        if not self.V:
            self.V.append(x.copy()); self.labels.append(y); self.n_spawn += 1; return
        sims   = np.array(self.V) @ x
        winner = int(np.argmax(sims))
        if sims[winner] < self.spawn_thresh:
            self.V.append(x.copy()); self.labels.append(y); self.n_spawn += 1; return
        v = self.V[winner]
        updated = v + self.lr * (x - v)
        n = np.linalg.norm(updated)
        self.V[winner] = updated / n if n > 1e-15 else updated

    def predict(self, x):
        if not self.V: return -1
        sims  = np.array(self.V) @ x
        k_eff = min(self.k, len(sims))
        topk  = np.argpartition(sims, -k_eff)[-k_eff:]
        votes = {}
        for i in topk:
            c = self.labels[i]
            votes[c] = votes.get(c, 0) + 1
        return max(votes, key=votes.get)


# ─── Evaluation: training set accuracy ────────────────────────────────────────

def train_accuracy(model, X, Y):
    correct = sum(1 for x, y in zip(X, Y) if model.predict(x) == y)
    return correct / len(Y)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Step 293 -- AMR Fold on a%b", flush=True)
    print(f"Dataset: a,b in 1..{TRAIN_MAX}, {TRAIN_MAX**2} pairs", flush=True)
    print(f"Encoding: thermometer ({2*TRAIN_MAX}d)", flush=True)
    print(f"K_vote={K_VOTE}, theta_merge={THETA_MERGE}, lr={LR}", flush=True)
    print(f"Baseline (Step 286 best LOO): 49%", flush=True)
    print(flush=True)

    X, Y = build_dataset()

    # --- AMR Fold ---
    amr = AMRFold()
    for x, y in zip(X, Y):
        amr.step(x, y)
    amr_acc = train_accuracy(amr, X, Y)

    print(f"AMR Fold:", flush=True)
    print(f"  Codebook size: {len(amr.V)}", flush=True)
    print(f"  Spawns: {amr.n_spawn}  Merges: {amr.n_merge}  Updates: {amr.n_update}",
          flush=True)
    print(f"  Training accuracy: {amr_acc*100:.1f}%", flush=True)
    print(flush=True)

    # --- Distance-based baseline ---
    dist = DistanceFold()
    for x, y in zip(X, Y):
        dist.step(x, y)
    dist_acc = train_accuracy(dist, X, Y)

    print(f"Distance Fold (theta=0.7):", flush=True)
    print(f"  Codebook size: {len(dist.V)}", flush=True)
    print(f"  Spawns: {dist.n_spawn}", flush=True)
    print(f"  Training accuracy: {dist_acc*100:.1f}%", flush=True)
    print(flush=True)

    # ─── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("=" * 60, flush=True)
    print("STEP 293 SUMMARY -- AMR Fold vs Distance Fold", flush=True)
    print("=" * 60, flush=True)
    print(f"{'':25} {'Acc':>7} {'CB size':>8} {'Spawns':>8}", flush=True)
    print("-" * 50, flush=True)
    print(f"{'AMR (disagree+merge)':25} {amr_acc*100:>6.1f}% {len(amr.V):>8} "
          f"{amr.n_spawn:>8}", flush=True)
    print(f"{'Distance (theta=0.7)':25} {dist_acc*100:>6.1f}% {len(dist.V):>8} "
          f"{dist.n_spawn:>8}", flush=True)
    delta = amr_acc - dist_acc
    print(f"\nAMR vs Distance: {delta*100:+.1f}pp", flush=True)
    print(f"AMR vs Step 286 best LOO (49%): {(amr_acc - 0.49)*100:+.1f}pp", flush=True)

    print(flush=True)
    print("KILL CRITERION (Spec):", flush=True)
    if amr_acc <= 0.49:
        print(f"  KILLED -- AMR ({amr_acc*100:.1f}%) <= Step 286 baseline (49%)",
              flush=True)
        print(f"  Disagreement-driven placement does not help.", flush=True)
    else:
        print(f"  PASSES -- AMR ({amr_acc*100:.1f}%) > 49% baseline.", flush=True)
        print(f"  Disagreement-driven placement produces better codebook topology.",
              flush=True)

    print(f"\nElapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
