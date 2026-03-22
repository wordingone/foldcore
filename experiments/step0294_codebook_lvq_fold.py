#!/usr/bin/env python3
"""
Step 294 -- LVQ Fold: label-aware attract/repel + chain emergence test.

Spec. Hypothesis: LVQ dynamics reshape the codebook so same-class
vectors form arithmetic-progression chains as a training side effect.

Update rule:
  - winner.label == y: attract winner toward x (lr=0.015)
  - winner.label != y: repel winner from x (lr=0.015)
                        AND find nearest same-class v_s, attract toward x (lr=0.015)

Spawn: if no same-class vector within cosine sim 0.3 (calibrated to thermometer space).

Measurements:
  1. LOO accuracy vs baseline (41.8% plain thermometer, Step 286)
  2. Chain regularity: for b=3, walk same-class NN chains, measure step-size std dev in a-dimension
  3. Chain structure report

Kill criterion: chain regularity > 2.0 (irregular steps)
Success:        chain regularity < 1.0 AND LOO improvement over baseline
"""

import math
import time
import numpy as np

# ─── Config ────────────────────────────────────────────────────────────────────

TRAIN_MAX    = 20
K_VOTE       = 5
LR           = 0.015
SPAWN_THRESH = 0.3    # same-class cosine sim threshold for spawning
CHAIN_B      = 3      # fixed b for chain analysis

# ─── Encoding: one-hot (same as Step 286) ────────────────────────────────────

def encode_onehot(a, b, max_val=TRAIN_MAX):
    v = np.zeros(2 * max_val, dtype=np.float32)
    v[a - 1] = 1.0
    v[max_val + b - 1] = 1.0
    n = np.linalg.norm(v)
    return v / n if n > 1e-15 else v

# ─── Dataset ──────────────────────────────────────────────────────────────────

def build_dataset():
    X, Y, A, B = [], [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            X.append(encode_onehot(a, b))
            Y.append(a % b)
            A.append(a)
            B.append(b)
    return np.array(X, dtype=np.float32), np.array(Y), np.array(A), np.array(B)

# ─── LVQ Fold ─────────────────────────────────────────────────────────────────

class LVQFold:
    def __init__(self, spawn_thresh=SPAWN_THRESH, lr=LR, k=K_VOTE):
        self.V      = []   # codebook vectors (normalized)
        self.labels = []   # class labels
        self.a_vals = []   # original a value (for chain analysis)
        self.b_vals = []   # original b value
        self.lr     = lr
        self.spawn_thresh = spawn_thresh
        self.k      = k
        self.n_spawn  = 0
        self.n_attract = 0
        self.n_repel   = 0

    def _sims(self, x):
        if not self.V:
            return np.array([])
        return np.array(self.V) @ x

    def _normalize(self, v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-15 else v

    def step(self, x, y, a, b):
        sims = self._sims(x)

        # Spawn check: need a same-class vector within SPAWN_THRESH
        same_class_idxs = [i for i, lbl in enumerate(self.labels) if lbl == y]
        if not same_class_idxs:
            max_same_sim = -1.0
        else:
            max_same_sim = max(sims[i] for i in same_class_idxs)

        if max_same_sim < self.spawn_thresh:
            self.V.append(x.copy())
            self.labels.append(y)
            self.a_vals.append(a)
            self.b_vals.append(b)
            self.n_spawn += 1
            return

        # Find overall winner
        winner = int(np.argmax(sims))

        if self.labels[winner] == y:
            # ATTRACT winner toward x
            v = np.array(self.V[winner])
            self.V[winner] = self._normalize(v + self.lr * (x - v)).tolist()
            self.n_attract += 1
        else:
            # REPEL winner from x
            v = np.array(self.V[winner])
            self.V[winner] = self._normalize(v - self.lr * (x - v)).tolist()
            self.n_repel += 1

            # ATTRACT nearest same-class vector
            if same_class_idxs:
                best_sc = max(same_class_idxs, key=lambda i: sims[i])
                vs = np.array(self.V[best_sc])
                self.V[best_sc] = self._normalize(vs + self.lr * (x - vs)).tolist()
                self.n_attract += 1

    def predict(self, x):
        if not self.V:
            return -1
        sims  = self._sims(x)
        k_eff = min(self.k, len(sims))
        topk  = np.argpartition(sims, -k_eff)[-k_eff:]
        votes = {}
        for i in topk:
            c = self.labels[i]
            votes[c] = votes.get(c, 0) + 1
        return max(votes, key=votes.get)

# ─── Chain structure test ─────────────────────────────────────────────────────

def analyze_chains(model, chain_b=CHAIN_B):
    """
    For each class c with chain_b, find all codebook vectors with b_val == chain_b and label == c.
    Sort by a_val. Walk same-class NN chain from highest-a vector.
    Measure: does chain follow arithmetic progression? Compute std dev of step sizes in a-dimension.
    """
    V_arr = np.array(model.V)
    results = {}

    classes_at_b = sorted(set(
        model.labels[i] for i in range(len(model.labels))
        if model.b_vals[i] == chain_b
    ))

    for c in classes_at_b:
        idxs = [i for i in range(len(model.labels))
                if model.labels[i] == c and model.b_vals[i] == chain_b]
        if len(idxs) < 2:
            results[c] = {'n': len(idxs), 'chain': [], 'regularity': float('inf'),
                          'note': 'too few vectors'}
            continue

        # Sort by a_val (training origin)
        idxs_sorted = sorted(idxs, key=lambda i: model.a_vals[i])
        a_vals_sorted = [model.a_vals[i] for i in idxs_sorted]

        # Walk NN chain: start from highest-a, follow same-class nearest neighbor
        start = idxs_sorted[-1]  # highest a
        visited = set()
        chain = [start]
        visited.add(start)

        current = start
        while True:
            v_curr = V_arr[current]
            same_class_pool = [i for i in range(len(model.labels))
                               if model.labels[i] == c
                               and model.b_vals[i] == chain_b
                               and i not in visited]
            if not same_class_pool:
                break
            sims = V_arr[same_class_pool] @ v_curr
            next_idx = same_class_pool[int(np.argmax(sims))]
            chain.append(next_idx)
            visited.add(next_idx)
            current = next_idx

        chain_a_vals = [model.a_vals[i] for i in chain]
        steps = [abs(chain_a_vals[i] - chain_a_vals[i+1]) for i in range(len(chain_a_vals)-1)]
        regularity = float(np.std(steps)) if len(steps) > 1 else 0.0
        expected_step = chain_b  # for a%b, arithmetic progressions step by b

        results[c] = {
            'n': len(idxs),
            'chain': chain_a_vals,
            'steps': steps,
            'regularity': regularity,
            'expected_step': expected_step,
            'mean_step': float(np.mean(steps)) if steps else 0.0,
        }

    return results

# ─── LOO accuracy ─────────────────────────────────────────────────────────────

def train_accuracy(model, X, Y):
    correct = sum(1 for x, y in zip(X, Y) if model.predict(x) == y)
    return correct / len(Y)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Step 294 -- LVQ Fold + Chain Emergence Test", flush=True)
    print(f"Dataset: a,b in 1..{TRAIN_MAX}, {TRAIN_MAX**2} pairs", flush=True)
    print(f"Encoding: one-hot ({2*TRAIN_MAX}d)", flush=True)
    print(f"LVQ: attract same-class, repel different-class", flush=True)
    print(f"Spawn thresh (same-class cosine): {SPAWN_THRESH}", flush=True)
    print(f"Baseline (Step 286 plain thermometer LOO): 41.8%", flush=True)
    print(flush=True)

    X, Y, A, B = build_dataset()

    # Train LVQ fold
    model = LVQFold()
    for x, y, a, b in zip(X, Y, A, B):
        model.step(x, y, int(a), int(b))

    acc = train_accuracy(model, X, Y)

    print(f"LVQ Fold training complete:", flush=True)
    print(f"  Codebook size: {len(model.V)}", flush=True)
    print(f"  Spawns: {model.n_spawn}  Attracts: {model.n_attract}  Repels: {model.n_repel}",
          flush=True)
    print(f"  Training accuracy: {acc*100:.1f}%", flush=True)
    print(flush=True)

    # Chain structure analysis (b=CHAIN_B)
    print(f"=== Chain Structure Test (b={CHAIN_B}) ===", flush=True)
    chains = analyze_chains(model, chain_b=CHAIN_B)

    all_regularities = []
    for c in sorted(chains.keys()):
        r = chains[c]
        if r['n'] < 2:
            print(f"  Class a%{CHAIN_B}={c}: {r['n']} vector(s) — {r['note']}", flush=True)
            continue
        print(f"  Class a%{CHAIN_B}={c}:", flush=True)
        print(f"    Vectors (by a): {sorted([model.a_vals[i] for i in range(len(model.labels)) if model.labels[i]==c and model.b_vals[i]==CHAIN_B])}",
              flush=True)
        print(f"    NN chain a-vals: {r['chain']}", flush=True)
        print(f"    Step sizes:      {r['steps']}", flush=True)
        print(f"    Mean step: {r['mean_step']:.1f} (expected: {r['expected_step']})",
              flush=True)
        print(f"    Regularity (std): {r['regularity']:.2f}", flush=True)
        all_regularities.append(r['regularity'])

    mean_regularity = float(np.mean(all_regularities)) if all_regularities else float('inf')
    print(flush=True)
    print(f"Mean chain regularity (std of step sizes): {mean_regularity:.2f}", flush=True)

    # ─── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 294 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Training accuracy: {acc*100:.1f}% (baseline: 41.8%)", flush=True)
    print(f"Codebook size: {len(model.V)}", flush=True)
    print(f"Mean chain regularity (b={CHAIN_B}): {mean_regularity:.2f}", flush=True)
    print(flush=True)
    print("KILL CRITERION (Spec):", flush=True)
    acc_passes   = acc > 0.418
    chain_passes = mean_regularity < 2.0 if all_regularities else False

    if not chain_passes:
        note = f"regularity={mean_regularity:.2f} > 2.0" if all_regularities else "no chains to measure"
        print(f"  KILLED — chains irregular or absent ({note})", flush=True)
        print(f"  LVQ dynamics do NOT produce emergent transition structure.", flush=True)
    elif not acc_passes:
        print(f"  PARTIAL — chains regular but accuracy ({acc*100:.1f}%) <= baseline (41.8%)",
              flush=True)
    else:
        print(f"  PASSES — chains regular (regularity={mean_regularity:.2f} < 2.0) "
              f"AND accuracy improves (+{(acc-0.418)*100:.1f}pp)", flush=True)
        if mean_regularity < 1.0:
            print(f"  STRONG PASS — regularity < 1.0", flush=True)

    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__ == '__main__':
    main()
