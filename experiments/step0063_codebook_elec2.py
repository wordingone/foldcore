#!/usr/bin/env python3
"""
Step 63: FluxCore vs. River baselines on Electricity (Elec2) benchmark.

Elec2 is the canonical streaming concept drift benchmark:
  - 45,312 samples, 8 features (already in [0,1])
  - Binary classification: UP/DOWN electricity price
  - Natural concept drift (time-of-day, seasonal)
  - Published baselines: HT ~85.7%, ARF ~88-89%

Evaluation protocol: prequential (interleaved test-then-train).
  For each sample:
    1. Predict using current model state
    2. Update model with (x, y)
    3. Record accuracy in rolling window

FluxCore readout: nearest-codebook-vector label assignment.
  Each codebook vector accumulates per-class vote counts.
  After step(), winner's vote count for true_y incremented.

FluxCore additional metrics:
  - Spawn rate (drift signal): spikes = concept drift detected
  - Codebook size (memory): ~constant once settled
"""

import sys
import math
import random
import time

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from fluxcore_manytofew import ManyToFewKernel, _vec_cosine
from rk import frob

# ─── Dataset ──────────────────────────────────────────────────────────────────

from river import datasets
from river.tree import HoeffdingTreeClassifier

# ─── Embedding: 8 features → d=64 ────────────────────────────────────────────

D = 64
FEATURE_KEYS = ['date', 'day', 'period', 'nswprice', 'nswdemand',
                'vicprice', 'vicdemand', 'transfer']

random.seed(7777)
_PROJ = [[random.gauss(0, 1.0 / math.sqrt(len(FEATURE_KEYS)))
          for _ in range(len(FEATURE_KEYS))]
         for _ in range(D)]


def embed(x):
    """8 features → unit vector in R^64 via random projection + tanh + normalize."""
    feats = [float(x[k]) for k in FEATURE_KEYS]
    projected = [
        math.tanh(sum(_PROJ[i][j] * feats[j] for j in range(len(feats))))
        for i in range(D)
    ]
    n = math.sqrt(sum(v * v for v in projected) + 1e-15)
    return [v / n for v in projected]


# ─── FluxCore prequential classifier ──────────────────────────────────────────

class FluxCoreClassifier:
    """
    Thin readout wrapper around ManyToFewKernel for binary classification.
    Maintains per-codebook-vector vote counts for label prediction.
    """

    def __init__(self, n_classes=2, **kernel_kwargs):
        self.kernel = ManyToFewKernel(**kernel_kwargs)
        self.n_classes = n_classes
        self.cb_labels = []  # parallel to kernel.codebook: list of [count_0, count_1, ...]

    def predict(self, r):
        """Nearest-codebook-vector prediction. Returns class index or None if empty."""
        if not self.kernel.codebook:
            return None
        sims = [_vec_cosine(v, r) for v in self.kernel.codebook]
        winner = max(range(len(self.kernel.codebook)), key=lambda i: sims[i])
        votes = self.cb_labels[winner]
        return votes.index(max(votes))

    def learn(self, r, y):
        """Update kernel with r, then update winner's vote counts with true label y."""
        n_before = len(self.kernel.codebook)
        self.kernel.step(r=r)
        n_after = len(self.kernel.codebook)

        # Sync label list size (spawn adds one vector at end)
        if n_after > n_before:
            self.cb_labels.append([0] * self.n_classes)
        # Spawn+merge: n_after == n_before, no sync needed

        # Update winner's vote counts
        if self.kernel.codebook:
            sims = [_vec_cosine(v, r) for v in self.kernel.codebook]
            winner = max(range(len(self.kernel.codebook)), key=lambda i: sims[i])
            self.cb_labels[winner][y] += 1


# ─── Main benchmark ───────────────────────────────────────────────────────────

def run_benchmark():
    WINDOW = 1000
    print('=' * 72)
    print('  Step 63 — FluxCore vs. Hoeffding Tree on Electricity (Elec2)')
    print(f'  45312 samples | d={D} | prequential | window={WINDOW}')
    print('=' * 72)

    fc = FluxCoreClassifier(
        n_classes=2,
        n_matrix=8, k=4, d=D, seed=42, proj_seed=999,
        tau=0.3, k_couple=5,
        spawn_thresh=0.5, merge_thresh=0.95, lr_codebook=0.015,
    )
    ht = HoeffdingTreeClassifier()

    data = list(datasets.Elec2())
    total = len(data)

    fc_correct = []
    ht_correct = []
    spawn_rate_windows = []

    print(f'\n  {"Window":>8}  {"FC Acc":>8}  {"HT Acc":>8}  {"CB size":>8}  {"Spawns/W":>10}')
    print('  ' + '-' * 52)

    t0 = time.time()
    prev_spawned = 0

    for idx, (x, y_raw) in enumerate(data):
        y = int(y_raw)
        r = embed(x)

        # Predict
        fc_pred = fc.predict(r)
        ht_pred = ht.predict_one(x)

        fc_correct.append(1 if fc_pred == y else 0)
        ht_correct.append(1 if (ht_pred is not None and bool(ht_pred) == bool(y)) else 0)

        # Learn
        fc.learn(r, y)
        ht.learn_one(x, y_raw)

        # Window report
        if (idx + 1) % WINDOW == 0:
            window_fc = fc_correct[-WINDOW:]
            window_ht = ht_correct[-WINDOW:]
            fc_acc = sum(window_fc) / len(window_fc)
            ht_acc = sum(window_ht) / len(window_ht)
            new_spawns = fc.kernel.total_spawned - prev_spawned
            spawn_rate_windows.append(new_spawns)
            prev_spawned = fc.kernel.total_spawned
            cb_size = len(fc.kernel.codebook)
            w = (idx + 1) // WINDOW
            print(f'  {(idx+1):>8}  {fc_acc:>8.4f}  {ht_acc:>8.4f}  {cb_size:>8}  {new_spawns:>10}')

    elapsed = time.time() - t0
    fc_overall = sum(fc_correct) / len(fc_correct)
    ht_overall = sum(ht_correct) / len(ht_correct)

    print('\n' + '=' * 72)
    print('  STEP 63 SUMMARY')
    print('=' * 72)
    print(f'\n  Dataset: Electricity (Elec2), {total} samples, binary')
    print(f'  Embedding: {len(FEATURE_KEYS)} features → d={D} (random projection + tanh + normalize)')
    print(f'  FluxCore: n_matrix=8, k=4, spawn=0.5, merge=0.95, lr_cb=0.015')

    print(f'\n  Overall prequential accuracy:')
    print(f'    FluxCore:                  {fc_overall:.4f}  ({fc_overall*100:.1f}%)')
    print(f'    Hoeffding Tree (River):    {ht_overall:.4f}  ({ht_overall*100:.1f}%)')
    print(f'    Published ARF baseline:    ~0.889  (88.9%)  [literature]')
    print(f'    Random baseline:           ~0.500  (50.0%)')

    print(f'\n  Memory (codebook size):')
    print(f'    FluxCore codebook vectors: {len(fc.kernel.codebook)}')
    print(f'    Total spawned:             {fc.kernel.total_spawned}')
    print(f'    Total merged:              {fc.kernel.total_merged}')

    print(f'\n  Drift signal (spawns per {WINDOW}-sample window):')
    print(f'    Max spawn window:          {max(spawn_rate_windows)}')
    print(f'    Mean spawn rate:           {sum(spawn_rate_windows)/len(spawn_rate_windows):.1f}')
    peak_windows = sorted(enumerate(spawn_rate_windows, 1), key=lambda x: -x[1])[:3]
    for w, s in peak_windows:
        print(f'    Window {w:>3} (samples {(w-1)*WINDOW+1}-{w*WINDOW}): {s} spawns')

    print(f'\n  Runtime: {elapsed:.1f}s')

    # Performance verdict
    above_random = fc_overall > 0.55
    gap_to_ht = ht_overall - fc_overall
    print(f'\n  Analysis:')
    if above_random:
        print(f'    [+] FluxCore exceeds random baseline (>{fc_overall*100:.1f}% > 50%)')
    else:
        print(f'    [-] FluxCore near random baseline — readout may need tuning')
    print(f'    [i] Gap to Hoeffding Tree: {gap_to_ht*100:.1f}pp')
    print(f'    [+] Memory: {len(fc.kernel.codebook)} vectors vs HT\'s full tree (unbounded)')
    print(f'    [+] Drift signal: spawn rate provides unsupervised drift indicator')

    print('\n' + '=' * 72)
    return fc_overall, ht_overall, len(fc.kernel.codebook), spawn_rate_windows


if __name__ == '__main__':
    run_benchmark()
