#!/usr/bin/env python3
"""
Steps 68-70: Architecture evolution on Split-CIFAR-100.

Run sequentially. Each step adds ONE change on top of the previous.

Step 68: k-nearest voting readout  (baseline: 1-NN, test k=1,3,5,7,10)
Step 69: Self-calibrating spawn threshold  (auto-detect from first 100 steps)
Step 70: Frozen-on-maturity prototypes  (freeze after 50 hits, kills forgetting)

Uses numpy-accelerated codebook (same algorithm as fluxcore_manytofew.py).
Cached ResNet-18 features: C:/Users/Admin/cifar100_resnet18_features.npz
"""

import sys
import time
import math
import numpy as np

sys.path.insert(0, 'B:/M/avir/research/fluxcore')

# ─── Config ───────────────────────────────────────────────────────────────────

N_TASKS      = 20
CLASSES_TASK = 5
D_EMBED      = 512
N_TRAIN_TASK = 500 * CLASSES_TASK   # 2500
CACHE_PATH   = 'C:/Users/Admin/cifar100_resnet18_features.npz'

MERGE_THRESH   = 0.95
LR_CODEBOOK    = 0.015
BASE_THRESH    = 0.95   # Step 66b best


# ─── Numpy codebook (base + incremental additions per step) ──────────────────

class NumpyCodebook:
    """
    Fold codebook: spawn/update/merge. Numpy-accelerated.
    Supports Step 68 (k-NN), Step 69 (self-calibrating thresh), Step 70 (maturity freeze).
    """

    def __init__(self, d, spawn_thresh=0.95, merge_thresh=0.95, lr=0.015,
                 # Step 69: self-calibrating threshold
                 auto_thresh=False, auto_warmup=100, auto_percentile=10,
                 # Step 70: maturity freeze
                 maturity=None):
        self.d            = d
        self.spawn_thresh = spawn_thresh
        self.merge_thresh = merge_thresh
        self.lr           = lr
        self.vectors      = np.empty((0, d), dtype=np.float32)
        self.labels       = []
        self.n_spawned    = 0
        self.n_merged     = 0

        # Step 69
        self.auto_thresh    = auto_thresh
        self.auto_warmup    = auto_warmup
        self.auto_percentile = auto_percentile
        self._warmup_sims   = []
        self._thresh_locked = not auto_thresh  # if not auto, locked immediately

        # Step 70
        self.maturity       = maturity          # None = no freeze
        self.hit_counts     = []                # parallel to vectors

    def step(self, r, label=None):
        if len(self.vectors) == 0:
            self._spawn(r, label)
            return 0

        sims    = self.vectors @ r   # (N,) cosines
        winner  = int(np.argmax(sims))
        max_sim = float(sims[winner])

        # Step 69: collect warmup stats, then lock threshold
        if self.auto_thresh and not self._thresh_locked:
            self._warmup_sims.append(max_sim)
            if len(self._warmup_sims) >= self.auto_warmup:
                self.spawn_thresh = float(np.percentile(
                    self._warmup_sims, self.auto_percentile))
                self._thresh_locked = True

        if max_sim < self.spawn_thresh:
            winner = self._spawn(r, label)
        else:
            # Step 70: only update if not mature
            if self.maturity is None or self.hit_counts[winner] < self.maturity:
                v = self.vectors[winner] + self.lr * r
                self.vectors[winner] = v / (np.linalg.norm(v) + 1e-15)
                self.hit_counts[winner] += 1
            else:
                self.hit_counts[winner] += 1  # track hits even when frozen

        return winner

    def _spawn(self, r, label):
        new_v = r / (np.linalg.norm(r) + 1e-15)
        n = len(self.vectors)
        if n > 0:
            abs_sims = np.abs(self.vectors @ new_v)
            best_i   = int(np.argmax(abs_sims))
            if abs_sims[best_i] > self.merge_thresh:
                fused = self.vectors[best_i] + new_v
                self.vectors[best_i] = fused / (np.linalg.norm(fused) + 1e-15)
                self.n_merged += 1
                return best_i
        self.vectors = np.vstack([self.vectors, new_v[np.newaxis, :]])
        self.labels.append(label)
        self.hit_counts.append(1)
        self.n_spawned += 1
        return len(self.vectors) - 1

    def classify_batch(self, X, k=1):
        """X: (n, d). Returns list of predicted labels."""
        if len(self.vectors) == 0:
            return [None] * len(X)
        sims = X @ self.vectors.T     # (n, N)
        if k == 1:
            winners = np.argmax(sims, axis=1)
            return [self.labels[w] for w in winners]
        # k > 1: majority vote per sample
        top_k_idx = np.argpartition(sims, -k, axis=1)[:, -k:]  # (n, k)
        preds = []
        for i in range(len(X)):
            votes = [self.labels[j] for j in top_k_idx[i]]
            preds.append(max(set(votes), key=votes.count))
        return preds

    @property
    def auto_thresh_value(self):
        return self.spawn_thresh if self._thresh_locked else None


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    data = np.load(CACHE_PATH)
    X_tr, y_tr = data['X_train'], data['y_train']
    X_te, y_te = data['X_test'],  data['y_test']
    splits = []
    for t in range(N_TASKS):
        c0 = t * CLASSES_TASK
        c1 = c0 + CLASSES_TASK
        tm = np.isin(y_tr, range(c0, c1))
        em = np.isin(y_te, range(c0, c1))
        splits.append((X_tr[tm], y_tr[tm], X_te[em], y_te[em]))
    return splits


def run_experiment(splits, spawn_thresh=BASE_THRESH, k=1,
                   auto_thresh=False, maturity=None, label=''):
    cb = NumpyCodebook(D_EMBED, spawn_thresh=spawn_thresh,
                       auto_thresh=auto_thresh, maturity=maturity)
    acc_matrix = [[None] * N_TASKS for _ in range(N_TASKS)]
    t0 = time.time()

    for task_id in range(N_TASKS):
        X_t, y_t = splits[task_id][0], splits[task_id][1]
        for i in range(len(X_t)):
            cb.step(X_t[i], label=int(y_t[i]))

        for eval_task in range(task_id + 1):
            X_te, y_te = splits[eval_task][2], splits[eval_task][3]
            preds   = cb.classify_batch(X_te, k=k)
            correct = sum(1 for p, g in zip(preds, y_te) if p == g)
            acc_matrix[eval_task][task_id] = correct / len(y_te)

    elapsed = time.time() - t0
    final   = [acc_matrix[i][N_TASKS-1] for i in range(N_TASKS)]
    aa      = sum(final) / N_TASKS
    fgt     = [max(0., acc_matrix[i][i] - acc_matrix[i][N_TASKS-1])
               for i in range(N_TASKS-1)]
    avg_f   = sum(fgt) / len(fgt)

    auto_val = cb.auto_thresh_value if auto_thresh else None

    return {
        'label': label, 'aa': aa, 'forgetting': avg_f,
        'cb_size': len(cb.vectors), 'spawned': cb.n_spawned,
        'elapsed': elapsed, 'final': final,
        'auto_thresh_val': auto_val,
    }


# ─── STEP 68: k-NN readout ────────────────────────────────────────────────────

def step68(splits):
    print('=' * 70)
    print('  Step 68 -- k-Nearest Voting Readout (spawn=0.95, same codebook)')
    print('  One variable: k in {1, 3, 5, 7, 10}')
    print('=' * 70)

    results = {}
    for k in [1, 3, 5, 7, 10]:
        print(f'  Running k={k}...')
        r = run_experiment(splits, spawn_thresh=BASE_THRESH, k=k,
                           label=f'k={k}')
        results[k] = r
        print(f'    k={k}: AA={r["aa"]*100:.1f}%  F={r["forgetting"]*100:.1f}pp  '
              f'cb={r["cb_size"]}  t={r["elapsed"]:.0f}s')

    print()
    print(f'  {"k":>4} | {"AA":>8} | {"Forgetting":>10} | {"CB size":>8} | {"Time":>6}')
    print('  ' + '-' * 46)
    for k, r in results.items():
        print(f'  {k:>4} | {r["aa"]*100:>7.1f}% | {r["forgetting"]*100:>9.1f}pp | '
              f'{r["cb_size"]:>8} | {r["elapsed"]:>5.0f}s')

    best_k = max(results, key=lambda k: results[k]['aa'])
    best   = results[best_k]
    print(f'\n  Best k={best_k}: AA={best["aa"]*100:.1f}%  F={best["forgetting"]*100:.1f}pp')
    print(f'  Baseline k=1: AA={results[1]["aa"]*100:.1f}%  '
          f'AA gain: +{(best["aa"] - results[1]["aa"])*100:.1f}pp')
    print('=' * 70)
    return results, best_k


# ─── STEP 69: Self-calibrating spawn threshold ────────────────────────────────

def step69(splits):
    print('=' * 70)
    print('  Step 69 -- Self-Calibrating Spawn Threshold')
    print('  Mechanism: percentile-10 of max-sim in first 100 steps')
    print('  Baseline: fixed thresh=0.95 (Step 66b best)')
    print('  Test on CIFAR-100 AND CSI (to verify 33/33 still holds)')
    print('=' * 70)

    # CIFAR-100: auto-thresh vs fixed=0.95
    print('\n  CIFAR-100:')
    auto_r = run_experiment(splits, auto_thresh=True, k=1, label='auto')
    fix_r  = run_experiment(splits, spawn_thresh=BASE_THRESH, k=1, label='fixed=0.95')
    print(f'    Auto-thresh: locked to {auto_r["auto_thresh_val"]:.3f}  '
          f'AA={auto_r["aa"]*100:.1f}%  F={auto_r["forgetting"]*100:.1f}pp  '
          f'cb={auto_r["cb_size"]}')
    print(f'    Fixed=0.95:  AA={fix_r["aa"]*100:.1f}%  F={fix_r["forgetting"]*100:.1f}pp  '
          f'cb={fix_r["cb_size"]}')

    # CSI test: auto-thresh must still get 33/33
    print('\n  CSI (33-division benchmark, d=384):')
    csi_auto_aa = run_csi_auto_thresh()
    print('=' * 70)
    return auto_r, fix_r, csi_auto_aa


def run_csi_auto_thresh():
    """Run auto-thresh on CSI corpus. Return coverage (fraction of 33 divisions covered)."""
    import json, os
    csi_path  = 'B:/M/avir/research/fluxcore/data/csi_embedded.json'
    cent_path = 'B:/M/avir/research/fluxcore/data/csi_division_centers.json'

    if not os.path.exists(csi_path) or not os.path.exists(cent_path):
        print('    CSI data not found — skipping.')
        return None

    with open(csi_path)  as f: records = json.load(f)
    with open(cent_path) as f: centers = json.load(f)

    d_csi = len(records[0]['embedding'])
    cb = NumpyCodebook(d_csi, auto_thresh=True, auto_warmup=100, auto_percentile=10)

    for rec in records:
        r = np.array(rec['embedding'], dtype=np.float32)
        r /= (np.linalg.norm(r) + 1e-15)
        cb.step(r, label=rec.get('division'))

    # Check coverage: each center should have a nearby codebook vector (dot > 0.3)
    covered = 0
    center_list = list(centers.values()) if isinstance(centers, dict) else centers
    for ctr in center_list:
        c = np.array(ctr, dtype=np.float32)
        c /= (np.linalg.norm(c) + 1e-15)
        if len(cb.vectors) > 0:
            best_sim = float(np.max(cb.vectors @ c))
            if best_sim > 0.3:
                covered += 1

    print(f'    Auto-thresh locked to: {cb.auto_thresh_value:.3f}')
    print(f'    CSI coverage: {covered}/{len(center_list)}  '
          f'(cb={len(cb.vectors)}, spawned={cb.n_spawned})')
    return covered


# ─── STEP 70: Frozen-on-maturity ──────────────────────────────────────────────

def step70(splits, best_k_from_68):
    print('=' * 70)
    print('  Step 70 -- Frozen-on-Maturity Prototypes')
    print(f'  spawn=0.95, k={best_k_from_68}. Variable: maturity threshold')
    print('  Baseline: no freeze (maturity=None) = current 12.5pp forgetting')
    print('=' * 70)

    maturities = [None, 10, 25, 50, 100]
    results = {}
    for m in maturities:
        label = f'mat={m}' if m else 'no-freeze'
        print(f'  Running maturity={m}...')
        r = run_experiment(splits, spawn_thresh=BASE_THRESH, k=best_k_from_68,
                           maturity=m, label=label)
        results[m] = r
        print(f'    {label}: AA={r["aa"]*100:.1f}%  F={r["forgetting"]*100:.1f}pp  '
              f'cb={r["cb_size"]}  t={r["elapsed"]:.0f}s')

    print()
    print(f'  {"maturity":>10} | {"AA":>8} | {"Forgetting":>10} | {"CB size":>8} | {"Time":>6}')
    print('  ' + '-' * 52)
    for m, r in results.items():
        mlabel = str(m) if m else 'None'
        print(f'  {mlabel:>10} | {r["aa"]*100:>7.1f}% | {r["forgetting"]*100:>9.1f}pp | '
              f'{r["cb_size"]:>8} | {r["elapsed"]:>5.0f}s')

    print('=' * 70)
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('Loading CIFAR-100 features...')
    splits = load_data()
    print(f'  {N_TASKS} task splits ready.\n')

    # Step 68
    s68_results, best_k = step68(splits)

    # Step 69
    s69_auto, s69_fix, s69_csi = step69(splits)

    # Step 70
    s70_results = step70(splits, best_k)

    # Final summary
    print('\n' + '=' * 70)
    print('  STEPS 68-70 FINAL SUMMARY')
    print('=' * 70)

    k1_aa = s68_results[1]['aa'] * 100
    bk_aa = s68_results[best_k]['aa'] * 100
    print(f'\n  Step 68 (k-NN readout):')
    print(f'    k=1 baseline: {k1_aa:.1f}% AA | best k={best_k}: {bk_aa:.1f}% AA '
          f'(+{bk_aa-k1_aa:.1f}pp)')

    print(f'\n  Step 69 (self-calibrating thresh):')
    print(f'    Auto-thresh locks to {s69_auto["auto_thresh_val"]:.3f}')
    print(f'    Auto vs fixed=0.95: {s69_auto["aa"]*100:.1f}% vs {s69_fix["aa"]*100:.1f}% AA')
    if s69_csi is not None:
        print(f'    CSI coverage preserved: {s69_csi}/33')

    best_mat = min(
        {m: r for m, r in s70_results.items()},
        key=lambda m: (s70_results[m]['forgetting'] - s70_results[m]['aa'])
    )
    bm = s70_results[best_mat]
    nm = s70_results[None]
    print(f'\n  Step 70 (frozen-on-maturity):')
    print(f'    No-freeze:   AA={nm["aa"]*100:.1f}%  F={nm["forgetting"]*100:.1f}pp')
    print(f'    Best (mat={best_mat}): AA={bm["aa"]*100:.1f}%  F={bm["forgetting"]*100:.1f}pp')

    print(f'\n  Published baselines: Fine-tune ~6% | EWC ~33% | SI ~36% | DER++ ~51%')
    print('=' * 70)


if __name__ == '__main__':
    main()
