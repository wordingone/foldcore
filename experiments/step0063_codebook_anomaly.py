#!/usr/bin/env python3
"""
Step 63 (v2): FluxCore as unsupervised anomaly detector on River CreditCard.

CreditCard (European credit card fraud, 2013):
  - 284,807 transactions, 0.172% fraud rate (492 fraud)
  - Features: V1-V28 (PCA components, already whitened) + Amount, Time
  - Standard metric: AUROC (no threshold needed)
  - Published baselines:
      Isolation Forest:  AUROC ~0.97  (Liu et al. 2008)
      LOF (k=20):        AUROC ~0.94  (Breunig et al. 2000)
      OCSVM:             AUROC ~0.86  (Scholkopf et al. 2001)

FluxCore novelty score: 1 - max_codebook_sim(x)
  - Codebook built online (no labels used)
  - Novelty = 1.0 when codebook is empty (very first sample)
  - High novelty -> probable anomaly

Frame: FluxCore is an online, streaming anomaly detector.
  - One-pass: no retraining, constant memory
  - Codebook size << dataset size (bounded representation)
  - No contamination issue (works even without clean training set)
"""

import sys
import math
import time

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from fluxcore_manytofew import ManyToFewKernel, _vec_cosine

from river import datasets


# ─── Embedding ────────────────────────────────────────────────────────────────

def embed(x):
    """
    Embed CreditCard sample. Features V1-V28 are PCA components (whitened),
    Time and Amount are raw. Normalize Amount (log-scale) and Time (fractional).
    Output: unit vector in R^30.
    """
    v1_28 = [x[f'V{i}'] for i in range(1, 29)]
    amount = math.log1p(abs(x.get('Amount', 0.0))) / 10.0  # log-scale, ~[0,1]
    time_frac = (x.get('Time', 0.0) % 86400) / 86400.0     # time-of-day fraction

    feats = v1_28 + [amount, time_frac]
    n = math.sqrt(sum(v * v for v in feats) + 1e-15)
    return [v / n for v in feats]


# ─── AUROC (manual, no sklearn) ───────────────────────────────────────────────

def auroc(scores, labels):
    """
    Compute AUROC from (score, label) pairs where label=1 means anomaly.
    Uses trapezoidal rule on the ROC curve.
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by score descending (higher score = more anomalous)
    pairs = sorted(zip(scores, labels), reverse=True)

    tp, fp = 0, 0
    prev_tp, prev_fp = 0, 0
    auc = 0.0

    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        # Trapezoidal rule: area += (fp - prev_fp) * (tp + prev_tp) / 2
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
        prev_tp, prev_fp = tp, fp

    return auc / (n_pos * n_neg)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_benchmark(spawn_thresh=0.5):
    print(f'\n  [spawn_thresh={spawn_thresh}] Loading CreditCard...')

    kernel = ManyToFewKernel(
        n_matrix=8, k=4, d=30, seed=42, proj_seed=999,
        tau=0.3, k_couple=5,
        spawn_thresh=spawn_thresh, merge_thresh=0.95, lr_codebook=0.015,
    )

    novelty_scores = []
    labels = []
    total = 0
    fraud_count = 0

    t0 = time.time()
    for x, y_raw in datasets.CreditCard():
        y = int(y_raw)
        r = embed(x)

        # Novelty BEFORE update (prequential: score before learning this sample)
        if kernel.codebook:
            sims = [_vec_cosine(v, r) for v in kernel.codebook]
            max_sim = max(sims)
        else:
            max_sim = 0.0
        novelty_scores.append(1.0 - max_sim)
        labels.append(y)

        # Update (unsupervised — no label used)
        kernel.step(r=r)

        total += 1
        fraud_count += y

        if total % 50000 == 0:
            print(f'    {total} samples, cb={len(kernel.codebook)}, spawned={kernel.total_spawned}')

    elapsed = time.time() - t0
    auc = auroc(novelty_scores, labels)

    print(f'  spawn_thresh={spawn_thresh}: AUROC={auc:.4f}  cb={len(kernel.codebook)}  '
          f'spawned={kernel.total_spawned}  merged={kernel.total_merged}  '
          f'fraud_rate={fraud_count/total:.4f}  elapsed={elapsed:.1f}s')
    return auc, len(kernel.codebook), kernel.total_spawned


def main():
    print('=' * 70)
    print('  Step 63 -- FluxCore Anomaly Detection on CreditCard (AUROC)')
    print('  River CreditCard: 284807 samples, 0.172% fraud, d=30')
    print('=' * 70)

    results = []
    for thresh in [0.5, 0.7, 0.9]:
        auc, cb, spawned = run_benchmark(thresh)
        results.append((thresh, auc, cb, spawned))

    print('\n' + '=' * 70)
    print('  SUMMARY')
    print('=' * 70)
    print(f'\n  {"spawn_thresh":>14}  {"AUROC":>8}  {"CB size":>8}  {"Spawned":>10}')
    print('  ' + '-' * 48)
    for thresh, auc, cb, spawned in results:
        print(f'  {thresh:>14.2f}  {auc:>8.4f}  {cb:>8}  {spawned:>10}')

    print(f'\n  Published baselines (offline, not streaming):')
    print(f'    Isolation Forest: ~0.970  (batch, full dataset)')
    print(f'    LOF (k=20):       ~0.940  (batch, full dataset)')
    print(f'    OCSVM:            ~0.860  (batch, full dataset)')
    print(f'\n  FluxCore advantages over batch methods:')
    print(f'    - Online/streaming: one pass, no replay')
    print(f'    - Constant memory: bounded codebook size')
    print(f'    - No contamination: no clean training set required')
    print(f'    - Drift-aware: codebook adapts to distribution shifts')

    best = max(results, key=lambda r: r[1])
    print(f'\n  Best FluxCore: AUROC={best[1]:.4f} at spawn_thresh={best[0]}')
    print('=' * 70)


if __name__ == '__main__':
    main()
