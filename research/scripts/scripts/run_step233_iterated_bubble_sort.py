#!/usr/bin/env python3
"""
Step 233 -- Iterated bubble sort via k-NN transducer.

Hypothesis: sorting needs iteration, not richer features.
Learn 1-step bubble sort pass with k-NN. Iterate until stable.
Same mechanism that closed the CA computation gap (Step 181).

Comparison:
  1. Direct k-NN: predict full sorted output from input (Step 216 baseline: 2.8%)
  2. Iterated k-NN: predict 1-bubble-pass output, iterate to convergence

[researcher] mail 1294.
"""

import numpy as np
import time

# ── Config ────────────────────────────────────────────────────────────────────

VOCAB     = 5    # integers 0..VOCAB-1
LEN       = 4    # sequence length
K         = 5    # k-NN k
N_TRAIN   = 400  # training pairs (input, bubble_pass(input))
N_TEST    = 500
MAX_STEPS = 10   # max iteration steps
SEED      = 42


# ── Bubble sort ───────────────────────────────────────────────────────────────

def bubble_pass(seq):
    """One bubble sort pass (adjacent swaps, left to right)."""
    s = list(seq)
    for i in range(len(s) - 1):
        if s[i] > s[i + 1]:
            s[i], s[i + 1] = s[i + 1], s[i]
    return s

def is_sorted(seq):
    return all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))


# ── Data ──────────────────────────────────────────────────────────────────────

def make_data(n, seed=0):
    """Random sequences + their sorted outputs."""
    r = np.random.RandomState(seed)
    X = r.randint(0, VOCAB, (n, LEN)).astype(np.float32)
    return X

def encode_seq(seq):
    """Encode length-4 vocab-5 sequence as integer class."""
    s = [int(x) for x in seq]
    return s[0] * 125 + s[1] * 25 + s[2] * 5 + s[3]


# ── k-NN ──────────────────────────────────────────────────────────────────────

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n < 1e-10, 1.0, n)

def topk_predict_class(V, lv, Q, k=K):
    """Top-k class vote, returns predicted class labels."""
    S = norm(Q) @ norm(V).T
    classes = np.unique(lv)
    n_q = Q.shape[0]
    preds = np.zeros(n_q, dtype=int)
    for i in range(n_q):
        best_c, best_s = -1, -1e9
        for c in classes:
            idx_c = np.where(lv == c)[0]
            k_eff = min(k, len(idx_c))
            sc = np.sort(S[i, idx_c])[-k_eff:].sum()
            if sc > best_s:
                best_s, best_c = sc, c
        preds[i] = best_c
    return preds

def nn_output(V, V_out, query_row):
    """Nearest-neighbor lookup: return stored output for most similar training input."""
    sims = norm(query_row.reshape(1, -1)) @ norm(V).T  # (1, n)
    idx = int(sims[0].argmax())
    return V_out[idx].copy()


# ── Test 1: Direct full-sort prediction ───────────────────────────────────────

def test_direct(X_tr, X_te):
    """Predict sorted sequence directly from input. Baseline from Step 216."""
    # Encode sorted output as class
    y_tr = np.array([encode_seq(np.sort(row)) for row in X_tr], dtype=int)
    unique, y_tr = np.unique(y_tr, return_inverse=True)

    y_te_true = np.array([encode_seq(np.sort(row)) for row in X_te], dtype=int)
    # Re-encode test labels against training classes
    y_te = np.array([np.where(unique == c)[0][0] if c in unique else -1
                     for c in y_te_true], dtype=int)

    preds = topk_predict_class(X_tr, y_tr, X_te)
    # Compare against true sorted class (must re-decode)
    correct = 0
    for i in range(len(X_te)):
        pred_code = unique[preds[i]]
        true_code = encode_seq(np.sort(X_te[i]))
        if pred_code == true_code:
            correct += 1
    return correct / len(X_te)


# ── Test 2: Iterated 1-step bubble sort ───────────────────────────────────────

def test_iterated(X_tr, X_te, max_steps=MAX_STEPS, k_iter=1):
    """
    k-NN transducer: predict one bubble pass, iterate to convergence.

    V       = training inputs
    V_out   = corresponding bubble_pass outputs (stored alongside)
    At inference: find nearest training input, use its bubble output as next state.
    Repeat until stable.
    """
    # Build transducer training set
    V     = X_tr.copy()
    V_out = np.array([bubble_pass(row) for row in X_tr], dtype=np.float32)

    correct = 0
    step_counts = []

    for i in range(len(X_te)):
        state = X_te[i].copy()
        target = np.sort(X_te[i]).tolist()

        for step in range(max_steps):
            if is_sorted(state.tolist()):
                break
            next_state = nn_output(V, V_out, state)
            if np.array_equal(state, next_state):
                break  # stuck
            state = next_state

        step_counts.append(step)
        if state.tolist() == target:
            correct += 1

    return correct / len(X_te), float(np.mean(step_counts)), float(np.max(step_counts))


# ── Test 3: Iterated with top-k (use top-k bubble outputs, vote per position) ─

def test_iterated_topk(X_tr, X_te, max_steps=MAX_STEPS, k=K):
    """
    Like test_iterated but uses top-k neighbors and votes per output position.
    """
    V     = X_tr.copy()
    V_out = np.array([bubble_pass(row) for row in X_tr], dtype=np.float32)
    Vn    = norm(V)

    correct = 0
    for i in range(len(X_te)):
        state = X_te[i].copy()
        target = np.sort(X_te[i]).tolist()

        for step in range(max_steps):
            if is_sorted(state.tolist()):
                break
            sims = (norm(state.reshape(1, -1)) @ Vn.T)[0]
            top_k_idx = np.argsort(sims)[-k:]
            # Per-position majority vote among top-k neighbor outputs
            next_state = np.zeros(LEN, dtype=np.float32)
            for pos in range(LEN):
                vals = V_out[top_k_idx, pos].astype(int)
                counts = np.bincount(vals, minlength=VOCAB)
                next_state[pos] = counts.argmax()
            if np.array_equal(state, next_state):
                break
            state = next_state

        if state.tolist() == target:
            correct += 1

    return correct / len(X_te)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("Step 233 -- Iterated bubble sort via k-NN transducer", flush=True)
    print(f"vocab={VOCAB}, len={LEN}, n_train={N_TRAIN}, n_test={N_TEST}, "
          f"k={K}, max_steps={MAX_STEPS}", flush=True)

    rng = np.random.RandomState(SEED)
    X_tr = make_data(N_TRAIN, seed=10)
    X_te = make_data(N_TEST,  seed=11)

    n_unique_tr = len(set(tuple(r) for r in X_tr.astype(int).tolist()))
    total_possible = VOCAB ** LEN
    print(f"Training coverage: {n_unique_tr}/{total_possible} = "
          f"{n_unique_tr/total_possible*100:.1f}%", flush=True)

    # ── Test 1: Direct ────────────────────────────────────────────────────────
    print(f"\n[Test 1] Direct k-NN full-sort prediction (Step 216 baseline)", flush=True)
    t0 = time.time()
    acc_direct = test_direct(X_tr, X_te)
    print(f"  Accuracy: {acc_direct*100:.1f}%  ({time.time()-t0:.1f}s)", flush=True)

    # ── Test 2: Iterated 1-NN ─────────────────────────────────────────────────
    print(f"\n[Test 2] Iterated 1-NN bubble transducer (max {MAX_STEPS} steps)", flush=True)
    t0 = time.time()
    acc_iter1, mean_steps, max_steps = test_iterated(X_tr, X_te, k_iter=1)
    print(f"  Accuracy: {acc_iter1*100:.1f}%  mean_steps={mean_steps:.1f}  "
          f"max_steps={max_steps:.0f}  ({time.time()-t0:.1f}s)", flush=True)

    # ── Test 3: Iterated top-k (per-position vote) ───────────────────────────
    print(f"\n[Test 3] Iterated top-k bubble transducer (per-position vote)", flush=True)
    t0 = time.time()
    acc_topk = test_iterated_topk(X_tr, X_te)
    print(f"  Accuracy: {acc_topk*100:.1f}%  ({time.time()-t0:.1f}s)", flush=True)

    elapsed = time.time() - t_start

    print(f"\n{'='*60}", flush=True)
    print(f"STEP 233 FINAL", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Direct k-NN sort:           {acc_direct*100:5.1f}%  (Step 216 baseline)", flush=True)
    print(f"  Iterated 1-NN bubble:       {acc_iter1*100:5.1f}%  (mean {mean_steps:.1f} steps)", flush=True)
    print(f"  Iterated top-k bubble:      {acc_topk*100:5.1f}%", flush=True)
    print(f"\n  Gap (iter vs direct):  {(acc_iter1 - acc_direct)*100:+.1f}pp", flush=True)

    if acc_iter1 > acc_direct + 0.10:
        verdict = "CONFIRMS: iteration mechanism is general (sort = CA parallel)"
    elif acc_iter1 > acc_direct:
        verdict = "PARTIAL: iteration helps but gap remains"
    else:
        verdict = "DISPROVES: iteration does not close sort gap"
    print(f"\n  Verdict: {verdict}", flush=True)
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
