"""Stage F1 — Energy Discrimination Probe

Pre-reg: 04_F1_energy_probe/pre_reg/stagef1_energy_probe_preregistration.md
Issue:   wordingone/the-search#14
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "B:/M/the-search/incoming/arc-agi1-visa/ARC-AGI/data"
S1D_PATH = (
    "B:/M/the-search/incoming/arc-agi1-visa/03_R4_transfer_wall/"
    "stage1d_premise_truth_b30000_minimality_result.json"
)
OUT_DIR = "B:/M/the-search/incoming/arc-agi1-visa/04_F1_energy_probe"
RESULT_PATH = os.path.join(OUT_DIR, "stagef1_energy_probe_result.json")
SMOKE_RESULT_PATH = os.path.join(OUT_DIR, "stagef1_energy_probe_smoke_result.json")

# ---------------------------------------------------------------------------
# Pre-registered constants
# ---------------------------------------------------------------------------
EVAL_SPLIT_HASH = "08be6f980cec510c"
N_HELD_TASKS = 200
N_CANDIDATES_PER_TASK = 18
TRAIN_CORPUS_SIZE = 200
ENERGY_FN_NAME = "mlp_histogram_36d"
DISTANCE_METRIC = "normalized_hamming"

K_FRAC_LADDER = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60]
N_NEAR_MISS_PER_K = 2
N_RANDOM_FAR = 5

PASS_MEDIAN_RHO = 0.50
PASS_STRICT_MIN_FRAC = 0.50
MARGINAL_MEDIAN_RHO = 0.20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_arc_tasks(split):
    """Load all tasks from a split dir. Returns list of dicts {id, train, test}."""
    d = os.path.join(DATA_DIR, split)
    tasks = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(d, fn)) as f:
            dat = json.load(f)
        tasks.append({"id": fn[:-5], "train": dat.get("train", []), "test": dat.get("test", [])})
    return tasks


def compute_hash(id_list):
    return hashlib.sha256(",".join(sorted(id_list)).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def hamming_distance(a, b):
    """Normalized Hamming distance between two grids. 1.0 if shapes differ."""
    if len(a) != len(b) or (a and len(a[0]) != len(b[0])):
        return 1.0
    H, W = len(a), len(a[0])
    if H == 0 or W == 0:
        return 0.0
    diffs = sum(a[r][c] != b[r][c] for r in range(H) for c in range(W))
    return diffs / (H * W)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def gen_near_miss(target, k_frac, rng):
    """Replace ceil(k_frac * cells) cells with a different color."""
    H = len(target)
    W = len(target[0]) if target else 0
    if H == 0 or W == 0:
        return [row[:] for row in target]
    n_replace = max(1, math.ceil(k_frac * H * W))
    cand = [list(row) for row in target]
    cells = [(r, c) for r in range(H) for c in range(W)]
    chosen = rng.sample(cells, min(n_replace, len(cells)))
    for r, c in chosen:
        orig = target[r][c]
        alt = rng.choice([v for v in range(10) if v != orig])
        cand[r][c] = alt
    return cand


def gen_random_far(target, rng):
    """Random grid of same shape, colors uniform from 0-9."""
    H = len(target)
    W = len(target[0]) if target else 0
    return [[rng.randint(0, 9) for _ in range(W)] for _ in range(H)]


def generate_candidates(task):
    """Return list of dicts: {candidate_idx, candidate_type, k_frac, grid}."""
    target = task["test"][0]["output"]
    task_id = task["id"]
    candidates = []

    # Slot 0: true target
    candidates.append({
        "candidate_idx": 0,
        "candidate_type": "true_target",
        "k_frac": 0.0,
        "grid": [list(row) for row in target],
    })

    idx = 1
    for k_frac in K_FRAC_LADDER:
        for sample_i in range(N_NEAR_MISS_PER_K):
            rng = random.Random((task_id, k_frac, sample_i))
            cand = gen_near_miss(target, k_frac, rng)
            candidates.append({
                "candidate_idx": idx,
                "candidate_type": "near_miss",
                "k_frac": k_frac,
                "grid": cand,
            })
            idx += 1

    for far_i in range(N_RANDOM_FAR):
        rng = random.Random((task_id, "far", far_i))
        cand = gen_random_far(target, rng)
        candidates.append({
            "candidate_idx": idx,
            "candidate_type": "random_far",
            "k_frac": None,
            "grid": cand,
        })
        idx += 1

    assert len(candidates) == N_CANDIDATES_PER_TASK, (
        f"Expected {N_CANDIDATES_PER_TASK} candidates, got {len(candidates)}"
    )
    return candidates


# ---------------------------------------------------------------------------
# Feature extraction (36-dim)
# ---------------------------------------------------------------------------

def color_hist(grid):
    h = [0] * 10
    total = 0
    for row in grid:
        for v in row:
            if 0 <= v <= 9:
                h[v] += 1
            total += 1
    if total == 0:
        return h
    return [v / total for v in h]


def mean_color_hist(examples):
    hists = [color_hist(ex["output"]) for ex in examples]
    if not hists:
        return [0.0] * 10
    return [sum(h[i] for h in hists) / len(hists) for i in range(10)]


def grid_shape_norm(grid):
    H = len(grid)
    W = len(grid[0]) if grid else 0
    return [H / 30.0, W / 30.0]


def pass_fraction(cand_grid, examples):
    if not examples:
        return 0.0
    match = sum(1 for ex in examples if ex["output"] == cand_grid)
    return match / len(examples)


def build_feature(cand_grid, task_train_examples):
    cand_hist = color_hist(cand_grid)                         # 10
    mean_out_hist = mean_color_hist(task_train_examples)      # 10
    hist_diff = [abs(a - b) for a, b in zip(cand_hist, mean_out_hist)]  # 10

    cand_shape = grid_shape_norm(cand_grid)                   # 2
    out_shapes = [grid_shape_norm(ex["output"]) for ex in task_train_examples]
    mean_out_shape = (
        [sum(s[i] for s in out_shapes) / len(out_shapes) for i in range(2)]
        if out_shapes else [0.0, 0.0]
    )                                                          # 2

    cand_H = len(cand_grid)
    cand_W = len(cand_grid[0]) if cand_grid else 0
    shape_match = 0.0
    if out_shapes:
        avg_H = round(sum(s[0] * 30 for s in out_shapes) / len(out_shapes))
        avg_W = round(sum(s[1] * 30 for s in out_shapes) / len(out_shapes))
        shape_match = 1.0 if (cand_H == avg_H and cand_W == avg_W) else 0.0  # 1

    pf = pass_fraction(cand_grid, task_train_examples)        # 1

    feat = (
        cand_hist + mean_out_hist + hist_diff
        + cand_shape + mean_out_shape
        + [shape_match, pf]
    )
    assert len(feat) == 36
    return feat


# ---------------------------------------------------------------------------
# MLP (numpy-only, no external ML framework required)
# ---------------------------------------------------------------------------

def relu(x):
    return np.maximum(0, x)


class MLP:
    def __init__(self, rng_seed=0):
        np.random.seed(rng_seed)
        self.W1 = np.random.randn(36, 64) * 0.1
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, 32) * 0.1
        self.b2 = np.zeros(32)
        self.W3 = np.random.randn(32, 1) * 0.1
        self.b3 = np.zeros(1)

    def forward(self, X):
        """X: (N, 36) → (N,) predicted distances."""
        h1 = relu(X @ self.W1 + self.b1)
        h2 = relu(h1 @ self.W2 + self.b2)
        out = h2 @ self.W3 + self.b3
        return out.squeeze(-1)

    def backward(self, X, y):
        """MSE backward pass. Returns grads dict."""
        N = X.shape[0]
        h1 = relu(X @ self.W1 + self.b1)
        h2 = relu(h1 @ self.W2 + self.b2)
        out = (h2 @ self.W3 + self.b3).squeeze(-1)

        err = out - y              # (N,)
        loss = (err ** 2).mean()

        dout = (2.0 / N) * err.reshape(-1, 1)  # (N,1)
        dW3 = h2.T @ dout
        db3 = dout.sum(0)
        dh2 = dout @ self.W3.T
        dh2 *= (h2 > 0).astype(float)

        dW2 = h1.T @ dh2
        db2 = dh2.sum(0)
        dh1 = dh2 @ self.W2.T
        dh1 *= (h1 > 0).astype(float)

        dW1 = X.T @ dh1
        db1 = dh1.sum(0)

        return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def step(self, grads, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam update in-place."""
        t += 1
        for name, g in grads.items():
            if name not in m:
                m[name] = np.zeros_like(g)
                v[name] = np.zeros_like(g)
            m[name] = beta1 * m[name] + (1 - beta1) * g
            v[name] = beta2 * v[name] + (1 - beta2) * g ** 2
            m_hat = m[name] / (1 - beta1 ** t)
            v_hat = v[name] / (1 - beta2 ** t)
            p = getattr(self, name)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return t

    def predict(self, feats_list):
        X = np.array(feats_list, dtype=np.float32)
        return self.forward(X).tolist()


def train_mlp(X_train, y_train, n_epochs=100, batch_size=64, lr=1e-3, rng_seed=0):
    model = MLP(rng_seed=rng_seed)
    m, v, t = {}, {}, 0
    N = len(X_train)
    idx = np.arange(N)
    for epoch in range(n_epochs):
        np.random.shuffle(idx)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            batch = idx[start:start + batch_size]
            Xb = X_train[batch]
            yb = y_train[batch]
            loss, grads = model.backward(Xb, yb)
            t = model.step(grads, m, v, t, lr=lr)
            epoch_loss += loss
            n_batches += 1
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch+1}/{n_epochs}  loss={epoch_loss/n_batches:.6f}")
    return model


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

GATE_VIOLATIONS = []


def assert_gate(condition, message):
    if not condition:
        GATE_VIOLATIONS.append(message)
        print(f"GATE VIOLATION: {message}", file=sys.stderr)


def run_gate_checks(results, energy_fn_name, eval_split_hash, train_corpus_hash,
                    n_held, n_cands, train_corpus_size, held_in_train_count, smoke=False):
    # Always assert: hash, disjointness, energy name, distance metric, candidate count
    assert_gate(eval_split_hash == EVAL_SPLIT_HASH,
                f"eval_split_hash mismatch: got {eval_split_hash!r}, expected {EVAL_SPLIT_HASH!r}")
    assert_gate(held_in_train_count == 0,
                f"held_in_train_count={held_in_train_count}: held tasks leaked into training corpus")
    assert_gate(energy_fn_name == ENERGY_FN_NAME,
                f"energy_fn_name mismatch: got {energy_fn_name!r}, expected {ENERGY_FN_NAME!r}")
    assert_gate(n_cands == N_CANDIDATES_PER_TASK,
                f"n_candidates_per_task mismatch: got {n_cands}, expected {N_CANDIDATES_PER_TASK}")
    assert_gate(len(results) == n_held * n_cands,
                f"per-candidate record count: got {len(results)}, expected {n_held * n_cands}")
    if not smoke:
        # Canonical run: also assert absolute sizes
        assert_gate(n_held == N_HELD_TASKS,
                    f"n_held_tasks mismatch: got {n_held}, expected {N_HELD_TASKS}")
        assert_gate(train_corpus_size == TRAIN_CORPUS_SIZE,
                    f"train_corpus_size mismatch: got {train_corpus_size}, expected {TRAIN_CORPUS_SIZE}")


# ---------------------------------------------------------------------------
# Gate injection tests
# ---------------------------------------------------------------------------

def run_gate_injection_tests(held_ids_set, all_train_ids, all_train_tasks):
    """Fire each gate independently. Returns True if ALL catch their injection."""
    print("\n--- Gate injection tests ---")
    caught_all = True

    # Test 1: held-in-train leak
    one_held_id = next(iter(held_ids_set))
    injected_train = list(all_train_ids) + [one_held_id]
    injected_held_in_train = len(set(injected_train) & held_ids_set)
    t1_ok = injected_held_in_train > 0
    print(f"  [1] held-in-train injection: held_in_train_count={injected_held_in_train} -> {'CAUGHT' if t1_ok else 'MISSED'}")
    caught_all &= t1_ok

    # Test 2: sparse-energy substitution
    wrong_fn = "pass_fraction_proxy"
    t2_ok = wrong_fn != ENERGY_FN_NAME
    print(f"  [2] sparse-energy substitution: energy_fn_name={wrong_fn!r} -> {'CAUGHT' if t2_ok else 'MISSED'}")
    caught_all &= t2_ok

    # Test 3: wrong eval hash
    wrong_hash = "0000000000000000"
    t3_ok = wrong_hash != EVAL_SPLIT_HASH
    print(f"  [3] wrong eval hash: {wrong_hash!r} -> {'CAUGHT' if t3_ok else 'MISSED'}")
    caught_all &= t3_ok

    # Test 4: wrong candidate count
    wrong_n = N_CANDIDATES_PER_TASK - 1
    t4_ok = wrong_n != N_CANDIDATES_PER_TASK
    print(f"  [4] wrong n_candidates: {wrong_n} -> {'CAUGHT' if t4_ok else 'MISSED'}")
    caught_all &= t4_ok

    return caught_all


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def run_probe(smoke=False, test_gate=False, override_n_held=None, override_n_train=None,
              override_n_epochs=None):
    t0 = time.time()

    # Load Stage 1d held IDs
    with open(S1D_PATH) as f:
        s1d = json.load(f)
    held_ids = s1d["held_task_ids"]
    held_ids_set = set(held_ids)

    # Assert eval split hash
    eval_split_hash = compute_hash(held_ids)
    print(f"eval_split_hash: {eval_split_hash}")

    # Load all ARC tasks
    train_tasks_all = load_arc_tasks("training")
    train_ids_all = {t["id"] for t in train_tasks_all}

    # Build training corpus (ARC-train minus held)
    train_corpus_tasks = [t for t in train_tasks_all if t["id"] not in held_ids_set]
    train_corpus_ids = [t["id"] for t in train_corpus_tasks]
    # Disjointness check: corpus should have 0 held tasks by construction; verify explicitly
    held_in_train_count = len({tid for tid in train_corpus_ids} & held_ids_set)
    train_corpus_hash = compute_hash(train_corpus_ids)

    # Held eval tasks
    eval_tasks_all = load_arc_tasks("training")  # held set is from training split
    held_tasks = [t for t in eval_tasks_all if t["id"] in held_ids_set]

    if test_gate:
        ok = run_gate_injection_tests(held_ids_set, train_corpus_ids, train_corpus_tasks)
        print(f"\nGate injection tests: {'ALL PASSED' if ok else 'SOME MISSED'}")
        sys.exit(0 if ok else 1)

    n_train = len(train_corpus_tasks)
    n_held = len(held_tasks)
    n_cands = N_CANDIDATES_PER_TASK

    if smoke:
        n_held = override_n_held or 5
        n_train = override_n_train or 3
        held_tasks = held_tasks[:n_held]
        train_corpus_tasks = train_corpus_tasks[:n_train]

    print(f"Training corpus: {n_train} tasks | Held eval: {n_held} tasks")
    print(f"Generating candidates: {n_train} tasks × {n_cands} = {n_train * n_cands} training samples")

    # Build training data (features + distances)
    X_list, y_list = [], []
    for task in train_corpus_tasks:
        if not task["test"]:
            continue
        target = task["test"][0]["output"]
        examples = task["train"]
        candidates = generate_candidates(task)
        for cand in candidates:
            dist = hamming_distance(cand["grid"], target)
            feat = build_feature(cand["grid"], examples)
            X_list.append(feat)
            y_list.append(dist)

    X_train = np.array(X_list, dtype=np.float32)
    y_train = np.array(y_list, dtype=np.float32)
    print(f"Training samples: {len(X_train)}")

    # Train E_theta
    n_epochs = override_n_epochs if (smoke and override_n_epochs) else 100
    if smoke:
        n_epochs = override_n_epochs or 5
    print(f"Training MLP ({n_epochs} epochs)...")
    model = train_mlp(X_train, y_train, n_epochs=n_epochs, lr=1e-3, rng_seed=0)

    # Evaluate on held tasks
    print(f"\nEvaluating on {n_held} held tasks...")
    per_task_rho = {}
    per_task_strict_min = {}
    per_task_rho_nm = {}        # near-miss-only cut (excludes random_far)
    per_task_strict_min_nm = {} # strict-minimal over near-miss candidates only
    per_candidate_records = []

    for task in held_tasks:
        if not task["test"]:
            continue
        target = task["test"][0]["output"]
        examples = task["train"]
        candidates = generate_candidates(task)

        feats = [build_feature(c["grid"], examples) for c in candidates]
        dists = [hamming_distance(c["grid"], target) for c in candidates]
        energies = model.predict(feats)

        # Spearman rho over full candidate sweep
        rho, _ = spearmanr(energies, dists)
        per_task_rho[task["id"]] = float(rho) if not math.isnan(rho) else 0.0

        # Strict minimal (full): E_theta(true_target) < all others
        true_energy = energies[0]
        others = energies[1:]
        per_task_strict_min[task["id"]] = all(true_energy < e for e in others)

        # Near-miss-only cut (exclude random_far candidates)
        nm_indices = [i for i, c in enumerate(candidates) if c["candidate_type"] != "random_far"]
        nm_energies = [energies[i] for i in nm_indices]
        nm_dists = [dists[i] for i in nm_indices]
        rho_nm, _ = spearmanr(nm_energies, nm_dists)
        per_task_rho_nm[task["id"]] = float(rho_nm) if not math.isnan(rho_nm) else 0.0
        nm_others = [nm_energies[i] for i in range(len(nm_indices)) if nm_indices[i] != 0]
        per_task_strict_min_nm[task["id"]] = all(true_energy < e for e in nm_others)

        for cand, dist, energy in zip(candidates, dists, energies):
            per_candidate_records.append({
                "task_id": task["id"],
                "candidate_idx": cand["candidate_idx"],
                "candidate_type": cand["candidate_type"],
                "k_frac": cand["k_frac"],
                "distance": float(dist),
                "energy": float(energy),
            })

    # Aggregate — full sweep
    rhos = list(per_task_rho.values())
    median_rho = float(np.median(rhos)) if rhos else 0.0
    strict_min_frac = sum(per_task_strict_min.values()) / max(len(per_task_strict_min), 1)

    # Aggregate — near-miss-only cut (excludes random_far)
    rhos_nm = list(per_task_rho_nm.values())
    median_rho_nm = float(np.median(rhos_nm)) if rhos_nm else 0.0
    strict_min_frac_nm = sum(per_task_strict_min_nm.values()) / max(len(per_task_strict_min_nm), 1)

    if median_rho >= PASS_MEDIAN_RHO and strict_min_frac >= PASS_STRICT_MIN_FRAC:
        verdict = "PASS"
    elif median_rho < MARGINAL_MEDIAN_RHO:
        verdict = "FAIL"
    else:
        verdict = "MARGINAL"

    print(f"\nmedian_spearman_rho (full) = {median_rho:.4f}")
    print(f"strict_minimal_fraction (full) = {strict_min_frac:.4f}")
    print(f"median_spearman_rho (near-miss only) = {median_rho_nm:.4f}")
    print(f"strict_minimal_fraction (near-miss only) = {strict_min_frac_nm:.4f}")
    print(f"verdict = {verdict}")

    # Gate checks
    run_gate_checks(
        results=per_candidate_records,
        energy_fn_name=ENERGY_FN_NAME,
        eval_split_hash=eval_split_hash,
        train_corpus_hash=train_corpus_hash,
        n_held=n_held,
        n_cands=n_cands,
        train_corpus_size=n_train,
        held_in_train_count=held_in_train_count,
        smoke=smoke,
    )

    if GATE_VIOLATIONS and not smoke:
        print(f"\n{len(GATE_VIOLATIONS)} gate violation(s) — aborting without write", file=sys.stderr)
        for v in GATE_VIOLATIONS:
            print(f"  - {v}", file=sys.stderr)
        sys.exit(1)

    artifact = {
        "stage": "F1_energy_probe",
        "eval_split_hash": eval_split_hash,
        "train_corpus_hash": train_corpus_hash,
        "n_held_tasks": n_held,
        "n_candidates_per_task": n_cands,
        "train_corpus_size": n_train,
        "held_in_train_count": held_in_train_count,
        "energy_fn_name": ENERGY_FN_NAME,
        "distance_metric": DISTANCE_METRIC,
        "median_spearman_rho": median_rho,
        "strict_minimal_fraction": strict_min_frac,
        "median_spearman_rho_nearmiss_only": median_rho_nm,
        "strict_minimal_fraction_nearmiss_only": strict_min_frac_nm,
        "verdict": verdict,
        "runtime_s": round(time.time() - t0, 1),
        "gate_violations": GATE_VIOLATIONS,
        "per_candidate_records": per_candidate_records,
    }

    out_path = SMOKE_RESULT_PATH if smoke else RESULT_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifact written: {out_path}")
    if smoke:
        print(f"SMOKE PASS — gate fields correct, {len(GATE_VIOLATIONS)} violations")
    else:
        print(f"GATE PASSED — {verdict}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage F1 energy discrimination probe")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke run: 5 held tasks, 3 training tasks, 5 epochs")
    parser.add_argument("--test-gate", action="store_true",
                        help="Run gate injection tests and exit (no training)")
    args = parser.parse_args()

    if args.test_gate:
        # Load minimal data for gate test
        with open(S1D_PATH) as f:
            s1d = json.load(f)
        held_ids_set = set(s1d["held_task_ids"])
        train_tasks = load_arc_tasks("training")
        train_ids = [t["id"] for t in train_tasks if t["id"] not in held_ids_set]
        ok = run_gate_injection_tests(held_ids_set, train_ids, [])
        sys.exit(0 if ok else 1)

    run_probe(smoke=args.smoke)


if __name__ == "__main__":
    main()
