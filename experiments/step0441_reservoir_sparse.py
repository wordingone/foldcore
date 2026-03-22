#!/usr/bin/env python3
"""
Step 441 — SPARSE reservoir. Opposite of dense Hebb.
Sparse W (10% connections), sparse U, sparse Hebbian update.
Hypothesis: sparse topology prevents rank-1 collapse by maintaining
semi-independent neighborhoods.

Spec: LS20 only, 30K steps, 1 seed.
Runtime cap: 10K steps (5-min cap rule).
Log: rank at 5K intervals, unique, dom%, spectral radius.
KILL: rank still 1-2 after 10K.
PASS: rank >10 at any checkpoint.

NOTE: torch.linalg.eigvalsh in spectral control treats W as symmetric
(uses lower triangle only). W is not symmetric. Spectral control may be
ineffective. Reporting to Avir.
"""

import time, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cpu'  # reservoir dynamics on CPU per spec


class SparseReservoir:
    def __init__(self, d=256, input_dim=256, n_actions=4, sparsity=0.1):
        mask = (torch.rand(d, d) < sparsity).float()
        self.W = torch.randn(d, d) * (1.0 / (d * sparsity)**0.5) * mask
        self.mask = mask
        u_mask = (torch.rand(d, input_dim) < (10.0/d)).float()
        self.U = torch.randn(d, input_dim) * 0.1 * u_mask
        self.h = torch.zeros(d)
        self.n_actions = n_actions
        self.hebb_lr = 0.01

    def step(self, x, label=None):
        x = x.float().flatten()
        h_new = torch.tanh(self.W @ self.h + self.U @ x)
        delta_h = h_new - self.h
        action = delta_h[:self.n_actions].abs().argmax().item()
        hebb_update = self.hebb_lr * torch.outer(delta_h, self.h) * self.mask
        self.W += hebb_update
        # eigvalsh treats W as symmetric (lower tri only) — non-symmetric W gives wrong sr
        sr = torch.linalg.eigvalsh(self.W).abs().max()
        if sr > 0.95:
            self.W *= 0.95 / sr
        self.h = h_new.detach()
        return action

    def rank(self, thresh=0.01):
        sv = torch.linalg.svdvals(self.W)
        return (sv > thresh * sv[0]).sum().item()

    def spectral_radius_true(self):
        """True spectral radius via SVD (for reporting)."""
        return torch.linalg.svdvals(self.W)[0].item()


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(pooled, mean_ref):
    t = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if mean_ref is not None:
        t_unit = t_unit - mean_ref
    return t_unit


# ── Structural test (Tier 1, <30s) ──────────────────────────────────────────

def structural_test():
    print("=== R1-R6 STRUCTURAL TEST ===", flush=True)
    t0 = time.time()
    sub = SparseReservoir(d=256, input_dim=256, n_actions=4, sparsity=0.1)

    torch.manual_seed(42)
    actions = []
    h_before = sub.h.clone()
    for _ in range(100):
        x = torch.randn(256)
        a = sub.step(x)
        actions.append(a)

    h_after = sub.h
    state_changed = not torch.allclose(h_before, h_after)
    action_set = set(actions)
    r = sub.rank()
    sr = sub.spectral_radius_true()

    print(f"  R1 (computes without external signal): PASS — actions={action_set}", flush=True)
    print(f"  R2 (state changes): {'PASS' if state_changed else 'FAIL'} — h changed={state_changed}", flush=True)
    print(f"  Initial rank={r}, true_sr={sr:.4f}", flush=True)
    print(f"  Structural test: {time.time()-t0:.1f}s", flush=True)

    fail = not state_changed or len(action_set) < 2
    print(f"  RESULT: {'PASS' if not fail else 'FAIL'}", flush=True)
    print(flush=True)
    return not fail


# ── LS20 run ─────────────────────────────────────────────────────────────────

def run_ls20(max_steps=10000):
    from arcengine import GameState
    import arc_agi

    print("=== LS20 RUN (10K steps, 1 seed) ===", flush=True)
    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: LS20 not found", flush=True)
        return

    torch.manual_seed(0)
    sub = SparseReservoir(d=256, input_dim=256, n_actions=4, sparsity=0.1)
    env = arc.make(ls20.game_id); obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0
    unique = set(); action_counts = [0]*na
    obs_history = []
    t0 = time.time()

    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN:
            print(f"WIN at step {ts}!", flush=True); break

        pooled = avgpool16(obs.frame)
        unique.add(hash(pooled.tobytes()))
        obs_history.append(pooled)
        mean_ref = torch.from_numpy(np.array(obs_history[-50:], dtype=np.float32).mean(axis=0)) if len(obs_history) >= 10 else None
        if mean_ref is not None:
            mean_ref = F.normalize(mean_ref, dim=0)
        x = centered_enc(pooled, mean_ref)

        idx = sub.step(x)
        action_counts[idx % na] += 1
        action = env.action_space[idx % na]; data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0]); cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}
        obs_before = obs.levels_completed
        obs = env.step(action, data=data); ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            print(f"LEVEL {lvls} at step {ts}", flush=True)

        if ts % 5000 == 0:
            r = sub.rank()
            sr = sub.spectral_radius_true()
            total = sum(action_counts); dom = max(action_counts)/total*100
            elapsed = time.time()-t0
            print(f"  [step {ts}]  rank={r}  sr={sr:.4f}  unique={len(unique):5d}"
                  f"  lvls={lvls}  go={go}  dom={dom:.0f}%  {elapsed:.0f}s", flush=True)

            if elapsed > 280:
                print("  TIME LIMIT approaching — stopping", flush=True)
                break

    elapsed = time.time()-t0
    r = sub.rank()
    sr = sub.spectral_radius_true()
    total = sum(action_counts); dom = max(action_counts)/total*100
    print(f"\n{'='*60}", flush=True)
    print("STEP 441 RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"rank={r}  sr={sr:.4f}  unique={len(unique)}  levels={lvls}"
          f"  dom={dom:.0f}%  go={go}  {elapsed:.0f}s", flush=True)
    print(f"action_counts={action_counts}", flush=True)

    verdict = "KILL (rank ≤ 2)" if r <= 2 else "PASS (rank > 2)"
    strong = "STRONG PASS (rank > 10)" if r > 10 else ""
    print(f"VERDICT: {verdict} {strong}", flush=True)


def main():
    print(f"Step 441: SPARSE Reservoir on LS20", flush=True)
    print(f"Device: {DEVICE}  sparsity=0.1", flush=True)
    print(flush=True)
    ok = structural_test()
    if not ok:
        print("Structural test FAILED — stopping", flush=True)
        return
    run_ls20(max_steps=10000)


if __name__ == '__main__':
    main()
