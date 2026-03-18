#!/usr/bin/env python3
"""
Constraint Map Validation Round C — process_novelty() baseline + 6 variants.
Mail 1753. Jun-approved 50K runs.

Run 0:  Baseline (process_novelty() exact) — MUST navigate Level 1
Reverse tests (remove V-derivedness):
  R1: Fixed thresh=0.95 (no Gram median)
  R2: Fixed lr=0.05 (no 1-sim)
  R3: Fixed thresh=0.95 AND fixed lr=0.05
Forward tests (what's unique to process_novelty vs SelfRef):
  F1: argmin -> index-based action (w % n_actions)
  F2: No centering (raw normalized input)
  F3: k=1 instead of k=3
"""

import sys, time, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 256
MAX_STEPS = 50000
CHECKPOINTS = {10000, 25000, 50000}


# ==============================================================================
# Base CompressedFold (exact Step 353 baseline)
# ==============================================================================

class CompressedFold:
    def __init__(self, d, k=3, device=DEVICE, max_cb=4096):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7; self.k = k; self.d = d; self.device = device
        self.max_cb = max_cb

    def _force_add(self, x, label):
        x_n = F.normalize(x.to(self.device).float(), dim=0)
        self.V = torch.cat([self.V, x_n.unsqueeze(0)])
        self.labels = torch.cat([self.labels, torch.tensor([label], device=self.device)])
        self._update_thresh()

    def _update_thresh(self):
        n = self.V.shape[0]
        if n < 2: return
        G = self.V @ self.V.T
        G.fill_diagonal_(-float('inf'))
        self.thresh = float(G.max(dim=1).values.median())

    def process_novelty(self, x, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            sl = label if label is not None else 0
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([sl], device=self.device)
            return sl
        sims = self.V @ x
        n_cls = int(self.labels.max().item()) + 1
        scores = torch.zeros(n_cls, device=self.device)
        for c in range(n_cls):
            mask = (self.labels == c)
            if mask.sum() == 0: continue
            cs = sims[mask]
            scores[c] = cs.topk(min(self.k, len(cs))).values.sum()
        prediction = scores.argmin().item()
        sl = label if label is not None else prediction
        target_mask = (self.labels == prediction)
        if target_mask.sum() == 0 or sims[target_mask].max() < self.thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([sl], device=self.device)])
        else:
            ts = sims.clone(); ts[~target_mask] = -float('inf')
            winner = ts.argmax().item()
            alpha = 1.0 - float(sims[winner].item())
            self.V[winner] = F.normalize(self.V[winner] + alpha * (x - self.V[winner]), dim=0)
        self._update_thresh()
        return prediction


# ==============================================================================
# Reverse test variants
# ==============================================================================

class CF_FixedThresh(CompressedFold):
    """R1: Fixed thresh=0.95. No Gram median update."""
    def __init__(self, d, k=3, device=DEVICE):
        super().__init__(d, k, device)
        self.thresh = 0.95
    def _update_thresh(self):
        pass  # Never update — stay fixed at 0.95


class CF_FixedLR(CompressedFold):
    """R2: Fixed lr=0.05 in attract. 1-sim replaced by constant."""
    def process_novelty(self, x, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            sl = label if label is not None else 0
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([sl], device=self.device)
            return sl
        sims = self.V @ x
        n_cls = int(self.labels.max().item()) + 1
        scores = torch.zeros(n_cls, device=self.device)
        for c in range(n_cls):
            mask = (self.labels == c)
            if mask.sum() == 0: continue
            cs = sims[mask]
            scores[c] = cs.topk(min(self.k, len(cs))).values.sum()
        prediction = scores.argmin().item()
        sl = label if label is not None else prediction
        target_mask = (self.labels == prediction)
        if target_mask.sum() == 0 or sims[target_mask].max() < self.thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([sl], device=self.device)])
        else:
            ts = sims.clone(); ts[~target_mask] = -float('inf')
            winner = ts.argmax().item()
            alpha = 0.05  # FIXED lr instead of 1-sim
            self.V[winner] = F.normalize(self.V[winner] + alpha * (x - self.V[winner]), dim=0)
        self._update_thresh()
        return prediction


class CF_FixedBoth(CF_FixedThresh):
    """R3: Fixed thresh=0.95 AND fixed lr=0.05."""
    def process_novelty(self, x, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            sl = label if label is not None else 0
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([sl], device=self.device)
            return sl
        sims = self.V @ x
        n_cls = int(self.labels.max().item()) + 1
        scores = torch.zeros(n_cls, device=self.device)
        for c in range(n_cls):
            mask = (self.labels == c)
            if mask.sum() == 0: continue
            cs = sims[mask]
            scores[c] = cs.topk(min(self.k, len(cs))).values.sum()
        prediction = scores.argmin().item()
        sl = label if label is not None else prediction
        target_mask = (self.labels == prediction)
        if target_mask.sum() == 0 or sims[target_mask].max() < self.thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([sl], device=self.device)])
        else:
            ts = sims.clone(); ts[~target_mask] = -float('inf')
            winner = ts.argmax().item()
            alpha = 0.05  # fixed
            self.V[winner] = F.normalize(self.V[winner] + alpha * (x - self.V[winner]), dim=0)
        # _update_thresh is no-op (inherited from CF_FixedThresh)
        return prediction


# ==============================================================================
# Forward test variants
# ==============================================================================

class CF_IndexAction(CompressedFold):
    """F1: argmin -> nearest-neighbor index % n_actions. No class scoring."""
    def process_novelty(self, x, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([0], device=self.device)
            return 0
        sims = self.V @ x
        winner = sims.argmax().item()
        prediction = winner  # index-based, not class-based
        if sims[winner] < self.thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([self.V.shape[0]-1], device=self.device)])
        else:
            alpha = 1.0 - float(sims[winner].item())
            self.V[winner] = F.normalize(self.V[winner] + alpha * (x - self.V[winner]), dim=0)
        self._update_thresh()
        return prediction


class CF_NoCentering(CompressedFold):
    """F2: No centering. Raw F.normalize(pooled) input."""
    pass  # centering is in the test runner, not the substrate


class CF_K1(CompressedFold):
    """F3: k=1 instead of k=3. Single nearest entry per class."""
    def __init__(self, d, device=DEVICE):
        super().__init__(d, k=1, device=device)


# ==============================================================================
# Encoding
# ==============================================================================

def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(pooled, fold):
    t = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if fold.V.shape[0] > 2:
        mean_V = fold.V.mean(dim=0).cpu()
        t_unit = t_unit - mean_V
    return t_unit


def raw_enc(pooled):
    """F2: no centering."""
    t = torch.from_numpy(pooled.astype(np.float32))
    return F.normalize(t, dim=0)


# ==============================================================================
# Runner
# ==============================================================================

def run_variant(run_id, name, fold, arc, ls20_game, use_centering=True):
    from arcengine import GameState
    print(f"\n{'='*60}")
    print(f"RUN {run_id}: {name}", flush=True)
    print(f"{'='*60}", flush=True)

    env = arc.make(ls20_game.game_id)
    obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0
    unique = set()
    action_counts = [0] * 4
    seeded = False
    t0 = time.time()

    while ts < MAX_STEPS:
        if obs is None:
            obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); seeded = False
            if obs is None: break
            continue
        if obs.state == GameState.WIN:
            print(f"WIN at step {ts}!", flush=True); break

        pooled = avgpool16(obs.frame)
        unique.add(hash(pooled.tobytes()))
        x = centered_enc(pooled, fold) if use_centering else raw_enc(pooled)

        if not seeded and fold.V.shape[0] < na:
            i = fold.V.shape[0]
            fold._force_add(x, label=i)
            action = env.action_space[i % na]
            data = {}
            if action.is_complex():
                arr = np.array(obs.frame[0])
                cy, cx = divmod(int(np.argmax(arr)), 64)
                data = {"x": cx, "y": cy}
            obs_before = obs.levels_completed
            obs = env.step(action, data=data); ts += 1
            action_counts[i % na] += 1
            if fold.V.shape[0] >= na: seeded = True
            if obs is not None and obs.levels_completed > obs_before: lvls = obs.levels_completed
            continue
        if not seeded: seeded = True

        cls_used = fold.process_novelty(x, label=None)
        action = env.action_space[cls_used % na]
        action_counts[cls_used % na] += 1
        data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0])
            cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}
        obs_before = obs.levels_completed
        obs = env.step(action, data=data); ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            print(f"LEVEL {lvls} at step {ts}  cb={fold.V.shape[0]}", flush=True)

        if ts in CHECKPOINTS:
            elapsed = time.time() - t0
            total = sum(action_counts)
            dom = max(action_counts) / total * 100 if total else 0
            print(f"\n--- step {ts} | {elapsed:.0f}s ---")
            print(f"  unique={len(unique)}  levels={lvls}  go={go}  dom={dom:.0f}%")
            print(f"  cb={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)

    elapsed = time.time() - t0
    total = sum(action_counts)
    dom = max(action_counts) / total * 100 if total else 0
    print(f"\n--- FINAL {name} ---")
    print(f"  unique={len(unique)}  levels={lvls}  go={go}  dom={dom:.0f}%  cb={fold.V.shape[0]}  {elapsed:.0f}s")
    return {'run': run_id, 'name': name, 'unique': len(unique), 'levels': lvls,
            'cb': fold.V.shape[0], 'dom': dom, 'elapsed': elapsed}


def main():
    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: ls20 not found"); return

    print("Constraint Map Round C -- process_novelty() baseline + 6 variants")
    print("Run 0 must navigate Level 1 to validate analysis.")

    runs = [
        ("0",  "Baseline (process_novelty exact)", CompressedFold(d=D_ENC), True),
        ("R1", "FixedThresh=0.95",                 CF_FixedThresh(d=D_ENC), True),
        ("R2", "FixedLR=0.05",                     CF_FixedLR(d=D_ENC),    True),
        ("R3", "FixedThresh+FixedLR",              CF_FixedBoth(d=D_ENC),  True),
        ("F1", "IndexAction (w%n, no class score)", CF_IndexAction(d=D_ENC),True),
        ("F2", "NoCentering (raw F.normalize)",     CF_NoCentering(d=D_ENC),False),
        ("F3", "k=1 (single NN per class)",         CF_K1(d=D_ENC),         True),
    ]

    results = []
    for run_id, name, fold, use_centering in runs:
        r = run_variant(run_id, name, fold, arc, ls20, use_centering)
        results.append(r)
        if run_id == "0" and r['levels'] == 0:
            print("\nWARNING: Baseline did not navigate. Analysis invalid. Continuing anyway.")

    print(f"\n{'='*60}")
    print("SUMMARY -- Constraint Map Round C")
    print(f"{'='*60}")
    print(f"{'Run':<4} {'Name':<40} {'unique':>7} {'levels':>7} {'cb':>6} {'dom':>6}")
    print(f"{'-'*75}")
    for r in results:
        nav = f"L{r['levels']}" if r['levels'] > 0 else "0"
        print(f"{r['run']:<4} {r['name']:<40} {r['unique']:>7} {nav:>7} {r['cb']:>6} {r['dom']:>5.0f}%")

    base = results[0]
    print(f"\nBaseline: unique={base['unique']}  levels={base['levels']}  cb={base['cb']}")
    if base['levels'] > 0:
        print("Baseline navigated. Analysis valid.")
        for r in results[1:]:
            if r['levels'] > 0:
                print(f"  {r['run']} navigates -> that element is NOT forced (genuine U or M)")
            else:
                print(f"  {r['run']} fails -> that element IS load-bearing")


if __name__ == '__main__':
    main()
