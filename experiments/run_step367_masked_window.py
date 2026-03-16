#!/usr/bin/env python3
"""
Step 367 -- Timer mask + windowed encoding on LS20.

1. Mask timer cells (row 15, cols 8-12 at 16x16 = 5 cells) to 0.
2. 367a: single-state + mask (2K steps). Verify timer suppressed.
3. 367b: windowed (k=3) + mask (2K steps). Kill: unique_windows > 1.5x unique_states.

Script: scripts/run_step367_masked_window.py
"""

import time
import logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_SINGLE = 256
WINDOW = 3

# Timer mask: row 15, cols 8-12 in 16x16 grid
TIMER_MASK = []
for col in range(8, 13):
    TIMER_MASK.append(15 * 16 + col)  # row 15, col 8-12


class CompressedFold:
    def __init__(self, d, k=3, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7
        self.k, self.d, self.device = k, d, device

    def _force_add(self, x, label):
        x_n = F.normalize(x.to(self.device).float(), dim=0)
        self.V = torch.cat([self.V, x_n.unsqueeze(0)])
        self.labels = torch.cat([self.labels, torch.tensor([label], device=self.device)])
        self._update_thresh()

    def _update_thresh(self):
        n = self.V.shape[0]
        if n < 2: return
        ss = min(500, n)
        idx = torch.randperm(n, device=self.device)[:ss]
        sims = self.V[idx] @ self.V.T
        topk = sims.topk(min(2, n), dim=1).values
        self.thresh = float((topk[:, 1] if topk.shape[1] >= 2 else topk[:, 0]).median())

    def process_novelty(self, x, n_cls, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            sl = label if label is not None else 0
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([sl], device=self.device)
            return sl
        sims = self.V @ x
        ac = int(self.labels.max().item()) + 1
        scores = torch.zeros(max(ac, n_cls), device=self.device)
        for c in range(ac):
            m = (self.labels == c)
            if m.sum() == 0: continue
            cs = sims[m]
            scores[c] = cs.topk(min(self.k, len(cs))).values.sum()
        pred = scores[:n_cls].argmin().item()
        sl = label if label is not None else pred
        tm = (self.labels == pred)
        if tm.sum() == 0 or sims[tm].max() < self.thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([sl], device=self.device)])
            self._update_thresh()
        else:
            ts = sims.clone(); ts[~tm] = -float('inf')
            w = ts.argmax().item()
            a = 1.0 - float(sims[w].item())
            self.V[w] = F.normalize(self.V[w] + a * (x - self.V[w]), dim=0)
        return pred


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def mask_timer(pooled):
    """Zero out timer cells."""
    masked = pooled.copy()
    for idx in TIMER_MASK:
        masked[idx] = 0.0
    return masked


def centered_enc(pooled, fold):
    t = F.normalize(torch.from_numpy(pooled.astype(np.float32)), dim=0)
    if fold.V.shape[0] > 2:
        mean_V = fold.V[:, :D_SINGLE].mean(dim=0).cpu()
        t = t - mean_V
    return t


def make_window(history, fold):
    parts = []
    for p in history[-WINDOW:]:
        parts.append(centered_enc(p, fold))
    while len(parts) < WINDOW:
        parts.insert(0, torch.zeros(D_SINGLE))
    return torch.cat(parts)


def run_trial(arc, game_id, windowed=False, max_steps=2000, label=""):
    from arcengine import GameState

    d = D_SINGLE * WINDOW if windowed else D_SINGLE
    fold = CompressedFold(d=d, k=3)
    env = arc.make(game_id)
    obs = env.reset()
    n_acts = len(env.action_space)

    total_steps = 0; go = 0; levels = 0; seeded = False
    unique_states = set(); unique_windows = set()
    action_counts = {}; history = []

    while total_steps < max_steps and go < 50:
        if obs is None or obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); history = []
            if obs is None: break; continue
        if obs.state == GameState.WIN: break

        curr_pooled = mask_timer(avgpool16(obs.frame))
        unique_states.add(hash(curr_pooled.tobytes()))
        history.append(curr_pooled)

        if windowed:
            enc = make_window(history, fold)
            unique_windows.add(hash(enc.numpy().tobytes()))
        else:
            enc = centered_enc(curr_pooled, fold)

        if not seeded and fold.V.shape[0] < n_acts:
            i = fold.V.shape[0]; fold._force_add(enc, label=i)
            obs = env.step(env.action_space[i]); total_steps += 1
            action_counts[env.action_space[i].name] = action_counts.get(env.action_space[i].name, 0) + 1
            if fold.V.shape[0] >= n_acts: seeded = True
            continue
        if not seeded: seeded = True

        cls = fold.process_novelty(enc, n_cls=n_acts)
        action = env.action_space[cls % n_acts]
        action_counts[action.name] = action_counts.get(action.name, 0) + 1

        ol = obs.levels_completed
        obs = env.step(action); total_steps += 1
        if obs is None: break
        if obs.levels_completed > ol:
            levels = obs.levels_completed
            print(f"    [{label}] LEVEL {levels} at step {total_steps} cb={fold.V.shape[0]}", flush=True)

        if total_steps % 500 == 0:
            print(f"    [{label}] step {total_steps:5d} cb={fold.V.shape[0]}"
                  f"  unique_s={len(unique_states)}"
                  f"  unique_w={len(unique_windows) if windowed else 'N/A'}"
                  f"  go={go}  acts={action_counts}", flush=True)

    return {
        'label': label, 'levels': levels, 'steps': total_steps, 'go': go,
        'cb': fold.V.shape[0], 'thresh': fold.thresh,
        'unique_states': len(unique_states),
        'unique_windows': len(unique_windows) if windowed else 0,
        'action_counts': action_counts,
    }


def main():
    t0 = time.time()
    print("Step 367 -- Timer mask + windowed encoding on LS20", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Timer mask: {len(TIMER_MASK)} cells (row 15, cols 8-12)", flush=True)
    print(flush=True)

    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    # 367a: single state + mask
    print("=== 367a: single state + timer mask (2K steps) ===", flush=True)
    r_a = run_trial(arc, ls20.game_id, windowed=False, max_steps=2000, label="367a")
    print(f"\n  367a: unique_states={r_a['unique_states']}  cb={r_a['cb']}", flush=True)

    # 367b: windowed + mask
    print(flush=True)
    print("=== 367b: windowed (k=3) + timer mask (2K steps) ===", flush=True)
    r_b = run_trial(arc, ls20.game_id, windowed=True, max_steps=2000, label="367b")
    ratio = r_b['unique_windows'] / max(r_b['unique_states'], 1)
    pass_b = ratio > 1.5
    print(f"\n  367b: unique_windows={r_b['unique_windows']}"
          f"  unique_states={r_b['unique_states']}  ratio={ratio:.2f}x", flush=True)
    print(f"  367b {'PASS' if pass_b else 'KILL'}: ratio>1.5={pass_b}", flush=True)

    elapsed = time.time() - t0
    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 367 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"367a: unique_states={r_a['unique_states']}  cb={r_a['cb']}"
          f"  actions={r_a['action_counts']}", flush=True)
    print(f"367b: unique_windows={r_b['unique_windows']}  unique_states={r_b['unique_states']}"
          f"  ratio={ratio:.2f}x  {'PASS' if pass_b else 'KILL'}", flush=True)
    print(f"  actions={r_b['action_counts']}", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
