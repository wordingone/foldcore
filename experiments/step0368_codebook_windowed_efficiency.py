#!/usr/bin/env python3
"""
Step 368 -- Windowed encoding efficiency: 3 trials x 10K steps on LS20.

Windowed (k=3) + timer mask vs single-state baseline.
Kill: does windowed find level 1 in fewer lives?
Script: scripts/run_step368_windowed_efficiency.py
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
TIMER_MASK = [15 * 16 + c for c in range(8, 13)]


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

    def process_novelty(self, x, n_cls):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([0], device=self.device)
            return 0
        sims = self.V @ x
        ac = int(self.labels.max().item()) + 1
        scores = torch.zeros(max(ac, n_cls), device=self.device)
        for c in range(ac):
            m = (self.labels == c)
            if m.sum() == 0: continue
            cs = sims[m]
            scores[c] = cs.topk(min(self.k, len(cs))).values.sum()
        pred = scores[:n_cls].argmin().item()
        tm = (self.labels == pred)
        if tm.sum() == 0 or sims[tm].max() < self.thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([pred], device=self.device)])
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

def mask_timer(p):
    m = p.copy()
    for i in TIMER_MASK: m[i] = 0.0
    return m

def centered(p, fold, d=D_SINGLE):
    t = F.normalize(torch.from_numpy(p.astype(np.float32)), dim=0)
    if fold.V.shape[0] > 2:
        t = t - fold.V[:, :d].mean(dim=0).cpu()
    return t

def make_window(hist, fold):
    parts = [centered(p, fold) for p in hist[-WINDOW:]]
    while len(parts) < WINDOW:
        parts.insert(0, torch.zeros(D_SINGLE))
    return torch.cat(parts)


def run_trial(arc, game_id, windowed, max_steps=10000):
    from arcengine import GameState
    d = D_SINGLE * WINDOW if windowed else D_SINGLE
    fold = CompressedFold(d=d, k=3)
    env = arc.make(game_id)
    obs = env.reset()
    n_acts = len(env.action_space)
    total_steps = 0; go = 0; levels = 0; seeded = False
    lvl_steps = []; lvl_start = 0; history = []

    while total_steps < max_steps and go < 200:
        if obs is None or obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); history = []; lvl_start = total_steps
            if obs is None: break; continue
        if obs.state == GameState.WIN: break

        p = mask_timer(avgpool16(obs.frame))
        history.append(p)
        enc = make_window(history, fold) if windowed else centered(p, fold)

        if not seeded and fold.V.shape[0] < n_acts:
            i = fold.V.shape[0]; fold._force_add(enc, label=i)
            obs = env.step(env.action_space[i]); total_steps += 1
            if fold.V.shape[0] >= n_acts: seeded = True
            continue
        if not seeded: seeded = True

        cls = fold.process_novelty(enc, n_cls=n_acts)
        ol = obs.levels_completed
        obs = env.step(env.action_space[cls % n_acts]); total_steps += 1
        if obs is None: break
        if obs.levels_completed > ol:
            levels = obs.levels_completed
            s = total_steps - lvl_start; lvl_steps.append(s); lvl_start = total_steps
    return {'levels': levels, 'steps': total_steps, 'go': go,
            'cb': fold.V.shape[0], 'lvl_steps': lvl_steps}


def main():
    t0 = time.time()
    print("Step 368 -- Windowed efficiency: 3 trials x 10K on LS20", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(flush=True)

    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    # A. Windowed (k=3) + mask
    print("=== A. Windowed (k=3) + timer mask ===", flush=True)
    win_results = []
    for i in range(3):
        t1 = time.time()
        r = run_trial(arc, ls20.game_id, windowed=True, max_steps=10000)
        dt = time.time() - t1
        win_results.append(r)
        first = r['lvl_steps'][0] if r['lvl_steps'] else 'none'
        print(f"  Trial {i+1}: levels={r['levels']}  first={first}"
              f"  cb={r['cb']}  go={r['go']}  {dt:.1f}s", flush=True)

    # B. Single-state + mask (baseline)
    print(flush=True)
    print("=== B. Single-state + timer mask (baseline) ===", flush=True)
    single_results = []
    for i in range(3):
        t1 = time.time()
        r = run_trial(arc, ls20.game_id, windowed=False, max_steps=10000)
        dt = time.time() - t1
        single_results.append(r)
        first = r['lvl_steps'][0] if r['lvl_steps'] else 'none'
        print(f"  Trial {i+1}: levels={r['levels']}  first={first}"
              f"  cb={r['cb']}  go={r['go']}  {dt:.1f}s", flush=True)

    elapsed = time.time() - t0
    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 368 SUMMARY", flush=True)
    print("=" * 60, flush=True)

    w_levels = [r['levels'] for r in win_results]
    s_levels = [r['levels'] for r in single_results]
    w_cb = [r['cb'] for r in win_results]
    s_cb = [r['cb'] for r in single_results]

    print(f"Windowed:     levels={w_levels}  avg_cb={np.mean(w_cb):.0f}", flush=True)
    for i, r in enumerate(win_results):
        print(f"  trial {i+1}: lvl_steps={r['lvl_steps']}  go={r['go']}", flush=True)
    print(f"Single-state: levels={s_levels}  avg_cb={np.mean(s_cb):.0f}", flush=True)
    for i, r in enumerate(single_results):
        print(f"  trial {i+1}: lvl_steps={r['lvl_steps']}  go={r['go']}", flush=True)

    print(flush=True)
    w_success = sum(1 for l in w_levels if l > 0)
    s_success = sum(1 for l in s_levels if l > 0)
    print(f"Windowed P(level in 10K):     {w_success}/3", flush=True)
    print(f"Single-state P(level in 10K): {s_success}/3", flush=True)
    print(f"Windowed avg cb: {np.mean(w_cb):.0f}  (768 dims)", flush=True)
    print(f"Single avg cb:   {np.mean(s_cb):.0f}  (256 dims)", flush=True)
    print(flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
