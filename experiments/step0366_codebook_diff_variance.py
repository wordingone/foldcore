#!/usr/bin/env python3
"""
Step 366 -- Diff-variance weighting + optional windowed encoding.

366a: Weights from variance of FRAME DIFFS (not raw values).
  Timer diffs constant → suppressed. Sprite diffs variable → amplified.
366b: Windowed (k=3) with diff-variance weights (if 366a passes).

Kill 366a: timer cells (row 15) weight < 0.1, sprite cells (rows 9-11) weight > 0.3.
Kill 366b: unique_windows > 1.5x unique_states.

2K steps on LS20.
Script: scripts/run_step366_diff_variance.py
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


class DiffVarianceFold:
    def __init__(self, d, k=3, device=DEVICE):
        self.V      = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7
        self.k, self.d, self.device = k, d, device
        self.weights = torch.ones(D_SINGLE, device=device)
        self.diff_history = []  # store frame diffs for variance computation
        self.step_count = 0

    def _force_add(self, x, label):
        x_n = F.normalize(x.to(self.device).float(), dim=0)
        self.V = torch.cat([self.V, x_n.unsqueeze(0)])
        self.labels = torch.cat([self.labels, torch.tensor([label], device=self.device)])
        self._update_thresh()

    def add_diff(self, diff_vec):
        """Store a frame diff for variance computation."""
        self.diff_history.append(diff_vec)
        if len(self.diff_history) > 2000:
            self.diff_history = self.diff_history[-1000:]

    def update_diff_weights(self):
        if len(self.diff_history) < 20: return
        diffs = np.stack(self.diff_history)  # (N, 256)
        cell_var = diffs.var(axis=0)  # (256,)
        mx = cell_var.max()
        if mx > 0:
            self.weights = torch.from_numpy((cell_var / mx).astype(np.float32)).to(self.device)
        else:
            self.weights = torch.ones(D_SINGLE, device=self.device)

    def _update_thresh(self):
        n = self.V.shape[0]
        if n < 2: return
        ss = min(500, n)
        idx = torch.randperm(n, device=self.device)[:ss]
        # Apply weights to first D_SINGLE dims of each entry
        w = self._expand_weights()
        wV_s = self.V[idx] * w.unsqueeze(0)
        wV_a = self.V * w.unsqueeze(0)
        sims = F.normalize(wV_s, dim=1) @ F.normalize(wV_a, dim=1).T
        topk = sims.topk(min(2, n), dim=1).values
        self.thresh = float((topk[:, 1] if topk.shape[1] >= 2 else topk[:, 0]).median())

    def _expand_weights(self):
        """Expand weights to match V dimension (repeat for windowed)."""
        if self.d == D_SINGLE:
            return self.weights
        # For windowed: tile weights across window slots
        return self.weights.repeat(self.d // D_SINGLE)

    def process_novelty(self, x, n_cls, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        self.step_count += 1
        if self.step_count % 100 == 0:
            self.update_diff_weights()

        if self.V.shape[0] == 0:
            sl = label if label is not None else 0
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([sl], device=self.device)
            return sl

        w = self._expand_weights()
        wx = F.normalize(x * w, dim=0)
        wV = F.normalize(self.V * w.unsqueeze(0), dim=1)
        sims = wV @ wx

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
            wi = ts.argmax().item()
            raw_sim = float((self.V[wi] @ x).item())
            a = 1.0 - raw_sim
            self.V[wi] = F.normalize(self.V[wi] + a * (x - self.V[wi]), dim=0)
        return pred


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


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
    fold = DiffVarianceFold(d=d, k=3)
    env  = arc.make(game_id)
    obs  = env.reset()
    n_acts = len(env.action_space)

    total_steps = 0; go = 0; levels = 0; seeded = False
    unique_states = set(); unique_windows = set()
    action_counts = {}; history = []
    prev_pooled = None

    while total_steps < max_steps and go < 50:
        if obs is None or obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); history = []; prev_pooled = None
            if obs is None: break; continue
        if obs.state == GameState.WIN: break

        curr_pooled = avgpool16(obs.frame)
        unique_states.add(hash(curr_pooled.tobytes()))
        history.append(curr_pooled)

        # Compute and store diff
        if prev_pooled is not None:
            diff = curr_pooled - prev_pooled
            fold.add_diff(diff)

        if windowed:
            enc = make_window(history, fold)
            unique_windows.add(hash(enc.numpy().tobytes()))
        else:
            enc = centered_enc(curr_pooled, fold)

        if not seeded and fold.V.shape[0] < n_acts:
            i = fold.V.shape[0]; fold._force_add(enc, label=i)
            obs = env.step(env.action_space[i]); total_steps += 1
            prev_pooled = curr_pooled
            action_counts[env.action_space[i].name] = action_counts.get(env.action_space[i].name, 0) + 1
            if fold.V.shape[0] >= n_acts: seeded = True
            continue
        if not seeded: seeded = True

        cls = fold.process_novelty(enc, n_cls=n_acts)
        action = env.action_space[cls % n_acts]
        action_counts[action.name] = action_counts.get(action.name, 0) + 1

        ol = obs.levels_completed
        prev_pooled = curr_pooled
        obs = env.step(action); total_steps += 1
        if obs is None: break
        if obs.levels_completed > ol:
            levels = obs.levels_completed
            print(f"    [{label}] LEVEL {levels} at step {total_steps}", flush=True)

        if total_steps % 500 == 0:
            w = fold.weights.cpu().numpy().reshape(16, 16)
            timer_w = w[15, 5:13].mean()
            sprite_w = w[9:12, 9:13].mean()
            print(f"    [{label}] step {total_steps:5d} cb={fold.V.shape[0]}"
                  f"  unique_s={len(unique_states)}"
                  f"  unique_w={len(unique_windows) if windowed else 'N/A'}"
                  f"  timer_w={timer_w:.3f}  sprite_w={sprite_w:.3f}"
                  f"  acts={action_counts}", flush=True)

    w = fold.weights.cpu().numpy()
    w_grid = w.reshape(16, 16)
    return {
        'label': label, 'levels': levels, 'steps': total_steps, 'go': go,
        'cb': fold.V.shape[0], 'thresh': fold.thresh,
        'unique_states': len(unique_states),
        'unique_windows': len(unique_windows) if windowed else 0,
        'action_counts': action_counts,
        'w_grid': w_grid,
        'timer_weight': float(w_grid[15, 5:13].mean()),
        'sprite_weight': float(w_grid[9:12, 9:13].mean()),
    }


def main():
    t0 = time.time()
    print("Step 366 -- Diff-variance weighting", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(flush=True)

    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    # 366a: single state + diff-variance weights
    print("=== 366a: single state + diff-variance weights (2K steps) ===", flush=True)
    r_a = run_trial(arc, ls20.game_id, windowed=False, max_steps=2000, label="366a")

    print(flush=True)
    print(f"  366a: timer_weight={r_a['timer_weight']:.4f}  sprite_weight={r_a['sprite_weight']:.4f}", flush=True)
    pass_a = r_a['timer_weight'] < 0.1 and r_a['sprite_weight'] > 0.3
    print(f"  366a {'PASS' if pass_a else 'KILL'}: timer<0.1={r_a['timer_weight']<0.1}"
          f"  sprite>0.3={r_a['sprite_weight']>0.3}", flush=True)

    # Weight grid
    print("\n  Diff-variance weight grid (16x16):", flush=True)
    for row in r_a['w_grid']:
        print("    " + " ".join(f"{v:.2f}" for v in row), flush=True)

    # 366b: windowed + diff-variance weights (only if 366a passes)
    r_b = None
    if pass_a:
        print(flush=True)
        print("=== 366b: windowed (k=3) + diff-variance weights (2K steps) ===", flush=True)
        r_b = run_trial(arc, ls20.game_id, windowed=True, max_steps=2000, label="366b")
        ratio = r_b['unique_windows'] / max(r_b['unique_states'], 1)
        pass_b = ratio > 1.5
        print(f"\n  366b: unique_windows={r_b['unique_windows']}"
              f"  unique_states={r_b['unique_states']}  ratio={ratio:.2f}x", flush=True)
        print(f"  366b {'PASS' if pass_b else 'KILL'}: ratio>1.5={pass_b}", flush=True)
    else:
        print("\n  Skipping 366b (366a failed).", flush=True)

    elapsed = time.time() - t0
    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 366 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"366a: timer_w={r_a['timer_weight']:.4f} sprite_w={r_a['sprite_weight']:.4f}"
          f"  {'PASS' if pass_a else 'KILL'}", flush=True)
    print(f"  actions: {r_a['action_counts']}", flush=True)
    if r_b:
        ratio = r_b['unique_windows'] / max(r_b['unique_states'], 1)
        print(f"366b: windows/states={ratio:.2f}x"
              f"  {'PASS' if ratio > 1.5 else 'KILL'}", flush=True)
        print(f"  actions: {r_b['action_counts']}", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
