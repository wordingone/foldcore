#!/usr/bin/env python3
"""
Step 418 — Read IS Write substrate on LS20.
Tau sweep: 0.01, 0.1, 1.0. 10K steps each for bracketing, 30K for winner.

Usage:
    python run_step418_read_is_write.py sweep        # 10K x 3 taus
    python run_step418_read_is_write.py run TAU       # 30K at specific tau
"""

import sys, time, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 256


class ReadIsWrite:
    def __init__(self, d, tau=0.1, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.d = d
        self.tau = tau
        self.device = device
        self.G = torch.zeros(0, 0, device=device)
        self.last_thresh = 0.0
        self.last_err = 0.0
        self.last_recon = 0.0

    def step(self, x, n_actions):
        x = F.normalize(x.to(self.device).float(), dim=0)

        if self.V.shape[0] < 2:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            n = self.V.shape[0]
            self.G = self.V @ self.V.T
            self.G.fill_diagonal_(-float('inf'))
            return self.V.shape[0] - 1

        # THE EQUATION
        sims = self.V @ x
        weights = F.softmax(sims / self.tau, dim=0)
        output = weights @ self.V
        error = x - output
        self.V += torch.outer(weights, error)
        self.V = F.normalize(self.V, dim=1)

        # Diagnostics
        self.last_err = error.norm().item()
        recon_quality = (weights * sims).sum().item()
        self.last_recon = recon_quality

        # Spawn threshold from Gram
        if self.V.shape[0] < 2:
            thresh = 0.5
        else:
            G_max = self.G.max(dim=1).values
            thresh = float(G_max.median())
        self.last_thresh = thresh

        if recon_quality < thresh:
            self.V = torch.cat([self.V, F.normalize(x.unsqueeze(0), dim=1)])

        # Full Gram recompute (all entries shifted)
        self.G = self.V @ self.V.T
        self.G.fill_diagonal_(-float('inf'))

        # Action from reconstruction
        action = output.argmax().item() % n_actions
        return action


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(pooled, substrate):
    t = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if substrate.V.shape[0] > 2:
        mean_V = substrate.V.mean(dim=0).cpu()
        t_unit = t_unit - mean_V
    return t_unit


def run_ls20(tau, max_steps, label=""):
    from arcengine import GameState
    import arc_agi

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: ls20 not found"); return None

    print(f"\n{'='*60}")
    print(f"Step 418: ReadIsWrite  tau={tau}  max_steps={max_steps}  {label}")
    print(f"{'='*60}", flush=True)

    sub = ReadIsWrite(d=D_ENC, tau=tau)
    env = arc.make(ls20.game_id)
    obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0
    unique = set()
    action_counts = [0] * na
    t0 = time.time()

    while ts < max_steps:
        if obs is None:
            obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset()
            if obs is None: break
            continue
        if obs.state == GameState.WIN:
            print(f"WIN at step {ts}!", flush=True); break

        pooled = avgpool16(obs.frame)
        unique.add(hash(pooled.tobytes()))
        x = centered_enc(pooled, sub)

        idx = sub.step(x, n_actions=na)
        action = env.action_space[idx % na]
        action_counts[idx % na] += 1

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
            print(f"LEVEL {lvls} at step {ts}  cb={sub.V.shape[0]}", flush=True)

        if ts % 5000 == 0:
            elapsed = time.time() - t0
            total = sum(action_counts)
            dom = max(action_counts) / total * 100 if total else 0
            print(f"  [step {ts:6d}]  cb={sub.V.shape[0]:5d}  thresh={sub.last_thresh:.4f}"
                  f"  err={sub.last_err:.4f}  recon={sub.last_recon:.4f}"
                  f"  unique={len(unique):5d}  lvls={lvls}  go={go}  dom={dom:.0f}%"
                  f"  {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    total = sum(action_counts)
    dom = max(action_counts) / total * 100 if total else 0
    print(f"\n--- FINAL tau={tau} ---")
    print(f"  unique={len(unique)}  levels={lvls}  go={go}  dom={dom:.0f}%")
    print(f"  cb={sub.V.shape[0]}  elapsed={elapsed:.0f}s")
    print(f"  acts={action_counts}")
    return {'tau': tau, 'unique': len(unique), 'levels': lvls, 'cb': sub.V.shape[0],
            'dom': dom, 'elapsed': elapsed, 'go': go}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sweep'

    if mode == 'sweep':
        print("Step 418 tau sweep: 0.01, 0.1, 1.0 at 10K steps each")
        results = []
        for tau in [0.01, 0.1, 1.0]:
            r = run_ls20(tau, max_steps=10000, label="(sweep)")
            if r: results.append(r)

        print(f"\n{'='*60}")
        print("TAU SWEEP SUMMARY")
        print(f"{'='*60}")
        print(f"{'tau':<8} {'unique':>7} {'levels':>7} {'cb':>6} {'dom':>6} {'elapsed':>8}")
        for r in results:
            print(f"{r['tau']:<8} {r['unique']:>7} {r['levels']:>7} {r['cb']:>6} {r['dom']:>5.0f}% {r['elapsed']:>7.0f}s")

    elif mode == 'run':
        tau = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
        print(f"Step 418 full run: tau={tau}, 30K steps")
        run_ls20(tau, max_steps=30000, label="(full)")


if __name__ == '__main__':
    main()
