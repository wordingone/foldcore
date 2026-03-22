#!/usr/bin/env python3
"""
Step 371 -- VC33 at 64x64 full resolution. 2K steps.

4096 dims. Click-space (64 classes). Pure argmin. Sampled thresh.
Kill: unique_states > 50 (was 50 at 16x16).
Script: scripts/run_step371_vc33_fullres.py
"""

import time
import logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 4096  # 64x64

CLICK_GRID = []
for gy in range(8):
    for gx in range(8):
        CLICK_GRID.append((gx * 8 + 4, gy * 8 + 4))
N_CLICK = 64


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


def full64(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.flatten()  # (4096,)


def centered_enc(pooled, fold):
    t = F.normalize(torch.from_numpy(pooled.astype(np.float32)), dim=0)
    if fold.V.shape[0] > 2:
        t = t - fold.V.mean(dim=0).cpu()
    return t


def main():
    t0 = time.time()
    print("Step 371 -- VC33 at 64x64 full resolution (4096 dims)", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(flush=True)

    import arc_agi
    from arcengine import GameState

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    vc33 = next(g for g in games if 'vc33' in g.game_id.lower())

    fold = CompressedFold(d=D_ENC, k=3)
    env = arc.make(vc33.game_id)
    obs = env.reset()
    action_space = env.action_space

    total_steps = 0; go = 0; levels = 0
    unique_states = set(); cls_counts = {}

    max_steps = 2000

    while total_steps < max_steps and go < 200:
        if obs is None or obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset()
            if obs is None: break; continue
        if obs.state == GameState.WIN:
            print(f"    WIN at step {total_steps}!", flush=True)
            break

        pooled = full64(obs.frame)
        unique_states.add(hash(pooled.tobytes()))
        enc = centered_enc(pooled, fold)

        cls = fold.process_novelty(enc, n_cls=N_CLICK)
        cls_counts[cls] = cls_counts.get(cls, 0) + 1
        cx, cy = CLICK_GRID[cls % N_CLICK]

        ol = obs.levels_completed
        obs = env.step(action_space[0], data={"x": cx, "y": cy})
        total_steps += 1
        if obs is None: break

        if obs.levels_completed > ol:
            levels = obs.levels_completed
            print(f"    LEVEL {levels} at step {total_steps} cb={fold.V.shape[0]}", flush=True)

        if total_steps % 500 == 0:
            print(f"    [step {total_steps:5d}] cb={fold.V.shape[0]}"
                  f"  unique={len(unique_states)}  go={go}"
                  f"  thresh={fold.thresh:.4f}", flush=True)

    elapsed = time.time() - t0

    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 371 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"levels={levels}  steps={total_steps}  go={go}", flush=True)
    print(f"cb_final={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)
    print(f"unique_states={len(unique_states)}", flush=True)
    print(f"classes_used={len(cls_counts)}", flush=True)
    print(flush=True)

    if len(unique_states) > 50:
        print(f"PASS: unique_states={len(unique_states)} > 50 (game visible at 64x64)", flush=True)
    else:
        print(f"KILL: unique_states={len(unique_states)} <= 50 (still invisible)", flush=True)

    print(flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
