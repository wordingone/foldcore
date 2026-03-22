#!/usr/bin/env python3
"""
Step 426 — Softmax voting on LS20 navigation.
process_novelty with softmax scoring (tau=0.01) + argmin action (exploration).
3 seeds, 30K steps each. Baseline: Level 1 at ~26K.
"""

import sys, time, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 256
TAU = 0.01


class ProcessNoveltySoftmax:
    """process_novelty with softmax scoring instead of top-K. Action = argmin."""
    def __init__(self, d, k=3, tau=TAU, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7; self.k = k; self.tau = tau
        self.d = d; self.device = device

    def _force_add(self, x, label):
        x_n = F.normalize(x.to(self.device).float(), dim=0)
        self.V = torch.cat([self.V, x_n.unsqueeze(0)])
        self.labels = torch.cat([self.labels, torch.tensor([label], device=self.device)])

    def _update_thresh(self):
        n = self.V.shape[0]
        if n < 2: return
        ss = min(500, n)
        idx = torch.randperm(n, device=self.device)[:ss]
        s = self.V[idx] @ self.V.T
        t = s.topk(min(2, n), dim=1).values
        self.thresh = float((t[:, 1] if t.shape[1] >= 2 else t[:, 0]).median())

    def process_novelty(self, x, n_cls, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            sl = label if label is not None else 0
            self.V = x.unsqueeze(0)
            self.labels = torch.tensor([sl], device=self.device)
            return sl

        sims = self.V @ x

        # SOFTMAX SCORING (change from top-K)
        weights = F.softmax(sims / self.tau, dim=0)
        actual_cls = int(self.labels.max().item()) + 1
        cls_ids = torch.arange(max(actual_cls, n_cls), device=self.device).unsqueeze(1)
        one_hot = (self.labels.unsqueeze(0) == cls_ids).float()
        scores = one_hot @ weights

        # ACTION = ARGMIN (exploration, NOT classification)
        prediction = scores[:n_cls].argmin().item()

        sl = label if label is not None else prediction
        target_mask = (self.labels == prediction)
        if target_mask.sum() == 0 or sims[target_mask].max() < self.thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([sl], device=self.device)])
            if self.V.shape[0] % 100 == 0:
                self._update_thresh()
        else:
            ts = sims.clone(); ts[~target_mask] = -float('inf')
            winner = ts.argmax().item()
            alpha = 1.0 - float(sims[winner].item())
            self.V[winner] = F.normalize(self.V[winner] + alpha * (x - self.V[winner]), dim=0)

        return prediction


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()

def centered_enc(pooled, fold):
    t = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if fold.V.shape[0] > 2:
        t_unit = t_unit - fold.V.mean(dim=0).cpu()
    return t_unit


def run_seed(seed, arc, ls20, max_steps=30000):
    from arcengine import GameState

    fold = ProcessNoveltySoftmax(d=D_ENC, k=3, tau=TAU)
    env = arc.make(ls20.game_id)
    obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0
    unique = set()
    action_counts = [0] * na
    seeded = False
    t0 = time.time()
    np.random.seed(seed)

    while ts < max_steps:
        if obs is None:
            obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); seeded = False
            if obs is None: break
            continue
        if obs.state == GameState.WIN:
            print(f"    WIN at step {ts}!", flush=True); break

        pooled = avgpool16(obs.frame)
        unique.add(hash(pooled.tobytes()))
        x = centered_enc(pooled, fold)

        if not seeded and fold.V.shape[0] < na:
            i = fold.V.shape[0]
            fold._force_add(x, label=i)
            action = env.action_space[i % na]
            data = {}
            if action.is_complex():
                arr = np.array(obs.frame[0])
                cy, cx = divmod(int(np.argmax(arr)), 64)
                data = {"x": cx, "y": cy}
            obs = env.step(action, data=data); ts += 1
            action_counts[i % na] += 1
            if fold.V.shape[0] >= na:
                seeded = True
                fold._update_thresh()
            continue
        if not seeded: seeded = True

        cls_used = fold.process_novelty(x, n_cls=na, label=None)
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
            print(f"    LEVEL {lvls} at step {ts}  cb={fold.V.shape[0]}", flush=True)

        if ts % 10000 == 0:
            elapsed = time.time() - t0
            total = sum(action_counts)
            dom = max(action_counts) / total * 100 if total else 0
            print(f"    [step {ts:6d}]  cb={fold.V.shape[0]:5d}  unique={len(unique):5d}"
                  f"  lvls={lvls}  go={go}  dom={dom:.0f}%  {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    total = sum(action_counts)
    dom = max(action_counts) / total * 100 if total else 0
    return {'seed': seed, 'unique': len(unique), 'levels': lvls,
            'cb': fold.V.shape[0], 'dom': dom, 'steps': ts, 'elapsed': elapsed}


def main():
    import arc_agi

    print(f"Step 426: Softmax voting (tau=0.01) + argmin on LS20")
    print(f"Device: {DEVICE}  3 seeds, 30K steps each", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: ls20 not found"); return

    results = []
    for seed in [42, 100, 200]:
        print(f"\n  Seed {seed}:", flush=True)
        r = run_seed(seed, arc, ls20, max_steps=30000)
        results.append(r)
        nav = f"Level {r['levels']} at {r['steps']}" if r['levels'] > 0 else f"0 levels at {r['steps']}"
        print(f"    FINAL: {nav}  unique={r['unique']}  cb={r['cb']}  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s")

    print(f"\n{'='*60}")
    print("STEP 426 SUMMARY")
    print(f"{'='*60}")
    for r in results:
        nav = f"L{r['levels']}@{r['steps']}" if r['levels'] > 0 else "none"
        print(f"  Seed {r['seed']}: {nav}  unique={r['unique']}  cb={r['cb']}  dom={r['dom']:.0f}%")

    navigated = [r for r in results if r['levels'] > 0]
    if not navigated:
        print("\nKILL: no seed navigated Level 1 by 30K")
    elif all(r['steps'] > 24000 for r in navigated):
        print(f"\nMARGINAL: navigated but still ~26K ({[r['steps'] for r in navigated]})")
    else:
        print(f"\nRESULT: Level 1 faster than baseline! Steps: {[r['steps'] for r in navigated]}")


if __name__ == '__main__':
    main()
