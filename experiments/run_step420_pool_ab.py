#!/usr/bin/env python3
"""
Step 420 — Mean vs max pooling at 16x16 on LS20. No centering.
A: mean pool (baseline). B: max pool. 10K steps each.
"""

import sys, time, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 256


class CompressedFold:
    def __init__(self, d, k=3, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7; self.k = k; self.d = d; self.device = device

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
        cls_ids = torch.arange(n_cls, device=self.device).unsqueeze(1)
        one_hot = (self.labels.unsqueeze(0) == cls_ids).float()
        masked = sims.unsqueeze(0) * one_hot + (1 - one_hot) * (-1e9)
        topk_k = min(self.k, masked.shape[1])
        scores = masked.topk(topk_k, dim=1).values.sum(dim=1)
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


def meanpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def maxpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(t, 4).squeeze().flatten()
    return pooled.numpy()


def raw_enc(pooled):
    t = torch.from_numpy(pooled.astype(np.float32))
    return F.normalize(t, dim=0)


def run_variant(name, pool_fn, arc, ls20):
    from arcengine import GameState

    print(f"\n{'='*60}")
    print(f"Run {name}", flush=True)
    print(f"{'='*60}")

    fold = CompressedFold(d=D_ENC, k=3)
    env = arc.make(ls20.game_id)
    obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0
    unique = set()
    action_counts = [0] * na
    seeded = False
    t0 = time.time()

    while ts < 10000:
        if obs is None:
            obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); seeded = False
            if obs is None: break
            continue
        if obs.state == GameState.WIN:
            print(f"WIN at step {ts}!", flush=True); break

        pooled = pool_fn(obs.frame)
        unique.add(hash(pooled.tobytes()))
        x = raw_enc(pooled)

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
            if fold.V.shape[0] >= na: seeded = True
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

        if ts % 5000 == 0:
            elapsed = time.time() - t0
            total = sum(action_counts)
            dom = max(action_counts) / total * 100 if total else 0
            print(f"  [step {ts:6d}]  cb={fold.V.shape[0]:5d}  unique={len(unique):5d}"
                  f"  lvls={lvls}  go={go}  dom={dom:.0f}%  {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    total = sum(action_counts)
    dom = max(action_counts) / total * 100 if total else 0
    print(f"\n--- FINAL {name} ---")
    print(f"  unique={len(unique)}  levels={lvls}  go={go}  dom={dom:.0f}%  cb={fold.V.shape[0]}  {elapsed:.0f}s")
    return {'name': name, 'unique': len(unique), 'levels': lvls, 'cb': fold.V.shape[0], 'dom': dom}


def main():
    import arc_agi

    print(f"Step 420: Mean vs Max pooling at 16x16 on LS20")
    print(f"Device: {DEVICE}  No centering.", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: ls20 not found"); return

    ra = run_variant("A: Mean pool", meanpool16, arc, ls20)
    rb = run_variant("B: Max pool", maxpool16, arc, ls20)

    print(f"\n{'='*60}")
    print("STEP 420 SUMMARY")
    print(f"{'='*60}")
    print(f"{'Pool':<15} {'unique':>7} {'cb':>6} {'dom':>6}")
    print(f"{'-'*40}")
    print(f"{'Mean':<15} {ra['unique']:>7} {ra['cb']:>6} {ra['dom']:>5.0f}%")
    print(f"{'Max':<15} {rb['unique']:>7} {rb['cb']:>6} {rb['dom']:>5.0f}%")

    diff = abs(ra['unique'] - rb['unique']) / max(ra['unique'], rb['unique']) * 100
    if diff < 10:
        print(f"\nDiff={diff:.0f}% (<10%) -> GAUGE: pooling type doesn't matter")
    elif diff > 30:
        better = 'Mean' if ra['unique'] > rb['unique'] else 'Max'
        print(f"\nDiff={diff:.0f}% (>30%) -> FORCED: {better} pool is required")
    else:
        print(f"\nDiff={diff:.0f}% (10-30%) -> MARGINAL: slight preference but not forced")


if __name__ == '__main__':
    main()
