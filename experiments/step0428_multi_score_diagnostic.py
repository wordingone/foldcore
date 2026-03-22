#!/usr/bin/env python3
"""
Step 428 — Action-score convergence diagnostic.
Baseline process_novelty on LS20. Log class scores, gap, action persistence.
Question: does the gap between action scores shrink → random walk?
"""

import sys, time, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 256


class ProcessNoveltyDiag:
    """process_novelty with score diagnostics."""
    def __init__(self, d, k=3, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7; self.k = k; self.d = d; self.device = device
        self.last_scores = None

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
        actual_cls = int(self.labels.max().item()) + 1
        # Vectorized top-K scoring
        cls_ids = torch.arange(max(actual_cls, n_cls), device=self.device).unsqueeze(1)
        one_hot = (self.labels.unsqueeze(0) == cls_ids).float()
        masked = sims.unsqueeze(0) * one_hot + (1 - one_hot) * (-1e9)
        topk_k = min(self.k, masked.shape[1])
        scores = masked.topk(topk_k, dim=1).values.sum(dim=1)

        self.last_scores = scores[:n_cls].cpu().numpy().copy()
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


def main():
    from arcengine import GameState
    import arc_agi

    print(f"Step 428: Action-score convergence diagnostic")
    print(f"Device: {DEVICE}  1 seed, 30K steps", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: ls20 not found"); return

    fold = ProcessNoveltyDiag(d=D_ENC, k=3)
    env = arc.make(ls20.game_id)
    obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0
    unique = set()
    seeded = False
    t0 = time.time()

    # Diagnostics
    checkpoints = {1000, 5000, 10000, 15000, 20000, 25000, 30000}
    recent_actions = []
    score_log = []

    while ts < 30000:
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
            recent_actions.append(i % na)
            if fold.V.shape[0] >= na:
                seeded = True
                fold._update_thresh()
            continue
        if not seeded: seeded = True

        cls_used = fold.process_novelty(x, n_cls=na, label=None)
        recent_actions.append(cls_used % na)
        if len(recent_actions) > 1000:
            recent_actions = recent_actions[-1000:]

        action = env.action_space[cls_used % na]
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

        if ts in checkpoints and fold.last_scores is not None:
            elapsed = time.time() - t0
            scores = fold.last_scores
            gap = float(scores.max() - scores.min())
            # Action persistence: % of last 1K steps with most common action
            from collections import Counter
            last_1k = recent_actions[-1000:]
            if last_1k:
                mc = Counter(last_1k).most_common(1)[0][1]
                persist = mc / len(last_1k) * 100
            else:
                persist = 0

            score_log.append((ts, scores.tolist(), gap, persist))
            print(f"\n  [step {ts:6d}]  cb={fold.V.shape[0]:5d}  unique={len(unique):5d}"
                  f"  lvls={lvls}  go={go}  {elapsed:.0f}s")
            print(f"    scores={[f'{s:.2f}' for s in scores]}")
            print(f"    gap={gap:.4f}  persist={persist:.0f}%", flush=True)

    print(f"\n{'='*60}")
    print("STEP 428 — ACTION-SCORE CONVERGENCE")
    print(f"{'='*60}")
    print(f"{'step':>6}  {'gap':>10}  {'persist':>8}  {'scores'}")
    print(f"{'-'*60}")
    for step, scores, gap, persist in score_log:
        print(f"{step:>6}  {gap:>10.4f}  {persist:>7.0f}%  {[f'{s:.1f}' for s in scores]}")

    gaps = [g for _, _, g, _ in score_log]
    if len(gaps) >= 3:
        if all(gaps[i] >= gaps[i+1] for i in range(len(gaps)-1)):
            print("\nGap MONOTONICALLY DECREASING -> action-score convergence IS the wall")
        elif gaps[-1] < gaps[0] * 0.5:
            print(f"\nGap HALVED ({gaps[0]:.4f} -> {gaps[-1]:.4f}) -> convergence is a factor")
        else:
            print(f"\nGap STABLE/FLUCTUATING ({gaps[0]:.4f} -> {gaps[-1]:.4f}) -> convergence NOT the bottleneck")


if __name__ == '__main__':
    main()
