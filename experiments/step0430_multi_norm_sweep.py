#!/usr/bin/env python3
"""
Step 430 — Fractional normalization sweep: score = top-K sum / count^p.
p=0.25 (weak), p=0.5 (sqrt), p=0.75 (strong). 1 seed each, 30K steps.
p=0 = baseline (gap converges). p=1 = Step 429 (bias inverts).
"""

import sys, time, logging
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 256


class ProcessNoveltyNormP:
    def __init__(self, d, k=3, p=0.5, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7; self.k = k; self.p = p
        self.d = d; self.device = device
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
        cls_ids = torch.arange(max(actual_cls, n_cls), device=self.device).unsqueeze(1)
        one_hot = (self.labels.unsqueeze(0) == cls_ids).float()
        masked = sims.unsqueeze(0) * one_hot + (1 - one_hot) * (-1e9)
        topk_k = min(self.k, masked.shape[1])
        raw_scores = masked.topk(topk_k, dim=1).values.sum(dim=1)

        # FRACTIONAL NORMALIZATION: score / count^p
        class_counts = one_hot.sum(dim=1).clamp(min=1)
        scores = raw_scores / (class_counts ** self.p)

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


def run_p(p_val, arc, ls20):
    from arcengine import GameState
    fold = ProcessNoveltyNormP(d=D_ENC, k=3, p=p_val)
    env = arc.make(ls20.game_id); obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0; unique = set(); action_counts = [0]*na
    seeded = False; t0 = time.time()
    checkpoints = {1000, 5000, 10000, 15000, 20000, 25000, 30000}
    recent_actions = []; score_log = []

    while ts < 30000:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); seeded = False
            if obs is None: break
            continue
        if obs.state == GameState.WIN:
            print(f"        WIN at step {ts}!", flush=True); break

        pooled = avgpool16(obs.frame); unique.add(hash(pooled.tobytes()))
        x = centered_enc(pooled, fold)

        if not seeded and fold.V.shape[0] < na:
            i = fold.V.shape[0]; fold._force_add(x, label=i)
            action = env.action_space[i % na]; data = {}
            if action.is_complex():
                arr = np.array(obs.frame[0]); cy, cx = divmod(int(np.argmax(arr)), 64)
                data = {"x": cx, "y": cy}
            obs = env.step(action, data=data); ts += 1; recent_actions.append(i % na)
            if fold.V.shape[0] >= na: seeded = True; fold._update_thresh()
            continue
        if not seeded: seeded = True

        cls_used = fold.process_novelty(x, n_cls=na, label=None)
        recent_actions.append(cls_used % na)
        if len(recent_actions) > 1000: recent_actions = recent_actions[-1000:]
        action = env.action_space[cls_used % na]; action_counts[cls_used % na] += 1
        data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0]); cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}
        obs_before = obs.levels_completed
        obs = env.step(action, data=data); ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            print(f"        LEVEL {lvls} at step {ts}  cb={fold.V.shape[0]}", flush=True)

        if ts in checkpoints and fold.last_scores is not None:
            scores = fold.last_scores
            gap = float(scores.max() - scores.min())
            last_1k = recent_actions[-1000:]
            persist = Counter(last_1k).most_common(1)[0][1] / len(last_1k) * 100 if last_1k else 0
            total = sum(action_counts); dom = max(action_counts) / total * 100 if total else 0
            score_log.append((ts, gap, dom, persist, fold.V.shape[0]))

    elapsed = time.time() - t0
    total = sum(action_counts); dom = max(action_counts) / total * 100 if total else 0
    return {'p': p_val, 'unique': len(unique), 'levels': lvls, 'cb': fold.V.shape[0],
            'dom': dom, 'elapsed': elapsed, 'score_log': score_log}


def main():
    import arc_agi
    print(f"Step 430: Fractional normalization sweep (p=0.25, 0.5, 0.75)")
    print(f"Device: {DEVICE}  1 seed, 30K each", flush=True)

    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return

    results = []
    for p in [0.25, 0.5, 0.75]:
        print(f"\n    p={p}:", flush=True)
        r = run_p(p, arc, ls20)
        results.append(r)
        nav = f"Level {r['levels']}" if r['levels'] > 0 else "none"
        print(f"      FINAL: {nav}  unique={r['unique']}  cb={r['cb']}  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s")
        if r['score_log']:
            print(f"      Gap:  {[f'{g:.4f}' for _, g, _, _, _ in r['score_log']]}")
            print(f"      Dom:  {[f'{d:.0f}%' for _, _, d, _, _ in r['score_log']]}")

    print(f"\n{'='*60}")
    print("STEP 430 SUMMARY")
    print(f"{'='*60}")
    print(f"{'p':<6} {'unique':>7} {'cb':>6} {'dom':>6} {'gap_1K':>8} {'gap_30K':>8} {'level'}")
    print(f"{'-'*55}")
    for r in results:
        g1 = r['score_log'][0][1] if r['score_log'] else 0
        gf = r['score_log'][-1][1] if r['score_log'] else 0
        nav = f"L{r['levels']}" if r['levels'] > 0 else "none"
        print(f"{r['p']:<6} {r['unique']:>7} {r['cb']:>6} {r['dom']:>5.0f}% {g1:>8.4f} {gf:>8.4f} {nav}")

    # Find sweet spot
    sweet = [r for r in results if r['score_log'] and r['score_log'][-1][1] > 0.01 and r['dom'] < 40]
    if sweet:
        best = min(sweet, key=lambda r: r['p'])
        print(f"\nSWEET SPOT: p={best['p']} (gap>{0.01}, dom<40%). Run 3 seeds.")
    elif all(r['dom'] > 90 for r in results):
        print("\nALL COLLAPSE. No sweet spot exists.")
    else:
        print("\nNo sweet spot found. Check individual trajectories.")


if __name__ == '__main__':
    main()
