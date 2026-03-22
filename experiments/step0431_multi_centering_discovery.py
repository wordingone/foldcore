#!/usr/bin/env python3
"""
Step 431 — Can the substrate discover centering via codebook health?
Phase A (0-5K): no centering. Phase B (5K-10K): centering ON.
Compare health metrics: thresh, cb_size, dom%.
"""

import sys, time, logging
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 256


class ProcessNoveltyHealth:
    def __init__(self, d, k=3, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7; self.k = k; self.d = d; self.device = device

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
        scores = masked.topk(topk_k, dim=1).values.sum(dim=1)
        prediction = scores[:n_cls].argmin().item()
        sl = label if label is not None else prediction
        target_mask = (self.labels == prediction)
        if target_mask.sum() == 0 or sims[target_mask].max() < self.thresh:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels, torch.tensor([sl], device=self.device)])
            if self.V.shape[0] % 100 == 0: self._update_thresh()
        else:
            ts = sims.clone(); ts[~target_mask] = -float('inf')
            winner = ts.argmax().item()
            alpha = 1.0 - float(sims[winner].item())
            self.V[winner] = F.normalize(self.V[winner] + alpha * (x - self.V[winner]), dim=0)
        return prediction


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()

def raw_enc(pooled):
    t = torch.from_numpy(pooled.astype(np.float32))
    return F.normalize(t, dim=0)

def centered_enc(pooled, fold):
    t = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if fold.V.shape[0] > 2:
        t_unit = t_unit - fold.V.mean(dim=0).cpu()
    return t_unit


def main():
    from arcengine import GameState
    import arc_agi

    print(f"Step 431: Centering discovery via codebook health")
    print(f"Device: {DEVICE}  Phase A (0-5K): no centering. Phase B (5K-10K): centering.", flush=True)

    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return

    fold = ProcessNoveltyHealth(d=D_ENC, k=3)
    env = arc.make(ls20.game_id); obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0; unique = set(); action_counts = [0]*na
    seeded = False; t0 = time.time()
    recent_actions = []

    # Health logs per phase
    checkpoints = [500, 1000, 2000, 3000, 4000, 5000, 5500, 6000, 7000, 8000, 9000, 10000]
    health_log = []
    phase_a_final = {}
    phase_b_final = {}

    while ts < 10000:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); seeded = False
            if obs is None: break
            continue
        if obs.state == GameState.WIN:
            print(f"WIN at step {ts}!", flush=True); break

        pooled = avgpool16(obs.frame); unique.add(hash(pooled.tobytes()))

        # Phase-dependent encoding
        use_centering = ts >= 5000
        x = centered_enc(pooled, fold) if use_centering else raw_enc(pooled)

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
            print(f"LEVEL {lvls} at step {ts}", flush=True)

        if ts in checkpoints:
            elapsed = time.time() - t0
            total = sum(action_counts); dom = max(action_counts) / total * 100 if total else 0
            last_1k = recent_actions[-1000:]
            persist = Counter(last_1k).most_common(1)[0][1] / len(last_1k) * 100 if last_1k else 0
            phase = "A(raw)" if ts <= 5000 else "B(centered)"
            cb = fold.V.shape[0]
            # Spawn rate in last segment
            entry = {'step': ts, 'phase': phase, 'cb': cb, 'thresh': fold.thresh,
                     'unique': len(unique), 'dom': dom, 'persist': persist, 'go': go}
            health_log.append(entry)

            if ts == 5000: phase_a_final = dict(entry)
            if ts == 10000: phase_b_final = dict(entry)

            print(f"  [{phase} step {ts:5d}]  cb={cb:5d}  thresh={fold.thresh:.4f}"
                  f"  unique={len(unique):5d}  dom={dom:.0f}%  persist={persist:.0f}%  {elapsed:.0f}s", flush=True)

    print(f"\n{'='*60}")
    print("STEP 431 — CENTERING DISCOVERY")
    print(f"{'='*60}")
    print(f"{'phase':<12} {'step':>5} {'cb':>6} {'thresh':>8} {'unique':>7} {'dom':>5} {'persist':>8}")
    print(f"{'-'*55}")
    for h in health_log:
        print(f"{h['phase']:<12} {h['step']:>5} {h['cb']:>6} {h['thresh']:>8.4f} {h['unique']:>7} {h['dom']:>4.0f}% {h['persist']:>7.0f}%")

    if phase_a_final and phase_b_final:
        print(f"\n--- Phase comparison ---")
        print(f"Phase A (raw):      cb={phase_a_final['cb']}  thresh={phase_a_final['thresh']:.4f}  unique={phase_a_final['unique']}  dom={phase_a_final['dom']:.0f}%")
        print(f"Phase B (centered): cb={phase_b_final['cb']}  thresh={phase_b_final['thresh']:.4f}  unique={phase_b_final['unique']}  dom={phase_b_final['dom']:.0f}%")

        cb_growth_a = phase_a_final['cb']
        cb_growth_b = phase_b_final['cb'] - phase_a_final['cb']
        print(f"\ncb growth: Phase A = +{cb_growth_a},  Phase B = +{cb_growth_b}")
        if cb_growth_b > cb_growth_a * 1.5:
            print("SIGNAL: centering increases cb growth by >50%. Discoverable.")
        elif abs(phase_b_final['thresh'] - phase_a_final['thresh']) > 0.01:
            print(f"SIGNAL: thresh changed ({phase_a_final['thresh']:.4f} -> {phase_b_final['thresh']:.4f}). Discoverable.")
        else:
            print("NO SIGNAL: health metrics indistinguishable. Not discoverable by this protocol.")


if __name__ == '__main__':
    main()
