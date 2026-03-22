#!/usr/bin/env python3
"""
Step 419 — Encoding self-discovery: centering detection via codebook health.
Phase 1 (0-500): no centering. Phase 2 (500-1000): centering.
Decision at 1000: if cb_phase2 > 2*cb_phase1 -> centering wins.
Phase 3 (1000-10000): use winner.
"""

import sys, time, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 256


# Use Step 353's CompressedFold process_novelty (exact baseline)
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
        # Vectorized scoring
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
        mean_V = fold.V.mean(dim=0).cpu()
        t_unit = t_unit - mean_V
    return t_unit


def main():
    from arcengine import GameState
    import arc_agi

    print(f"Step 419: Encoding self-discovery (centering detection)")
    print(f"Device: {DEVICE}", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: ls20 not found"); return

    fold = CompressedFold(d=D_ENC, k=3)
    env = arc.make(ls20.game_id)
    obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0
    unique = set()
    action_counts = [0] * na
    seeded = False
    t0 = time.time()

    # Phase tracking
    cb_at_500 = 0
    cb_at_1000_start = 0
    cb_at_1000 = 0
    use_centering = False  # decided at step 1000

    while ts < 10000:
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

        # Phase-dependent encoding
        if ts < 500:
            x = raw_enc(pooled)  # Phase 1: no centering
        elif ts < 1000:
            x = centered_enc(pooled, fold)  # Phase 2: centering
        else:
            if use_centering:
                x = centered_enc(pooled, fold)
            else:
                x = raw_enc(pooled)

        # Seed
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

        # Phase boundaries
        if ts == 500:
            cb_at_500 = fold.V.shape[0]
            cb_at_1000_start = cb_at_500  # reset growth tracking
            print(f"\n  PHASE 1 (no centering) complete: cb={cb_at_500}", flush=True)

        if ts == 1000:
            cb_at_1000 = fold.V.shape[0]
            cb_growth_phase1 = cb_at_500  # growth from 0 to 500 steps
            cb_growth_phase2 = cb_at_1000 - cb_at_500  # growth from 500 to 1000

            print(f"  PHASE 2 (centering) complete: cb={cb_at_1000}")
            print(f"  Phase 1 growth (no centering): +{cb_growth_phase1}")
            print(f"  Phase 2 growth (centering):    +{cb_growth_phase2}")

            if cb_growth_phase2 > 2 * cb_growth_phase1:
                use_centering = True
                print(f"  DECISION: centering WINS ({cb_growth_phase2} > 2*{cb_growth_phase1})")
            else:
                use_centering = False
                print(f"  DECISION: raw WINS ({cb_growth_phase2} <= 2*{cb_growth_phase1})")
            print(flush=True)

        if ts % 2000 == 0 and ts > 0:
            elapsed = time.time() - t0
            total = sum(action_counts)
            dom = max(action_counts) / total * 100 if total else 0
            enc_label = "centered" if (ts > 1000 and use_centering) or (500 <= ts <= 1000) else "raw"
            print(f"  [step {ts:6d}]  cb={fold.V.shape[0]:5d}  unique={len(unique):5d}"
                  f"  lvls={lvls}  go={go}  dom={dom:.0f}%  enc={enc_label}  {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    total = sum(action_counts)
    dom = max(action_counts) / total * 100 if total else 0

    print(f"\n{'='*60}")
    print("STEP 419 RESULTS")
    print(f"{'='*60}")
    print(f"Decision: {'CENTERING' if use_centering else 'RAW (no centering)'}")
    print(f"cb at phase boundaries: phase1={cb_at_500}  phase2={cb_at_1000}")
    print(f"unique={len(unique)}  levels={lvls}  go={go}  dom={dom:.0f}%")
    print(f"cb={fold.V.shape[0]}  elapsed={elapsed:.0f}s")

    if use_centering and len(unique) > 500:
        print("PASS: correctly selected centering AND unique>500")
    elif not use_centering:
        print("FAIL: selected raw (wrong for LS20)")
    else:
        print(f"PARTIAL: selected centering but unique={len(unique)}<500")


if __name__ == '__main__':
    main()
