#!/usr/bin/env python3
"""
Step 356 -- ARC-AGI-3: local sim as exploration probability.

epsilon = nearest_sim (sims.max()):
  sim ~ 1.0 (well-explored): almost always random action (explore)
  sim ~ 0.5 (frontier): 50/50 random vs argmax
  sim ~ 0.1 (deep unknown): 90% argmax (exploit signal)

Kill criterion: correlation between nearest_sim and action entropy.
2000 steps on LS20. Per-100-step bins.
Script: scripts/run_step356_local_epsilon.py
"""

import time
import math
import random as rng
import logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC  = 256


class CompressedFold:
    def __init__(self, d, k=3, device=DEVICE):
        self.V      = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7
        self.k      = k
        self.d      = d
        self.device = device

    def _force_add(self, x, label):
        x_n = F.normalize(x.to(self.device).float(), dim=0)
        self.V      = torch.cat([self.V, x_n.unsqueeze(0)])
        self.labels = torch.cat([self.labels,
                                  torch.tensor([label], device=self.device)])
        self._update_thresh()

    def _update_thresh(self):
        n = self.V.shape[0]
        if n < 2: return
        sample_size = min(500, n)
        idx = torch.randperm(n, device=self.device)[:sample_size]
        sample = self.V[idx]
        sims = sample @ self.V.T
        topk_vals = sims.topk(min(2, n), dim=1).values
        if topk_vals.shape[1] >= 2:
            nearest = topk_vals[:, 1]
        else:
            nearest = topk_vals[:, 0]
        self.thresh = float(nearest.median())

    def process(self, x, label=None):
        x = F.normalize(x.to(self.device).float(), dim=0)
        if self.V.shape[0] == 0:
            spawn_label = label if label is not None else 0
            self.V      = x.unsqueeze(0)
            self.labels = torch.tensor([spawn_label], device=self.device)
            return spawn_label, 0.0
        sims = self.V @ x
        n_cls = int(self.labels.max().item()) + 1
        scores = torch.zeros(n_cls, device=self.device)
        for c in range(n_cls):
            mask = (self.labels == c)
            if mask.sum() == 0: continue
            cs = sims[mask]
            scores[c] = cs.topk(min(self.k, len(cs))).values.sum()

        # Local epsilon: nearest_sim as exploration probability
        nearest_sim = float(sims.max().item())

        if rng.random() < nearest_sim:
            # Near known territory -> random action (explore)
            prediction = rng.choice(range(n_cls))
        else:
            # Far from known -> follow best signal (exploit)
            prediction = int(scores.argmax().item())

        spawn_label = label if label is not None else prediction
        max_sim = nearest_sim
        if max_sim < self.thresh:
            self.V      = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([spawn_label], device=self.device)])
            self._update_thresh()
        else:
            winner = int(sims.argmax().item())
            alpha  = 1.0 - max_sim
            self.V[winner] = F.normalize(
                self.V[winner] + alpha * (x - self.V[winner]), dim=0)
        return prediction, nearest_sim


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(pooled, fold):
    t      = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if fold.V.shape[0] > 2:
        mean_V = fold.V.mean(dim=0).cpu()
        t_unit = t_unit - mean_V
    return t_unit


def action_entropy(counts, total):
    if total == 0: return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def run_game(arc, game_id, max_steps=2000, max_resets=50, k=3, verbose=True):
    from arcengine import GameState

    fold = CompressedFold(d=D_ENC, k=k)
    env  = arc.make(game_id)
    obs  = env.reset()

    total_steps     = 0
    total_resets    = 0
    total_levels    = 0
    game_over_count = 0
    action_counts   = {}
    unique_states   = set()
    win             = False
    seeded          = False

    bin_size       = 100
    bin_sims       = []
    bin_actions    = {}
    bins_data      = []

    while total_steps < max_steps and total_resets < max_resets:
        if obs is None:
            total_resets += 1
            if total_resets >= max_resets: break
            obs = env.reset()
            if obs is None: break
            continue

        if obs.state == GameState.GAME_OVER:
            game_over_count += 1
            total_resets += 1
            if total_resets >= max_resets: break
            obs = env.reset()
            if obs is None: break
            continue

        if obs.state == GameState.WIN:
            win = True
            break

        action_space = env.action_space
        n_acts       = len(action_space)

        curr_pooled = avgpool16(obs.frame)
        enc         = centered_enc(curr_pooled, fold)
        unique_states.add(hash(curr_pooled.tobytes()))

        if not seeded and fold.V.shape[0] < n_acts:
            i = fold.V.shape[0]
            fold._force_add(enc, label=i)
            action = action_space[i % n_acts]
            data = {}
            if action.is_complex():
                arr = np.array(obs.frame[0])
                cy, cx = divmod(int(np.argmax(arr)), 64)
                data = {"x": cx, "y": cy}
            obs = env.step(action, data=data)
            total_steps += 1
            action_counts[action.name] = action_counts.get(action.name, 0) + 1
            if fold.V.shape[0] >= n_acts:
                seeded = True
                if verbose:
                    print(f"    [seed done, step {total_steps}]"
                          f"  cb={fold.V.shape[0]}  thresh={fold.thresh:.4f}", flush=True)
            continue

        if not seeded:
            seeded = True

        cls_used, nearest_sim = fold.process(enc, label=None)
        action = action_space[cls_used % n_acts]
        action_counts[action.name] = action_counts.get(action.name, 0) + 1

        bin_sims.append(nearest_sim)
        bin_actions[action.name] = bin_actions.get(action.name, 0) + 1

        data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0])
            cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}

        obs_levels_before = obs.levels_completed
        obs = env.step(action, data=data)
        total_steps += 1
        if obs is None: break

        if len(bin_sims) >= bin_size:
            avg_sim = sum(bin_sims) / len(bin_sims)
            ent     = action_entropy(bin_actions, sum(bin_actions.values()))
            bins_data.append((avg_sim, ent, dict(bin_actions), fold.V.shape[0]))
            if verbose:
                print(f"    [step {total_steps:5d}] avg_sim={avg_sim:.4f}"
                      f"  entropy={ent:.3f}  cb={fold.V.shape[0]}"
                      f"  acts={dict(bin_actions)}", flush=True)
            bin_sims    = []
            bin_actions = {}

        if obs.levels_completed > obs_levels_before:
            total_levels = obs.levels_completed
            if verbose:
                print(f"    [step {total_steps}] LEVEL {obs.levels_completed}"
                      f"  cb={fold.V.shape[0]}", flush=True)

        if obs.state == GameState.WIN:
            win = True
            break

    cls_dist = {}
    if fold.V.shape[0] > 0:
        for lbl in fold.labels.cpu().numpy():
            k_ = int(lbl)
            cls_dist[k_] = cls_dist.get(k_, 0) + 1

    return {
        'win': win, 'levels': total_levels, 'steps': total_steps,
        'resets': total_resets, 'game_over': game_over_count,
        'cb_final': fold.V.shape[0], 'thresh_final': fold.thresh,
        'cls_dist': dict(sorted(cls_dist.items())),
        'action_counts': action_counts, 'unique_states': len(unique_states),
        'bins_data': bins_data,
    }


def main():
    t0 = time.time()
    print("Step 356 -- local sim as exploration probability (2000 steps)", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("epsilon = nearest_sim: high sim -> random, low sim -> argmax", flush=True)
    print(flush=True)

    import arc_agi
    arc   = arc_agi.Arcade()
    games = arc.get_environments()
    ls20  = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if ls20 is None:
        print("ERROR: LS20 not found", flush=True)
        return

    print(f"Running: {ls20.title} ({ls20.game_id})", flush=True)
    print(f"max_steps=2000  k=3  bin_size=100", flush=True)
    print(flush=True)

    r = run_game(arc, ls20.game_id, max_steps=2000, max_resets=50, k=3, verbose=True)
    elapsed = time.time() - t0

    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 356 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"win={r['win']}  levels={r['levels']}  steps={r['steps']}", flush=True)
    print(f"resets={r['resets']}  game_over={r['game_over']}", flush=True)
    print(f"cb_final={r['cb_final']}  thresh={r['thresh_final']:.4f}", flush=True)
    print(f"unique_states={r['unique_states']}", flush=True)
    print(flush=True)
    print(f"action_counts: {r['action_counts']}", flush=True)
    print(f"cls_dist: {r['cls_dist']}", flush=True)
    print(flush=True)

    print("Per-100-step bins (avg_nearest_sim, entropy, cb):", flush=True)
    for i, (sim, ent, acts, cb) in enumerate(r['bins_data']):
        print(f"  bin {i+1:3d} (step {(i+1)*100:5d}): sim={sim:.4f}"
              f"  entropy={ent:.3f}  cb={cb}", flush=True)

    if len(r['bins_data']) >= 3:
        sims_list = [b[0] for b in r['bins_data']]
        ents_list = [b[1] for b in r['bins_data']]
        sim_range = max(sims_list) - min(sims_list)
        ent_range = max(ents_list) - min(ents_list)
        n = len(sims_list)
        mean_s = sum(sims_list) / n
        mean_e = sum(ents_list) / n
        cov = sum((s - mean_s) * (e - mean_e) for s, e in zip(sims_list, ents_list)) / n
        std_s = (sum((s - mean_s)**2 for s in sims_list) / n) ** 0.5
        std_e = (sum((e - mean_e)**2 for e in ents_list) / n) ** 0.5
        corr = cov / (std_s * std_e) if std_s > 0 and std_e > 0 else 0
        print(flush=True)
        print(f"Sim range: {min(sims_list):.4f} -> {max(sims_list):.4f} (delta={sim_range:.4f})", flush=True)
        print(f"Entropy range: {min(ents_list):.3f} -> {max(ents_list):.3f} (delta={ent_range:.3f})", flush=True)
        print(f"Pearson correlation (sim vs entropy): {corr:.4f}", flush=True)
        if abs(corr) > 0.3 and sim_range > 0.01:
            print("PASS: nearest_sim modulates action entropy.", flush=True)
        else:
            print("KILL: nearest_sim does NOT modulate action entropy.", flush=True)

    print(flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
