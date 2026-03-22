#!/usr/bin/env python3
"""
Step 418f — Hybrid: ReadIsWrite distributed update + argmin class scoring + error spawn.
LS20, 16x16 avgpool + centered_enc, tau=0.1, 10K steps.
"""

import sys, time, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 256


class RIWHybrid:
    def __init__(self, d, tau=0.1, k=3, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.d = d
        self.tau = tau
        self.k = k
        self.device = device
        self.recent_errors = []
        self.n_spawns = 0
        self.last_err = 0.0

    def step(self, x, n_actions):
        x = F.normalize(x.to(self.device).float(), dim=0)

        if self.V.shape[0] < n_actions:
            label = self.V.shape[0]
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([label], device=self.device)])
            self.n_spawns += 1
            return label

        # ReadIsWrite: read + reconstruct
        sims = self.V @ x
        weights = F.softmax(sims / self.tau, dim=0)
        output = weights @ self.V
        error = x - output

        # Distributed update (THE equation)
        self.V += torch.outer(weights, error)
        self.V = F.normalize(self.V, dim=1)

        # ACTION: argmin class scoring (vectorized)
        n_cls = int(self.labels.max().item()) + 1
        cls_ids = torch.arange(n_cls, device=self.device).unsqueeze(1)
        one_hot = (self.labels.unsqueeze(0) == cls_ids).float()
        masked = sims.unsqueeze(0) * one_hot + (1 - one_hot) * (-1e9)
        topk_k = min(self.k, masked.shape[1])
        scores = masked.topk(topk_k, dim=1).values.sum(dim=1)
        prediction = scores.argmin().item()

        # Spawn: error-based
        err_norm = error.norm().item()
        self.last_err = err_norm
        self.recent_errors.append(err_norm)
        if len(self.recent_errors) > 1000:
            self.recent_errors = self.recent_errors[-1000:]

        if len(self.recent_errors) < 10 or err_norm > sorted(self.recent_errors)[len(self.recent_errors) // 2]:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([prediction], device=self.device)])
            self.n_spawns += 1

        return prediction


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(pooled, sub):
    t = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if sub.V.shape[0] > 2:
        mean_V = sub.V.mean(dim=0).cpu()
        t_unit = t_unit - mean_V
    return t_unit


def main():
    from arcengine import GameState
    import arc_agi

    print(f"Step 418f: RIW Hybrid (distributed update + argmin + error spawn)")
    print(f"Device: {DEVICE}  tau=0.1  k=3  10K steps", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: ls20 not found"); return

    sub = RIWHybrid(d=D_ENC, tau=0.1, k=3)
    env = arc.make(ls20.game_id)
    obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0
    unique = set()
    action_counts = [0] * na
    t0 = time.time()
    spawn_at = []

    while ts < 10000:
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

        if ts % 2000 == 0:
            elapsed = time.time() - t0
            total = sum(action_counts)
            dom = max(action_counts) / total * 100 if total else 0
            spawn_rate = sub.n_spawns / ts * 100
            recent_spawn = sub.n_spawns - (spawn_at[-1] if spawn_at else 0)
            recent_rate = recent_spawn / 2000 * 100
            spawn_at.append(sub.n_spawns)

            print(f"  [step {ts:6d}]  cb={sub.V.shape[0]:5d}  err={sub.last_err:.4f}"
                  f"  spawn={spawn_rate:.0f}%(recent={recent_rate:.0f}%)"
                  f"  unique={len(unique):5d}  lvls={lvls}  go={go}  dom={dom:.0f}%"
                  f"  {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    total = sum(action_counts)
    dom = max(action_counts) / total * 100 if total else 0
    spawn_rate = sub.n_spawns / ts * 100

    print(f"\n{'='*60}")
    print("STEP 418f RESULTS")
    print(f"{'='*60}")
    print(f"unique={len(unique)}  levels={lvls}  go={go}  dom={dom:.0f}%")
    print(f"cb={sub.V.shape[0]}  spawn_rate={spawn_rate:.0f}%  elapsed={elapsed:.0f}s")
    print(f"acts={action_counts}")

    kills = []
    if len(unique) < 500: kills.append(f"unique={len(unique)}<500")
    if dom > 70: kills.append(f"dom={dom:.0f}%>70%")
    if spawn_rate > 80: kills.append(f"spawn={spawn_rate:.0f}%>80%")
    if kills:
        print(f"KILL: {', '.join(kills)}")
    else:
        print("PASS: all kill criteria clear")
        if len(unique) > 1000 and dom < 50:
            print("EXTEND: unique>1000 AND dom<50% -> run 30K for Level 1 test")


if __name__ == '__main__':
    main()
