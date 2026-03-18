#!/usr/bin/env python3
"""
Step 418e — ReadIsWrite with error-based spawn on LS20.
No Gram matrix. Spawn when reconstruction error > running median.
tau=0.1, 10K steps, 16x16 avgpool + centered_enc.
"""

import sys, time, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_ENC = 256


class ReadIsWriteErrSpawn:
    def __init__(self, d, tau=0.1, device=DEVICE):
        self.V = torch.zeros(0, d, device=device)
        self.d = d
        self.tau = tau
        self.device = device
        self.recent_errors = []
        self.n_spawns = 0
        self.last_err = 0.0

    def step(self, x, n_actions):
        x = F.normalize(x.to(self.device).float(), dim=0)

        if self.V.shape[0] < 2:
            self.V = torch.cat([self.V, x.unsqueeze(0)])
            self.n_spawns += 1
            return self.V.shape[0] - 1

        # THE EQUATION
        sims = self.V @ x
        weights = F.softmax(sims / self.tau, dim=0)
        output = weights @ self.V
        error = x - output
        self.V += torch.outer(weights, error)
        self.V = F.normalize(self.V, dim=1)

        # Error-based spawn
        err_norm = error.norm().item()
        self.last_err = err_norm
        self.recent_errors.append(err_norm)
        if len(self.recent_errors) > 1000:
            self.recent_errors = self.recent_errors[-1000:]

        if len(self.recent_errors) < 10:
            spawn = True
        else:
            err_median = sorted(self.recent_errors)[len(self.recent_errors) // 2]
            spawn = err_norm > err_median

        if spawn:
            self.V = torch.cat([self.V, F.normalize(x.unsqueeze(0), dim=1)])
            self.n_spawns += 1

        # Action from reconstruction
        action = output.argmax().item() % n_actions
        return action


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

    print(f"Step 418e: ReadIsWrite + error-based spawn on LS20")
    print(f"Device: {DEVICE}  tau=0.1  10K steps", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: ls20 not found"); return

    sub = ReadIsWriteErrSpawn(d=D_ENC, tau=0.1)
    env = arc.make(ls20.game_id)
    obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0
    unique = set()
    action_counts = [0] * na
    t0 = time.time()
    spawn_at_checkpoints = []

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

        spawns_before = sub.n_spawns
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
            # Spawn rate in last 1000 steps
            recent_spawn = sub.n_spawns - (spawn_at_checkpoints[-1] if spawn_at_checkpoints else 0)
            recent_rate = recent_spawn / 2000 * 100
            spawn_at_checkpoints.append(sub.n_spawns)

            print(f"  [step {ts:6d}]  cb={sub.V.shape[0]:5d}  err={sub.last_err:.4f}"
                  f"  spawn_rate={spawn_rate:.0f}%(recent={recent_rate:.0f}%)"
                  f"  unique={len(unique):5d}  lvls={lvls}  go={go}  dom={dom:.0f}%"
                  f"  {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    total = sum(action_counts)
    dom = max(action_counts) / total * 100 if total else 0
    spawn_rate = sub.n_spawns / ts * 100

    print(f"\n{'='*60}")
    print("STEP 418e RESULTS")
    print(f"{'='*60}")
    print(f"unique={len(unique)}  levels={lvls}  go={go}  dom={dom:.0f}%")
    print(f"cb={sub.V.shape[0]}  spawn_rate={spawn_rate:.0f}%  elapsed={elapsed:.0f}s")
    print(f"acts={action_counts}")

    # Kill criteria
    kills = []
    if spawn_rate > 90: kills.append(f"spawn_rate={spawn_rate:.0f}%>90%")
    if len(unique) < 200: kills.append(f"unique={len(unique)}<200")
    if dom > 80: kills.append(f"dom={dom:.0f}%>80%")
    if kills:
        print(f"KILL: {', '.join(kills)}")
    else:
        print("PASS: all kill criteria clear")


if __name__ == '__main__':
    main()
