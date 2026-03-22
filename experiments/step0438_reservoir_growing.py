#!/usr/bin/env python3
"""
Step 438 — Growing reservoir. Start d=16, grow when trajectory rank < d/2.
Velocity readout. Spectral control. LS20 only, 30K steps.
"""

import time, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cpu'
SR_TARGET = 0.95


class GrowingReservoir:
    def __init__(self, d_init=16, input_dim=256, n_actions=4):
        self.d = d_init
        self.W = torch.randn(d_init, d_init) * (1.0 / d_init**0.5)
        self.U = torch.randn(d_init, input_dim) * (0.1 / input_dim**0.5)
        self.h = torch.zeros(d_init)
        self.n_actions = n_actions
        self.hebb_lr = 0.001
        self.h_history = []
        self.grow_every = 500
        self.step_count = 0
        self.n_grows = 0
        self.last_rank = 0
        self._normalize_sr()

    def _normalize_sr(self):
        sv = torch.linalg.svdvals(self.W)
        if sv[0].item() > SR_TARGET:
            self.W *= SR_TARGET / sv[0].item()

    def step(self, x, label=None):
        x = x.float().flatten()
        h_new = torch.tanh(self.W @ self.h + self.U @ x)
        delta_h = h_new - self.h
        # Velocity readout (only first n_actions dims, but d may be < n_actions initially)
        readout = delta_h[:self.n_actions] if self.d >= self.n_actions else delta_h
        action = readout.abs().argmax().item() % self.n_actions

        self.W += self.hebb_lr * torch.outer(delta_h, self.h)
        self._normalize_sr()

        # Update h BEFORE growth check
        self.h = h_new.detach()

        self.h_history.append(self.h.clone())
        self.step_count += 1

        if self.step_count % self.grow_every == 0 and len(self.h_history) >= 100:
            H = torch.stack(self.h_history[-100:])
            self.last_rank = torch.linalg.matrix_rank(H, tol=0.1).item()
            if self.last_rank < self.d * 0.5:
                self._grow()
            self.h_history = self.h_history[-100:]

        return action

    def _grow(self):
        new_d = self.d + 8
        new_W = torch.zeros(new_d, new_d)
        new_W[:self.d, :self.d] = self.W
        new_W[self.d:, :self.d] = torch.randn(8, self.d) * 0.01
        new_W[:self.d, self.d:] = torch.randn(self.d, 8) * 0.01
        self.W = new_W
        new_U = torch.zeros(new_d, self.U.shape[1])
        new_U[:self.d] = self.U
        new_U[self.d:] = torch.randn(8, self.U.shape[1]) * 0.01
        self.U = new_U
        new_h = torch.zeros(new_d)
        new_h[:self.d] = self.h
        self.h = new_h
        self.d = new_d
        self.n_grows += 1
        self._normalize_sr()


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def main():
    from arcengine import GameState
    import arc_agi

    print(f"Step 438: Growing reservoir on LS20. d_init=16, grow +8 when rank<d/2.", flush=True)

    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return

    res = GrowingReservoir(d_init=16, input_dim=256, n_actions=4)
    env = arc.make(ls20.game_id); obs = env.reset()
    na = len(env.action_space)
    ts = go = lvls = 0; unique = set(); action_counts = [0]*na; t0 = time.time()

    while ts < 30000:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: print(f"WIN at step {ts}!", flush=True); break
        pooled = avgpool16(obs.frame); unique.add(hash(pooled.tobytes()))
        x = F.normalize(torch.from_numpy(pooled.astype(np.float32)), dim=0)
        idx = res.step(x); action_counts[idx % na] += 1
        action = env.action_space[idx % na]; data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0]); cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}
        obs_before = obs.levels_completed
        obs = env.step(action, data=data); ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed; print(f"LEVEL {lvls} at step {ts}  d={res.d}", flush=True)
        if ts % 5000 == 0:
            total = sum(action_counts); dom = max(action_counts)/total*100
            print(f"  [step {ts}]  d={res.d}  rank={res.last_rank}  grows={res.n_grows}"
                  f"  unique={len(unique)}  lvls={lvls}  go={go}  dom={dom:.0f}%  {time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    total = sum(action_counts); dom = max(action_counts)/total*100
    print(f"\n{'='*60}")
    print("STEP 438 RESULTS")
    print(f"{'='*60}")
    print(f"d: 16 -> {res.d} ({res.n_grows} grows)")
    print(f"unique={len(unique)}  levels={lvls}  go={go}  dom={dom:.0f}%")
    print(f"acts={action_counts}  {elapsed:.0f}s")
    if res.d > 512 and lvls == 0:
        print("KILL: d>512 with 0 levels. Growth isn't helping.")
    elif lvls > 0:
        print(f"BREAKTHROUGH: Level {lvls} with growing reservoir!")

if __name__ == '__main__':
    main()
