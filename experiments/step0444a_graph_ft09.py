#!/usr/bin/env python3
"""
Step 444a — Graph Substrate on FT09.
FT09 baseline: Level 1 at ~82 steps (fast game).
Testing if graph navigation works on a different game.
16x16 avgpool + centered_enc. 10K steps, 3 seeds. 5-min cap.
"""

import time, logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class GraphSubstrate:
    def __init__(self, d=256, n_actions=4, sim_thresh=0.99):
        self.nodes = []
        self.edges = {}
        self.n_actions = n_actions
        self.sim_thresh = sim_thresh
        self.prev_node = None
        self.prev_action = None

    def _find_nearest(self, x):
        if len(self.nodes) == 0:
            return None, -1.0
        V = torch.stack(self.nodes)
        sims = F.cosine_similarity(V, x.unsqueeze(0))
        best_idx = sims.argmax().item()
        return best_idx, sims[best_idx].item()

    def step(self, x, label=None):
        x = F.normalize(x.float().flatten(), dim=0)
        node_idx, sim = self._find_nearest(x)
        if node_idx is None or sim < self.sim_thresh:
            node_idx = len(self.nodes)
            self.nodes.append(x.clone())
        else:
            lr = max(0, 1 - sim)
            self.nodes[node_idx] = self.nodes[node_idx] + lr * (x - self.nodes[node_idx])
            self.nodes[node_idx] = F.normalize(self.nodes[node_idx], dim=0)
        if self.prev_node is not None and self.prev_action is not None:
            key = (self.prev_node, self.prev_action)
            if key not in self.edges:
                self.edges[key] = {}
            self.edges[key][node_idx] = self.edges[key].get(node_idx, 0) + 1
        visit_counts = [sum(self.edges.get((node_idx, a), {}).values()) for a in range(self.n_actions)]
        min_count = min(visit_counts)
        candidates = [a for a, c in enumerate(visit_counts) if c == min_count]
        action = candidates[torch.randint(len(candidates), (1,)).item()]
        self.prev_node = node_idx
        self.prev_action = action
        return action


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(pooled, sub):
    t = torch.from_numpy(pooled.astype(np.float32))
    t_unit = F.normalize(t, dim=0)
    if len(sub.nodes) > 2:
        t_unit = t_unit - torch.stack(sub.nodes).mean(dim=0).cpu()
    return t_unit


def run_seed(arc, game_id, seed, max_steps=10000):
    from arcengine import GameState
    torch.manual_seed(seed)

    env = arc.make(game_id); obs = env.reset()
    na = len(env.action_space)
    sub = GraphSubstrate(d=256, n_actions=na, sim_thresh=0.99)

    ts = go = lvls = 0; unique = set(); action_counts = [0]*na
    level_step = None
    t0 = time.time()

    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue

        pooled = avgpool16(obs.frame)
        unique.add(hash(pooled.tobytes()))
        x = centered_enc(pooled, sub)

        idx = sub.step(x)
        action_counts[idx % na] += 1
        action = env.action_space[idx % na]; data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0]); cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}
        obs_before = obs.levels_completed
        obs = env.step(action, data=data); ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
            print(f"  LEVEL {lvls} at step {ts}  nodes={len(sub.nodes)}  edges={len(sub.edges)}", flush=True)
        if time.time() - t0 > 280: break

    elapsed = time.time() - t0
    dom = max(action_counts) / max(sum(action_counts), 1) * 100
    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique': len(unique), 'nodes': len(sub.nodes), 'na': na,
        'dom': dom, 'elapsed': elapsed,
    }


def main():
    import arc_agi
    print(f"Step 444a: Graph Substrate on FT09", flush=True)
    print(f"Device: {DEVICE}  sim_thresh=0.99  10K steps  3 seeds", flush=True)
    print(flush=True)

    arc = arc_agi.Arcade(); games = arc.get_environments()
    ft09 = next((g for g in games if 'ft09' in g.game_id.lower()), None)
    if not ft09: print("SKIP: FT09 not found"); return

    results = []
    for seed in [0, 1, 2]:
        print(f"--- Seed {seed} ---", flush=True)
        r = run_seed(arc, ft09.game_id, seed=seed, max_steps=10000)
        status = f"LEVEL {r['levels']} at step {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed}  {status}  unique={r['unique']}  nodes={r['nodes']}"
              f"  na={r['na']}  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s", flush=True)
        results.append(r)

    print(f"\n{'='*60}", flush=True)
    print("STEP 444a RESULTS (FT09)", flush=True)
    print(f"{'='*60}", flush=True)
    wins = [r for r in results if r['levels'] > 0]
    print(f"Reliability: {len(wins)}/3", flush=True)
    if wins:
        print(f"Step-to-level: {[r['level_step'] for r in wins]}", flush=True)
    print(f"Avg unique: {sum(r['unique'] for r in results)/len(results):.0f}", flush=True)
    print(f"FT09 codebook baseline: Level 1 at ~82 steps", flush=True)


if __name__ == '__main__':
    main()
