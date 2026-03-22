#!/usr/bin/env python3
"""
Step 443 — Graph Substrate reliability: 10 seeds, 30K steps each.
Characterize X/10 reliability and step-to-level distribution.
Compare to codebook baseline: 6/10 at 50K, median ~19K.
5-min cap per seed.
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


def run_seed(arc, game_id, seed, max_steps=30000):
    from arcengine import GameState
    torch.manual_seed(seed)
    sub = GraphSubstrate(d=256, n_actions=4, sim_thresh=0.99)
    env = arc.make(game_id); obs = env.reset()
    na = len(env.action_space)
    ts = go = lvls = 0; unique = set(); action_counts = [0]*na
    level_step = None
    t0 = time.time()

    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN:
            break
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
        if time.time() - t0 > 280: break

    elapsed = time.time() - t0
    dom = max(action_counts) / sum(action_counts) * 100
    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique': len(unique), 'nodes': len(sub.nodes),
        'dom': dom, 'elapsed': elapsed,
    }


def main():
    import arc_agi

    print(f"Step 443: Graph Substrate reliability — 10 seeds × 30K steps", flush=True)
    print(f"Device: {DEVICE}  sim_thresh=0.99", flush=True)
    print(flush=True)

    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP: LS20 not found"); return

    results = []
    for seed in range(10):
        r = run_seed(arc, ls20.game_id, seed=seed, max_steps=30000)
        status = f"LEVEL {r['levels']} at step {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed:2d}  {status:30s}  unique={r['unique']:5d}  nodes={r['nodes']:5d}"
              f"  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s", flush=True)
        results.append(r)

    print(f"\n{'='*60}", flush=True)
    print("STEP 443 FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)

    wins = [r for r in results if r['levels'] > 0]
    level_steps = [r['level_step'] for r in wins]
    reliability = len(wins)

    print(f"Reliability: {reliability}/10", flush=True)
    print(f"Step-to-level: {sorted(level_steps)}", flush=True)
    if level_steps:
        print(f"Median step-to-level: {sorted(level_steps)[len(level_steps)//2]}", flush=True)
        print(f"Min: {min(level_steps)}  Max: {max(level_steps)}", flush=True)
    print(f"Avg unique: {sum(r['unique'] for r in results)/len(results):.0f}", flush=True)
    print(f"Avg dom: {sum(r['dom'] for r in results)/len(results):.0f}%", flush=True)
    print(flush=True)
    print(f"Codebook baseline: 6/10 at 50K, median ~19K", flush=True)
    if reliability >= 6:
        print(f"MATCH OR BETTER than codebook baseline reliability!", flush=True)
    elif reliability >= 3:
        print(f"PARTIAL: lower reliability than codebook but navigating confirmed.", flush=True)
    else:
        print(f"LOW reliability — 1/3 in 442b may have been noise.", flush=True)


if __name__ == '__main__':
    main()
