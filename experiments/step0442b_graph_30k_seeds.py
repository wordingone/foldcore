#!/usr/bin/env python3
"""
Step 442b — Graph Substrate, 30K steps, 3 seeds.
Approved after 10K showed unique=3379 (matches codebook baseline), dom=25%.
First non-codebook architecture to reach baseline exploration threshold.
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
            if node_idx not in self.edges[key]:
                self.edges[key][node_idx] = 0
            self.edges[key][node_idx] += 1

        visit_counts = []
        for a in range(self.n_actions):
            key = (node_idx, a)
            if key in self.edges:
                visit_counts.append(sum(self.edges[key].values()))
            else:
                visit_counts.append(0)

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
        mean_v = torch.stack(sub.nodes).mean(dim=0).cpu()
        t_unit = t_unit - mean_v
    return t_unit


def run_seed(arc, game_id, seed, max_steps=30000):
    from arcengine import GameState

    torch.manual_seed(seed)
    sub = GraphSubstrate(d=256, n_actions=4, sim_thresh=0.99)
    env = arc.make(game_id); obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0
    unique = set(); action_counts = [0]*na
    t0 = time.time()

    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN:
            print(f"  WIN at step {ts}!", flush=True); break

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
            print(f"  LEVEL {lvls} at step {ts}  nodes={len(sub.nodes)}  edges={len(sub.edges)}", flush=True)

        if ts % 5000 == 0:
            total = sum(action_counts); dom = max(action_counts)/total*100
            elapsed = time.time()-t0
            print(f"  [step {ts:5d}]  nodes={len(sub.nodes):5d}  edges={len(sub.edges):5d}"
                  f"  unique={len(unique):5d}  lvls={lvls}  go={go}"
                  f"  dom={dom:.0f}%  {elapsed:.0f}s", flush=True)
            if elapsed > 280:
                print("  TIME LIMIT — stopping seed", flush=True)
                break

    elapsed = time.time()-t0
    total = sum(action_counts); dom = max(action_counts)/total*100
    return {
        'seed': seed, 'levels': lvls, 'unique': len(unique),
        'nodes': len(sub.nodes), 'edges': len(sub.edges),
        'dom': dom, 'go': go, 'steps': ts, 'elapsed': elapsed,
        'action_counts': action_counts,
    }


def main():
    import arc_agi
    print(f"Step 442b: Graph Substrate, 30K steps, 3 seeds", flush=True)
    print(f"Device: {DEVICE}  sim_thresh=0.99", flush=True)
    print(flush=True)

    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP: LS20 not found"); return

    results = []
    for seed in [0, 1, 2]:
        print(f"\n--- Seed {seed} ---", flush=True)
        r = run_seed(arc, ls20.game_id, seed=seed, max_steps=30000)
        results.append(r)
        print(f"  Seed {seed}: levels={r['levels']}  unique={r['unique']}"
              f"  nodes={r['nodes']}  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("STEP 442b FINAL RESULTS (30K, 3 seeds)", flush=True)
    print(f"{'='*60}", flush=True)
    for r in results:
        print(f"  seed={r['seed']}  levels={r['levels']}  unique={r['unique']:5d}"
              f"  nodes={r['nodes']:5d}  edges={r['edges']:5d}"
              f"  dom={r['dom']:.0f}%  go={r['go']}  {r['elapsed']:.0f}s", flush=True)

    levels_reached = [r['levels'] for r in results]
    avg_unique = sum(r['unique'] for r in results) / len(results)
    avg_dom = sum(r['dom'] for r in results) / len(results)
    print(f"\nSummary: levels={levels_reached}  avg_unique={avg_unique:.0f}  avg_dom={avg_dom:.0f}%", flush=True)

    if any(l > 0 for l in levels_reached):
        print("BREAKTHROUGH: First non-codebook architecture to pass navigation gate!", flush=True)
    elif avg_unique > 3000:
        print("STRONG: Exploration matches codebook baseline. Needs more steps for navigation.", flush=True)


if __name__ == '__main__':
    main()
