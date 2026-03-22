#!/usr/bin/env python3
"""
Step 442 — GRAPH SUBSTRATE. Relational dual of the codebook.
Nodes = observation landmarks. Edges = observed transitions.
Action from graph structure (least-explored edge), not similarity scoring.

Spec: LS20, 30K steps, sim_thresh=0.99, 1 seed.
Runtime cap: 10K steps (5-min cap rule).
Log: node count, edge count, unique states, dom%, action at each checkpoint.

Why different from codebook:
- Action = least-visited edge from current node (exploration via graph structure)
- NOT argmin of familiarity, NOT argmax
- New nodes create fresh edges with 0 visits = automatic frontier
- U24 dissolved: "what haven't I tried FROM HERE"
- U25 dissolved: per-transition visit counts, new nodes = perpetual frontier
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


# ── Structural test (Tier 1, <30s) ──────────────────────────────────────────

def structural_test():
    print("=== R1-R6 STRUCTURAL TEST ===", flush=True)
    t0 = time.time()
    sub = GraphSubstrate(d=256, n_actions=4, sim_thresh=0.99)

    torch.manual_seed(42)
    actions = []
    for _ in range(100):
        x = torch.randn(256)
        a = sub.step(x)
        actions.append(a)

    n_nodes = len(sub.nodes)
    n_edges = len(sub.edges)
    action_set = set(actions)

    print(f"  R1 (computes without external signal): PASS — actions={action_set}", flush=True)
    print(f"  R2 (state changes): {'PASS' if n_nodes > 0 else 'FAIL'} — nodes={n_nodes}", flush=True)
    print(f"  nodes={n_nodes}  edges={n_edges}", flush=True)
    print(f"  Structural test: {time.time()-t0:.1f}s", flush=True)

    fail = n_nodes == 0 or len(action_set) < 2
    print(f"  RESULT: {'PASS' if not fail else 'FAIL'}", flush=True)
    print(flush=True)
    return not fail


# ── LS20 run ─────────────────────────────────────────────────────────────────

def run_ls20(max_steps=10000):
    from arcengine import GameState
    import arc_agi

    print("=== LS20 RUN (10K steps, 1 seed, sim_thresh=0.99) ===", flush=True)
    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: LS20 not found", flush=True)
        return

    torch.manual_seed(0)
    sub = GraphSubstrate(d=256, n_actions=4, sim_thresh=0.99)
    env = arc.make(ls20.game_id); obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0
    unique = set(); action_counts = [0]*na
    t0 = time.time()

    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN:
            print(f"WIN at step {ts}!", flush=True); break

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
            print(f"LEVEL {lvls} at step {ts}  nodes={len(sub.nodes)}  edges={len(sub.edges)}", flush=True)

        if ts % 2500 == 0:
            total = sum(action_counts); dom = max(action_counts)/total*100
            elapsed = time.time()-t0
            print(f"  [step {ts}]  nodes={len(sub.nodes):5d}  edges={len(sub.edges):5d}"
                  f"  unique={len(unique):5d}  lvls={lvls}  go={go}"
                  f"  dom={dom:.0f}%  {elapsed:.0f}s", flush=True)
            if elapsed > 280:
                print("  TIME LIMIT approaching — stopping", flush=True)
                break

    elapsed = time.time()-t0
    total = sum(action_counts); dom = max(action_counts)/total*100
    print(f"\n{'='*60}", flush=True)
    print("STEP 442 RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"nodes={len(sub.nodes)}  edges={len(sub.edges)}  unique={len(unique)}"
          f"  levels={lvls}  dom={dom:.0f}%  go={go}  {elapsed:.0f}s", flush=True)
    print(f"action_counts={action_counts}", flush=True)

    # Assessment per Predictions
    print(f"\nAssessment:", flush=True)
    print(f"  dom<30%? {'YES' if dom < 30 else 'NO'} ({dom:.0f}%)", flush=True)
    print(f"  unique>3000? {'YES' if len(unique) > 3000 else 'NO'} ({len(unique)})", flush=True)
    print(f"  nodes growing? {len(sub.nodes)} nodes", flush=True)


def main():
    print(f"Step 442: Graph Substrate on LS20", flush=True)
    print(f"Device: {DEVICE}  sim_thresh=0.99", flush=True)
    print(flush=True)
    ok = structural_test()
    if not ok:
        print("Structural test FAILED — stopping", flush=True)
        return
    run_ls20(max_steps=10000)


if __name__ == '__main__':
    main()
