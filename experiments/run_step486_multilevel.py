#!/usr/bin/env python3
"""
Step 486 — Multi-level navigation. LSH k=12 argmin, no graph reset at level transitions.
Q: After Level 1, does accumulated graph help or hurt Level 2?
200K steps, 5 seeds. Track per-level step counts + sig_q at transition.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
K = 12
MAX_STEPS = 200000
TIME_CAP = 150  # per seed


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


def compute_sig_q(edges, cells_seen, n_actions):
    qualities = []
    for c in cells_seen:
        counts = [sum(edges.get((c, a), {}).values()) for a in range(n_actions)]
        total = sum(counts)
        if total > 0:
            qualities.append((max(counts) - min(counts)) / total)
    return sum(qualities) / len(qualities) if qualities else 0.0


class ArgminGraph:
    def __init__(self, k=K, n_actions=4, seed=0):
        rng = np.random.RandomState(seed + 9999)
        self.H = rng.randn(k, 256).astype(np.float32)
        self.powers = np.array([1 << i for i in range(k)], dtype=np.int64)
        self.n_actions = n_actions
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()

    def step(self, x):
        bits = (self.H @ x > 0).astype(np.int64)
        cell = int(np.dot(bits, self.powers))
        self.cells_seen.add(cell)
        if self.prev_cell is not None and self.prev_action is not None:
            d = self.edges.setdefault((self.prev_cell, self.prev_action), {})
            d[cell] = d.get(cell, 0) + 1
        counts = [sum(self.edges.get((cell, a), {}).values()) for a in range(self.n_actions)]
        min_c = min(counts)
        candidates = [a for a, c in enumerate(counts) if c == min_c]
        action = candidates[int(np.random.randint(len(candidates)))]
        self.prev_cell = cell
        self.prev_action = action
        return action


def run_seed(arc, game_id, seed, max_steps=MAX_STEPS):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    na = len(env.action_space)
    g = ArgminGraph(k=K, n_actions=na, seed=seed)
    obs = env.reset()
    ts = go = 0
    prev_levels = 0
    level_steps = {}  # level_num -> step when completed
    sig_q_at_transition = {}  # level_num -> sig_q when that level completed
    t0 = time.time()

    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue

        x = centered_enc(avgpool16(obs.frame))
        action_idx = g.step(x)
        action = env.action_space[action_idx % na]
        data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0])
            cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}

        obs = env.step(action, data=data)
        ts += 1
        if obs is None: break

        if obs.levels_completed > prev_levels:
            for lvl in range(prev_levels + 1, obs.levels_completed + 1):
                level_steps[lvl] = ts
                sig_q_at_transition[lvl] = compute_sig_q(g.edges, g.cells_seen, na)
            prev_levels = obs.levels_completed

        if time.time() - t0 > TIME_CAP: break

    elapsed = time.time() - t0
    edge_count = sum(sum(d.values()) for d in g.edges.values())
    return {
        'seed': seed,
        'level_steps': level_steps,
        'sig_q_at_transition': sig_q_at_transition,
        'max_levels': prev_levels,
        'game_overs': go,
        'unique_cells': len(g.cells_seen),
        'edge_count': edge_count,
        'steps_reached': ts,
        'elapsed': elapsed,
        'timed_out': elapsed >= TIME_CAP - 1
    }


def main():
    import arc_agi
    n_seeds = 5
    print(f"Step 486: Multi-level LSH k={K}. {MAX_STEPS//1000}K steps, {n_seeds} seeds.", flush=True)
    print(f"No graph reset at level transitions. Track Level 1 + Level 2.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    t0 = time.time()
    results = []
    for seed in range(n_seeds):
        r = run_seed(arc, ls20.game_id, seed=seed)
        lvl1 = r['level_steps'].get(1)
        lvl2 = r['level_steps'].get(2)
        sq1 = r['sig_q_at_transition'].get(1, 0)
        sq2 = r['sig_q_at_transition'].get(2, 0)
        l1_str = f"L1@{lvl1}" if lvl1 else "L1-FAIL"
        l2_str = f"L2@{lvl2}" if lvl2 else "L2-none"
        timeout_flag = " [TIMEOUT]" if r['timed_out'] else ""
        print(f"  seed={seed}  {l1_str:12s}  {l2_str:12s}  "
              f"sq1={sq1:.3f}  sq2={sq2:.3f}  "
              f"cells={r['unique_cells']:4d}  edges={r['edge_count']:6d}  "
              f"{r['elapsed']:.0f}s{timeout_flag}", flush=True)
        results.append(r)
    wins_l1 = [r for r in results if 1 in r['level_steps']]
    wins_l2 = [r for r in results if 2 in r['level_steps']]
    print(f"\nLevel 1: {len(wins_l1)}/{n_seeds}  Level 2: {len(wins_l2)}/{n_seeds}", flush=True)
    if wins_l2:
        l2_steps_after_l1 = [r['level_steps'][2] - r['level_steps'][1] for r in wins_l2]
        avg_l2_extra = sum(l2_steps_after_l1) / len(l2_steps_after_l1)
        print(f"Steps from L1 to L2: {l2_steps_after_l1}  avg={avg_l2_extra:.0f}", flush=True)
    print(f"\nVERDICT:", flush=True)
    if len(wins_l2) >= 3:
        print(f"  TRANSFER WORKS: {len(wins_l2)}/{n_seeds} reach Level 2.", flush=True)
        print(f"  Accumulated graph does NOT contaminate Level 2 navigation.", flush=True)
    elif len(wins_l2) >= 1:
        print(f"  PARTIAL TRANSFER: {len(wins_l2)}/{n_seeds}. Graph partially helps.", flush=True)
    else:
        print(f"  GRAPH CONTAMINATION: 0/{n_seeds} reach Level 2.", flush=True)
        print(f"  Level 1 edges block Level 2 exploration. Graph needs reset or partitioning.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
