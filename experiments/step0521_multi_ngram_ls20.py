#!/usr/bin/env python3
"""
Step 521 -- N-gram sequence agent on LS20. New family: sequence-aware.

Discretize via LSH k=12. Maintain history of last N (cell, action) tuples.
Action selection: pick action with LOWEST n-gram recency count from history.
This is sequence-aware (temporal context) vs memoryless argmin.

Prediction: 2/10 at 50K (worse than LSH 6/10 due to recency bias oscillation).
Kill: 0/5 at 50K.
5-min cap.

Variants:
  A: N=10 history, recency-weighted (recent events count more)
  B: N=5 history, uniform count
  C: N=20 history, uniform count
"""
import time, logging
from collections import deque
import numpy as np
logging.getLogger().setLevel(logging.WARNING)

K = 12
N_CELLS = 2 ** K       # 4096
N_ACTIONS = 4
MAX_STEPS = 50_000
N_SEEDS = 5


def encode_arc(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class NGramAgent:
    """
    Sequence-aware agent: picks action with lowest recent frequency from history.
    For current cell c, count how many times (c, a) appears in recent N steps.
    Pick argmin count (least recently used action from this cell).
    Falls back to global edge-count argmin when history is empty.
    """

    def __init__(self, n=10, k=K, seed=0, weighted=False):
        rng = np.random.RandomState(seed + 9999)
        self.H = rng.randn(k, 256).astype(np.float32)
        self.powers = np.array([1 << i for i in range(k)], dtype=np.int64)
        self.n = n
        self.weighted = weighted
        self.history = deque(maxlen=n)   # (cell, action) tuples
        self.global_edges = {}           # fallback: (cell, action) -> count
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()

    def _hash(self, x):
        bits = (self.H @ x > 0).astype(np.int64)
        return int(np.dot(bits, self.powers))

    def step(self, x):
        cell = self._hash(x)
        self.cells_seen.add(cell)

        # Update global edges (fallback)
        if self.prev_cell is not None:
            key = (self.prev_cell, self.prev_action)
            self.global_edges[key] = self.global_edges.get(key, 0) + 1

        # Recency count: how many times (cell, a) in recent history
        recency = {a: 0 for a in range(N_ACTIONS)}
        for i, (hc, ha) in enumerate(self.history):
            if hc == cell:
                if self.weighted:
                    # More weight to recent entries
                    weight = (i + 1) / len(self.history)
                    recency[ha] += weight
                else:
                    recency[ha] += 1

        # If any action has been seen in history: use recency argmin
        if any(v > 0 for v in recency.values()):
            min_r = min(recency.values())
            cands = [a for a, r in recency.items() if r == min_r]
        else:
            # Fall back to global edge-count argmin
            counts = [self.global_edges.get((cell, a), 0) for a in range(N_ACTIONS)]
            min_c = min(counts)
            cands = [a for a, c in enumerate(counts) if c == min_c]

        action = cands[int(np.random.randint(len(cands)))]
        self.history.append((cell, action))
        self.prev_cell = cell
        self.prev_action = action
        return action


def run_ls20(agent, arc, game_id, seed=0):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    action_space = env.action_space
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()
    while ts < MAX_STEPS:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1
            agent.history.clear()  # reset history on death (new episode)
            obs = env.reset()
            continue
        if obs.state == GameState.WIN: break
        if not obs.frame: obs = env.reset(); continue
        x = encode_arc(obs.frame)
        a = agent.step(x)
        obs_before = obs.levels_completed
        obs = env.step(action_space[a])
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
    cells = len(agent.cells_seen)
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    print(f"  seed={seed}: {status}  cells={cells}/{N_CELLS}  "
          f"global_edges={len(agent.global_edges)}  go={go}  steps={ts}  "
          f"{time.time()-t0:.0f}s", flush=True)
    return lvls > 0


def tier1_sanity():
    print("T1: N-gram agent sanity", flush=True)
    rng = np.random.RandomState(0)
    agent = NGramAgent(n=5, seed=0)

    # Deterministic hash
    x = rng.randn(256).astype(np.float32)
    x -= x.mean()
    c1 = agent._hash(x)
    c2 = agent._hash(x)
    assert c1 == c2, "Hash must be deterministic"

    # History fills correctly
    for i in range(3):
        xx = rng.randn(256).astype(np.float32)
        xx -= xx.mean()
        agent.step(xx)
    assert len(agent.history) == 3, f"History length wrong: {len(agent.history)}"

    # History caps at n
    for i in range(10):
        xx = rng.randn(256).astype(np.float32)
        xx -= xx.mean()
        agent.step(xx)
    assert len(agent.history) == 5, f"History should cap at {agent.n}"

    print(f"  T1 PASS: hash deterministic, history caps at n={agent.n}", flush=True)


def run_variant(name, n, weighted, seeds):
    wins = 0
    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())
    for s in seeds:
        agent = NGramAgent(n=n, seed=0, weighted=weighted)
        result = run_ls20(agent, arc, ls20.game_id, seed=s)
        if result: wins += 1
    print(f"  {name}: {wins}/{len(seeds)} WIN", flush=True)
    return wins


def main():
    t_total = time.time()
    print("Step 521: N-gram sequence agent on LS20", flush=True)
    print(f"k={K}  n_seeds={N_SEEDS}  max_steps={MAX_STEPS//1000}K", flush=True)
    print(f"Baseline: LSH graph Step 459 = 6/10. Prediction: ~2/10 (recency oscillation).", flush=True)

    tier1_sanity()

    seeds = list(range(N_SEEDS))

    print(f"\n--- Variant A: N=10 history, weighted (recency-decayed) ---", flush=True)
    wins_a = run_variant("A(N=10,weighted)", n=10, weighted=True, seeds=seeds)

    print(f"\n--- Variant B: N=5 history, uniform count ---", flush=True)
    wins_b = run_variant("B(N=5,uniform)", n=5, weighted=False, seeds=seeds)

    print(f"\n--- Variant C: N=20 history, uniform count ---", flush=True)
    wins_c = run_variant("C(N=20,uniform)", n=20, weighted=False, seeds=seeds)

    print(f"\n{'='*60}", flush=True)
    print("STEP 521 SUMMARY", flush=True)
    print(f"  Variant A (N=10, weighted): {wins_a}/{N_SEEDS}", flush=True)
    print(f"  Variant B (N=5,  uniform):  {wins_b}/{N_SEEDS}", flush=True)
    print(f"  Variant C (N=20, uniform):  {wins_c}/{N_SEEDS}", flush=True)
    print(f"  LSH graph baseline:         6/10", flush=True)

    best = max(wins_a, wins_b, wins_c)
    print(f"\nVERDICT:", flush=True)
    if best == 0:
        print(f"  KILL: Sequence-aware mechanism fails completely.", flush=True)
        print(f"  Recency bias does not help navigation.", flush=True)
    elif best < 3:
        print(f"  PARTIAL: {best}/{N_SEEDS} best variant.", flush=True)
        print(f"  Sequence context adds some signal but recency bias costs more.", flush=True)
    else:
        print(f"  PASS: {best}/{N_SEEDS} best variant.", flush=True)
        print(f"  Temporal context helps LS20 navigation.", flush=True)

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
