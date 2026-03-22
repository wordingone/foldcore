#!/usr/bin/env python3
"""
Step 490 — Edge decay sweep. LSH k=12 + fresh H per level.
decay_factor in {0.999, 0.9999, 0.99999}: edge counts *= decay each step.
Hypothesis: decay prevents U25 convergence, breaks 259-cell plateau.
500K steps, seed 0. ~207s per decay value.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
K = 12
MAX_STEPS = 500_000
TIME_CAP = 220  # per decay run (207s expected + buffer)


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


class DecayArgminGraph:
    def __init__(self, k=K, n_actions=4, seed=0, level=0, decay=1.0):
        self.k = k
        self.n_actions = n_actions
        self.decay = decay
        self._reinit(seed, level)

    def _reinit(self, seed, level):
        rng = np.random.RandomState(seed * 1000 + level + 9999)
        self.H = rng.randn(self.k, 256).astype(np.float32)
        self.powers = np.array([1 << i for i in range(self.k)], dtype=np.int64)
        self.edges = {}   # (cell, action) -> float (decayed count)
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()

    def reset_for_level(self, seed, level):
        self._reinit(seed, level)

    def step(self, x):
        bits = (self.H @ x > 0).astype(np.int64)
        cell = int(np.dot(bits, self.powers))
        self.cells_seen.add(cell)

        # Update edge: decay existing count then increment
        if self.prev_cell is not None and self.prev_action is not None:
            key = (self.prev_cell, self.prev_action)
            old = self.edges.get(key, 0.0)
            self.edges[key] = old * self.decay + 1.0

        counts = [self.edges.get((cell, a), 0.0) for a in range(self.n_actions)]

        min_c = min(counts)
        candidates = [a for a, c in enumerate(counts) if c <= min_c + 1e-9]
        action = candidates[int(np.random.randint(len(candidates)))]
        self.prev_cell = cell
        self.prev_action = action
        return action


def run_decay(arc, game_id, seed, decay, max_steps=MAX_STEPS):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    na = len(env.action_space)
    current_level = 0
    g = DecayArgminGraph(k=K, n_actions=na, seed=seed, level=current_level, decay=decay)
    obs = env.reset()
    ts = go = 0
    prev_levels = 0
    level_steps = {}
    level_budgets = {}
    level_start_step = 0
    cells_at_100k = {}  # track cell growth
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
                level_budgets[lvl] = ts - level_start_step
            prev_levels = obs.levels_completed
            current_level = prev_levels
            level_start_step = ts
            g.reset_for_level(seed=seed, level=current_level)

        # Track cells at 100K, 200K, 300K after L1 (on L2)
        steps_on_level = ts - level_start_step
        if prev_levels >= 1 and steps_on_level in (100000, 200000, 300000):
            cells_at_100k[steps_on_level] = len(g.cells_seen)

        if time.time() - t0 > TIME_CAP: break

    elapsed = time.time() - t0
    l2_steps = ts - level_start_step if prev_levels >= 1 else 0
    return {
        'decay': decay,
        'max_levels': prev_levels,
        'level_steps': level_steps,
        'level_budgets': level_budgets,
        'l2_steps_used': l2_steps,
        'cells_final': len(g.cells_seen),
        'cells_milestones': cells_at_100k,
        'game_overs': go,
        'elapsed': elapsed,
        'timed_out': elapsed >= TIME_CAP - 1
    }


def main():
    import arc_agi
    seed = 0
    decay_values = [0.999, 0.9999, 0.99999]
    print(f"Step 490: Edge decay sweep. LSH k={K}, seed={seed}, {MAX_STEPS//1000}K steps.", flush=True)
    print(f"Decay = {decay_values}. Fresh H+edges per level.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    t0 = time.time()
    results = []
    for decay in decay_values:
        halflife = round(np.log(0.5) / np.log(decay))
        r = run_decay(arc, ls20.game_id, seed=seed, decay=decay)
        l1 = r['level_steps'].get(1, None)
        l2 = r['level_steps'].get(2, None)
        l1_str = f"L1@{l1}({r['level_budgets'].get(1,'?')})" if l1 else "L1-FAIL"
        l2_str = f"L2@{l2}({r['level_budgets'].get(2,'?')})" if l2 else "L2-none"
        mil = r['cells_milestones']
        cells_str = f"@100K={mil.get(100000,'?')} @200K={mil.get(200000,'?')} @300K={mil.get(300000,'?')}"
        timeout_flag = " [TIMEOUT]" if r['timed_out'] else ""
        print(f"  decay={decay} (hl~{halflife})  {l1_str}  {l2_str}  "
              f"cells_final={r['cells_final']}  {cells_str}  {r['elapsed']:.0f}s{timeout_flag}", flush=True)
        results.append(r)
    l2_wins = [r for r in results if r['max_levels'] >= 2]
    print(f"\nLevel 2 reached: {len(l2_wins)}/{len(decay_values)}", flush=True)
    print(f"\nVERDICT:", flush=True)
    if l2_wins:
        for r in l2_wins:
            print(f"  decay={r['decay']} BREAKS CONVERGENCE: L2@{r['level_steps'].get(2)} "
                  f"(budget={r['level_budgets'].get(2)} steps)", flush=True)
    else:
        print(f"  ALL DECAY RATES FAIL Level 2.", flush=True)
        cell_growth = [f"decay={r['decay']}: cells={r['cells_final']}" for r in results]
        print(f"  {', '.join(cell_growth)}", flush=True)
        print(f"  259-cell trap is SPATIAL, not convergence. Different mechanism needed.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
