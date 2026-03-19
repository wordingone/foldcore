#!/usr/bin/env python3
"""
Step 479 — UCB1 action selection on LSH k=12. C sweep.
score(a) = C * sqrt(ln(total_visits+1) / max(edge_count(a), 1))
Pick highest score. C={0.5, 1.0, 2.0, 5.0}, 3 seeds x 50K per C.
Baseline: argmin = 6/10 at 50K (Step 459).
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
K = 12


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


class UCBGraph:
    def __init__(self, k=K, n_actions=4, seed=0, C=1.0):
        rng = np.random.RandomState(seed + 9999)
        self.H = rng.randn(k, 256).astype(np.float32)
        self.powers = np.array([1 << i for i in range(k)], dtype=np.int64)
        self.n_actions = n_actions
        self.C = C
        self.edges = {}
        self.cell_visits = {}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()

    def _hash(self, x):
        bits = (self.H @ x > 0).astype(np.int64)
        return int(np.dot(bits, self.powers))

    def step(self, x):
        cell = self._hash(x)
        self.cells_seen.add(cell)
        self.cell_visits[cell] = self.cell_visits.get(cell, 0) + 1

        if self.prev_cell is not None and self.prev_action is not None:
            d = self.edges.setdefault((self.prev_cell, self.prev_action), {})
            d[cell] = d.get(cell, 0) + 1

        total = self.cell_visits[cell]
        counts = [sum(self.edges.get((cell, a), {}).values()) for a in range(self.n_actions)]

        if total <= 1:
            # First visit or no history — pick randomly
            action = int(np.random.randint(self.n_actions))
        else:
            log_total = np.log(total + 1)
            scores = [self.C * np.sqrt(log_total / max(c, 1)) for c in counts]
            # Unvisited actions get max score
            for a in range(self.n_actions):
                if counts[a] == 0:
                    scores[a] = float('inf')
            max_score = max(scores)
            candidates = [a for a, s in enumerate(scores) if s == max_score]
            action = candidates[int(np.random.randint(len(candidates)))]

        self.prev_cell = cell
        self.prev_action = action
        return action


def run_seed(arc, game_id, seed, C, max_steps=50000):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    na = len(env.action_space)
    g = UCBGraph(k=K, n_actions=na, seed=seed, C=C)
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
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
        obs_before = obs.levels_completed
        obs = env.step(action, data=data)
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
        if time.time() - t0 > 280: break
    return {'seed': seed, 'C': C, 'levels': lvls, 'level_step': level_step,
            'unique_cells': len(g.cells_seen), 'occupancy': len(g.cells_seen) / (2**K),
            'elapsed': time.time() - t0}


def main():
    import arc_agi
    C_values = [0.5, 1.0, 2.0, 5.0]
    n_seeds = 3  # 4 C-values x 3 seeds = 12 runs ~5 min
    print(f"Step 479: UCB1 C={C_values} on LSH k={K}. {n_seeds} seeds x 50K.", flush=True)
    print(f"Baseline (argmin Step 459): 6/10 at 50K.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    t_total = time.time()
    all_results = {}
    for C in C_values:
        print(f"\n--- C={C} ---", flush=True)
        results = []
        for seed in range(n_seeds):
            r = run_seed(arc, ls20.game_id, seed=seed, C=C)
            status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
            print(f"  seed={seed}  {status:22s}  cells={r['unique_cells']:4d}/{2**K}"
                  f"  occ={r['occupancy']:.4f}  {r['elapsed']:.0f}s", flush=True)
            results.append(r)
        wins = [r for r in results if r['levels'] > 0]
        all_results[C] = {'wins': len(wins), 'level_steps': sorted([r['level_step'] for r in wins])}
        print(f"  -> {len(wins)}/{n_seeds}  steps={all_results[C]['level_steps']}", flush=True)
    print(f"\n{'='*50}", flush=True)
    print("STEP 479 SUMMARY", flush=True)
    for C in C_values:
        r = all_results[C]
        print(f"  C={C}: {r['wins']}/{n_seeds}  steps={r['level_steps']}", flush=True)
    best_C = max(C_values, key=lambda C: all_results[C]['wins'])
    best_wins = all_results[best_C]['wins']
    print("\nVERDICT:", flush=True)
    if best_wins >= 3:
        print(f"  UCB1 C={best_C} BEATS ARGMIN: {best_wins}/{n_seeds}. Exploration balance helps.", flush=True)
    elif best_wins >= 2:
        print(f"  UCB1 COMPARABLE: {best_wins}/{n_seeds} ~ argmin baseline (3/5 expected).", flush=True)
    else:
        print(f"  UCB1 WEAKER: {best_wins}/{n_seeds} < baseline. Argmin is better.", flush=True)
    print(f"Total elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
