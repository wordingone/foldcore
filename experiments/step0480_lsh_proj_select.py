#!/usr/bin/env python3
"""
Step 480 — Hash projection selection via early sig_q.
N=10 projections, each probed for 2K steps. Best sig_q selected.
Then 48K steps with selected projection. Total: 68K per seed.
Baseline: LSH k=12 = 6/10 at 50K (Step 459).
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
K = 12
N_PROBES = 10
PROBE_STEPS = 2000
MAIN_STEPS = 48000


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


def sig_q(edges, cells_seen, n_actions):
    qualities = []
    for c in cells_seen:
        counts = [sum(edges.get((c, a), {}).values()) for a in range(n_actions)]
        total = sum(counts)
        if total > 0:
            qualities.append((max(counts) - min(counts)) / total)
    return sum(qualities) / len(qualities) if qualities else 0.0


class LSHGraph:
    def __init__(self, H, k=K, n_actions=4):
        self.H = H
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


def probe_projection(arc, game_id, H, seed, na, max_steps):
    """Run a probe: return sig_q after max_steps."""
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    g = LSHGraph(H, n_actions=na)
    obs = env.reset()
    ts = 0
    t0 = time.time()
    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: obs = env.reset(); continue
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
    return sig_q(g.edges, g.cells_seen, na)


def run_main(arc, game_id, H, seed, na, max_steps):
    """Run main phase with selected projection."""
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    g = LSHGraph(H, n_actions=na)
    obs = env.reset()
    ts = lvls = 0
    level_step = None
    t0 = time.time()
    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: obs = env.reset(); continue
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
    return {'levels': lvls, 'level_step': level_step,
            'unique_cells': len(g.cells_seen)}


def run_seed(arc, game_id, seed, na):
    t0 = time.time()
    # Generate N probe projections
    matrices = []
    for i in range(N_PROBES):
        rng = np.random.RandomState(seed * 10000 + i * 37 + 9999)
        H = rng.randn(K, 256).astype(np.float32)
        matrices.append(H)

    # Probe each projection
    probe_scores = []
    for i, H in enumerate(matrices):
        score = probe_projection(arc, game_id, H, seed=seed, na=na, max_steps=PROBE_STEPS)
        probe_scores.append(score)

    best_i = int(np.argmax(probe_scores))
    best_score = probe_scores[best_i]
    best_H = matrices[best_i]

    # Run main phase with best projection
    result = run_main(arc, game_id, best_H, seed=seed, na=na, max_steps=MAIN_STEPS)
    result.update({
        'seed': seed, 'best_probe': best_i, 'best_probe_sig_q': best_score,
        'all_probe_sig_q': probe_scores, 'elapsed': time.time() - t0
    })
    return result


def main():
    import arc_agi
    n_seeds = 5
    print(f"Step 480: Projection selection via sig_q. N={N_PROBES} probes x {PROBE_STEPS} steps.", flush=True)
    print(f"Then {MAIN_STEPS} steps with best projection. Baseline: 6/10.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    # Get n_actions
    env0 = arc.make(ls20.game_id)
    na = len(env0.action_space)
    t0 = time.time()
    results = []
    for seed in range(n_seeds):
        r = run_seed(arc, ls20.game_id, seed=seed, na=na)
        status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
        probe_max = max(r['all_probe_sig_q'])
        probe_min = min(r['all_probe_sig_q'])
        print(f"  seed={seed}  {status:22s}  best_probe={r['best_probe']}(sig_q={r['best_probe_sig_q']:.3f})"
              f"  probe_range=[{probe_min:.3f},{probe_max:.3f}]  {r['elapsed']:.0f}s", flush=True)
        results.append(r)
    wins = [r for r in results if r['levels'] > 0]
    print(f"\n{len(wins)}/{n_seeds}  level_steps={sorted([r['level_step'] for r in wins])}", flush=True)
    print("\nVERDICT:", flush=True)
    if len(wins) >= 4:
        print(f"  SELECTION WORKS: {len(wins)}/{n_seeds}. Early sig_q predicts navigation success.", flush=True)
    elif len(wins) >= 3:
        print(f"  SELECTION COMPARABLE: {len(wins)}/{n_seeds} ~ baseline. Early sig_q doesn't select.", flush=True)
    else:
        print(f"  SELECTION FAILS: {len(wins)}/{n_seeds}. sig_q at 2K is not predictive.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
