#!/usr/bin/env python3
"""
Step 500 — VC33 full diagnostic. Spec.
VC33: 1 action, 50 visual states (Step 476). Timing game (I4).
1. Action space inspection (is there really only 1 action? complex?)
2. Reward signal (any at all in 50K steps?)
3. Transition structure: cycle? linear? branching?
4. Deaths: count, pattern, timing
5. Is the game deterministic? (same seed = same cycle?)
6. With 1 action: saturation speed, cycle detection
7. Death penalty sweep: pen=0 vs pen=10k — do deaths have timing structure?
3 seeds, 30K steps. Expected fast (~7s/seed).
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
MAX_STEPS = 30_000
TIME_CAP = 40
WARMUP = 500
N_CLUSTERS = 50  # match known 50 VC33 states


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


def inspect_action_space(arc, game_id):
    env = arc.make(game_id)
    na = len(env.action_space)
    print(f"\nVC33 Action space: {na} total actions", flush=True)
    for i, a in enumerate(env.action_space):
        print(f"  [{i}] {a}  complex={a.is_complex()}", flush=True)
    obs = env.reset()
    attrs = [attr for attr in dir(obs) if not attr.startswith('_')]
    has_reward = hasattr(obs, 'reward')
    has_score = hasattr(obs, 'score')
    print(f"  obs attrs: state, frame, levels_completed, win_levels present", flush=True)
    print(f"  has .reward: {has_reward}, has .score: {has_score}", flush=True)
    return na, has_reward, has_score


class KMeansGraphVC33:
    def __init__(self, n_clusters=N_CLUSTERS, n_actions=1, warmup=WARMUP, death_penalty=0):
        self.n_clusters = n_clusters
        self.n_actions = n_actions
        self.warmup = warmup
        self.death_penalty = death_penalty
        self.centroids = None
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()
        self._buf = []
        # Diagnostics
        self.saturation_step = None
        self.transitions = set()      # (from_cell, action, to_cell)
        self.cell_sequence = []       # ordered sequence of cells (for cycle detection)
        self.death_steps = []         # steps at which deaths occur
        self.step_count = 0

    def _fit(self):
        from sklearn.cluster import MiniBatchKMeans
        X = np.array(self._buf, dtype=np.float32)
        n = min(self.n_clusters, len(set(x.tobytes() for x in X)), len(X))
        n = max(n, 2)
        km = MiniBatchKMeans(n_clusters=n, random_state=42,
                             n_init=3, max_iter=100, batch_size=256)
        km.fit(X)
        self.centroids = km.cluster_centers_.astype(np.float32)
        self._buf = []

    def step(self, x):
        self.step_count += 1
        if self.centroids is None:
            self._buf.append(x.copy())
            if len(self._buf) >= self.warmup:
                self._fit()
            return 0  # only 1 action

        diffs = self.centroids - x
        cell = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        self.cells_seen.add(cell)
        self.cell_sequence.append(cell)

        if self.saturation_step is None and len(self.cells_seen) >= len(self.centroids):
            self.saturation_step = self.step_count

        if self.prev_cell is not None:
            self.transitions.add((self.prev_cell, 0, cell))
            d = self.edges.setdefault((self.prev_cell, 0), {})
            d[cell] = d.get(cell, 0) + 1

        self.prev_cell = cell
        self.prev_action = 0
        return 0  # always action 0

    def on_death(self):
        self.death_steps.append(self.step_count)
        if self.death_penalty > 0 and self.prev_cell is not None:
            d = self.edges.setdefault((self.prev_cell, 0), {})
            d[-1] = d.get(-1, 0) + self.death_penalty
        self.prev_cell = None
        self.prev_action = None


def detect_cycle(seq, min_len=3, max_len=100):
    """Detect if seq contains a repeating cycle."""
    if len(seq) < 2 * min_len:
        return None, None
    for period in range(min_len, min(max_len, len(seq) // 2)):
        # Check last 2*period elements
        a = seq[-2*period:-period]
        b = seq[-period:]
        if a == b:
            return period, seq[-period:]
    return None, None


def run_seed(arc, game_id, seed, na, has_reward, has_score, death_penalty=0):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    g = KMeansGraphVC33(n_clusters=N_CLUSTERS, n_actions=na,
                         warmup=WARMUP, death_penalty=death_penalty)
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    total_reward = 0.0
    max_reward = 0.0
    reward_events = 0
    t0 = time.time()

    while ts < MAX_STEPS:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1
            g.on_death()
            obs = env.reset()
            continue
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

        if has_reward and obs and hasattr(obs, 'reward') and obs.reward:
            r = obs.reward
            if r != 0:
                total_reward += r
                max_reward = max(max_reward, abs(r))
                reward_events += 1
        if obs and obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
        if time.time() - t0 > TIME_CAP: break

    n_c = len(g.centroids) if g.centroids is not None else 0
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    elapsed = time.time() - t0

    # Cycle detection on cell sequence (post-warmup)
    period, cycle = detect_cycle(list(g.cell_sequence), min_len=3, max_len=200)

    # Death timing analysis
    if len(g.death_steps) > 2:
        diffs = [g.death_steps[i+1] - g.death_steps[i] for i in range(min(10, len(g.death_steps)-1))]
        death_timing = f"intervals={diffs[:5]}"
    else:
        death_timing = f"count={len(g.death_steps)}"

    # Transition diversity: how many unique next-cells per (cell,action)?
    fanout = [len(v) for v in g.edges.values()]
    avg_fanout = sum(fanout) / len(fanout) if fanout else 0

    print(f"\n  seed={seed}  pen={death_penalty}  {status}", flush=True)
    print(f"    cells: {len(g.cells_seen)}/{n_c}  saturation={g.saturation_step}", flush=True)
    print(f"    transitions: {len(g.transitions)} unique triples  avg_fanout={avg_fanout:.2f}", flush=True)
    print(f"    cycle: period={period}  (first 10: {list(g.cell_sequence)[:10]})", flush=True)
    print(f"    deaths: go={go}  {death_timing}", flush=True)
    print(f"    reward: total={total_reward:.3f}  max={max_reward:.3f}  events={reward_events}", flush=True)
    print(f"    {elapsed:.0f}s", flush=True)

    # Sample transition structure
    print(f"    Transition sample:", flush=True)
    for (src, act), dest_d in sorted(g.edges.items())[:6]:
        dest_str = " ".join(f"c{k}x{v}" for k, v in sorted(dest_d.items())[:3])
        print(f"      cell{src} -> {dest_str}", flush=True)

    return {'levels': lvls, 'cells': len(g.cells_seen), 'n_c': n_c,
            'transitions': len(g.transitions), 'saturation': g.saturation_step,
            'period': period, 'go': go, 'reward_events': reward_events,
            'max_reward': max_reward, 'fanout': avg_fanout}


def main():
    import arc_agi
    n_seeds = 3
    print(f"Step 500: VC33 full diagnostic. {n_seeds} seeds, {MAX_STEPS//1000}K steps.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    vc33 = next((g for g in games if 'vc33' in g.game_id.lower()), None)
    if not vc33:
        print("SKIP — VC33 not found"); return

    t0 = time.time()

    na, has_reward, has_score = inspect_action_space(arc, vc33.game_id)

    print(f"\n--- pen=0 (baseline) ---", flush=True)
    results_base = []
    for seed in range(n_seeds):
        r = run_seed(arc, vc33.game_id, seed=seed, na=na,
                     has_reward=has_reward, has_score=has_score, death_penalty=0)
        results_base.append(r)

    print(f"\n--- pen=10000 (death avoidance) ---", flush=True)
    results_pen = []
    for seed in range(n_seeds):
        r = run_seed(arc, vc33.game_id, seed=seed, na=na,
                     has_reward=has_reward, has_score=has_score, death_penalty=10000)
        results_pen.append(r)

    print(f"\n{'='*50}", flush=True)
    print(f"STEP 500 SUMMARY:", flush=True)
    print(f"  Action space: {na} total", flush=True)
    wins_base = sum(1 for r in results_base if r['levels'] > 0)
    wins_pen = sum(1 for r in results_pen if r['levels'] > 0)
    avg_go_base = sum(r['go'] for r in results_base) / n_seeds
    avg_go_pen = sum(r['go'] for r in results_pen) / n_seeds
    periods = [r['period'] for r in results_base if r['period']]
    print(f"  Baseline: {wins_base}/{n_seeds}  avg_go={avg_go_base:.0f}", flush=True)
    print(f"  pen10k:   {wins_pen}/{n_seeds}  avg_go={avg_go_pen:.0f}", flush=True)
    print(f"  Cycle periods: {periods}", flush=True)
    any_reward = any(r['reward_events'] > 0 for r in results_base + results_pen)
    print(f"  Reward signal: {'YES' if any_reward else 'NONE'}", flush=True)

    print(f"\nVERDICT:", flush=True)
    if periods:
        print(f"  DETERMINISTIC CYCLE confirmed: period={periods[0]}", flush=True)
        if not any_reward:
            print(f"  ZERO reward — timing action doesn't trigger win via discrete action.", flush=True)
            print(f"  VC33 requires DIFFERENT action mechanism (click at specific position?)", flush=True)
        elif wins_base > 0:
            print(f"  NAVIGATION: {wins_base}/{n_seeds}. Timing works.", flush=True)
        else:
            print(f"  Reward exists but navigation fails. I4 timing problem.", flush=True)
    else:
        print(f"  No clean cycle detected. Game structure more complex.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
