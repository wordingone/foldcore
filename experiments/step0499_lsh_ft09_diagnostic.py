#!/usr/bin/env python3
"""
Step 499 — FT09 full diagnostic. Spec.
1. Full action space inspection (ALL actions, is_complex flags)
2. Does the game EVER give reward? (track reward signals)
3. Transition structure: unique (cell, action, next_cell) triples
4. Unique (cell, action) pairs visited (target: 192 = 32x6)
5. Saturation speed: how many steps to visit all 32 cells?
6. Action distribution per cell: uniform or collapsed?
7. Max consecutive steps in same cell
8. Check if complex/click actions exist (Avir: codebook used 69-class CLICK)
3 seeds, 30K steps (fast game, ~7s/seed expected).
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
MAX_STEPS = 30_000
TIME_CAP = 30
WARMUP = 500
N_CLUSTERS = 32


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


class KMeansGraphDiagnostic:
    def __init__(self, n_clusters=N_CLUSTERS, n_actions=6, warmup=WARMUP):
        self.n_clusters = n_clusters
        self.n_actions = n_actions
        self.warmup = warmup
        self.centroids = None
        self.edges = {}          # (cell, action) -> {next_cell: count}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()
        self._buf = []
        # Diagnostics
        self.saturation_step = None   # step when all cells first seen
        self.max_same_cell_streak = 0
        self._curr_streak = 0
        self._last_cell = None
        self.action_counts = {}       # cell -> [count per action]
        self.transitions = set()      # (from_cell, action, to_cell)
        self.cell_action_pairs = set()

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

    def step(self, x, global_step):
        if self.centroids is None:
            self._buf.append(x.copy())
            if len(self._buf) >= self.warmup:
                self._fit()
            return int(np.random.randint(self.n_actions))

        diffs = self.centroids - x
        cell = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        self.cells_seen.add(cell)

        # Saturation tracking
        if self.saturation_step is None and len(self.cells_seen) >= len(self.centroids):
            self.saturation_step = global_step

        # Streak tracking
        if cell == self._last_cell:
            self._curr_streak += 1
            self.max_same_cell_streak = max(self.max_same_cell_streak, self._curr_streak)
        else:
            self._curr_streak = 1
            self._last_cell = cell

        # Edge + transition tracking
        if self.prev_cell is not None and self.prev_action is not None:
            d = self.edges.setdefault((self.prev_cell, self.prev_action), {})
            d[cell] = d.get(cell, 0) + 1
            self.transitions.add((self.prev_cell, self.prev_action, cell))

        # Action selection (argmin)
        counts = [sum(self.edges.get((cell, a), {}).values()) for a in range(self.n_actions)]
        min_c = min(counts)
        candidates = [a for a, c in enumerate(counts) if c == min_c]
        action = candidates[int(np.random.randint(len(candidates)))]

        # Track (cell, action) pairs and action distribution
        self.cell_action_pairs.add((cell, action))
        ac = self.action_counts.setdefault(cell, [0] * self.n_actions)
        ac[action] += 1

        self.prev_cell = cell
        self.prev_action = action
        return action


def inspect_action_space(arc, game_id):
    """Full action space inspection."""
    env = arc.make(game_id)
    na = len(env.action_space)
    print(f"\nAction space: {na} total actions", flush=True)
    complex_count = 0
    for i, a in enumerate(env.action_space):
        is_cx = a.is_complex()
        if is_cx:
            complex_count += 1
        print(f"  [{i}] {a}  complex={is_cx}", flush=True)
    print(f"Complex actions: {complex_count}/{na}", flush=True)
    return na, complex_count


def check_reward_api(arc, game_id):
    """Check if obs has a reward attribute."""
    from arcengine import GameState
    env = arc.make(game_id)
    obs = env.reset()
    # Check obs attributes
    attrs = [attr for attr in dir(obs) if not attr.startswith('_')]
    print(f"\nObservation attributes: {attrs}", flush=True)
    has_reward = hasattr(obs, 'reward')
    has_score = hasattr(obs, 'score')
    print(f"  has .reward: {has_reward}", flush=True)
    print(f"  has .score: {has_score}", flush=True)
    if has_reward:
        print(f"  obs.reward = {obs.reward}", flush=True)
    if has_score:
        print(f"  obs.score = {obs.score}", flush=True)
    return has_reward, has_score


def run_seed(arc, game_id, seed, na, has_reward, has_score):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    g = KMeansGraphDiagnostic(n_clusters=N_CLUSTERS, n_actions=na, warmup=WARMUP)
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    total_reward = 0.0
    max_reward = 0.0
    reward_events = 0
    t0 = time.time()
    while ts < MAX_STEPS:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue
        x = centered_enc(avgpool16(obs.frame))
        action_idx = g.step(x, ts)
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
        # Reward tracking
        if has_reward and obs is not None and hasattr(obs, 'reward'):
            r = obs.reward or 0.0
            if r != 0:
                total_reward += r
                max_reward = max(max_reward, abs(r))
                reward_events += 1
        if has_score and obs is not None and hasattr(obs, 'score'):
            sc = obs.score or 0.0
            if sc != 0 and sc > max_reward:
                max_reward = sc
                reward_events += 1
        if obs is not None and obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
        if time.time() - t0 > TIME_CAP: break

    n_c = len(g.centroids) if g.centroids is not None else 0
    n_transitions = len(g.transitions)
    n_pairs = len(g.cell_action_pairs)
    max_pairs = n_c * na

    # Action distribution entropy per cell
    entropies = []
    for cell, ac in g.action_counts.items():
        total = sum(ac)
        if total > 0:
            probs = [c / total for c in ac if c > 0]
            ent = -sum(p * np.log(p) for p in probs)
            entropies.append(ent)
    avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    max_entropy = np.log(na)

    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    print(f"\n  seed={seed}  {status}", flush=True)
    print(f"    cells: {len(g.cells_seen)}/{n_c}  saturation_step={g.saturation_step}", flush=True)
    print(f"    (cell,action) pairs: {n_pairs}/{max_pairs} ({100*n_pairs/max(max_pairs,1):.0f}%)", flush=True)
    print(f"    transitions: {n_transitions} unique (cell,action,next_cell) triples", flush=True)
    print(f"    max_transitions_possible: {n_c * na * n_c}", flush=True)
    print(f"    max_same_cell_streak: {g.max_same_cell_streak}", flush=True)
    print(f"    action_entropy: {avg_entropy:.3f}/{max_entropy:.3f} ({100*avg_entropy/max_entropy:.0f}% uniform)", flush=True)
    print(f"    reward: total={total_reward:.3f}  max={max_reward:.3f}  events={reward_events}", flush=True)
    print(f"    go={go}  {time.time()-t0:.0f}s", flush=True)

    # Sample transition structure: for first 3 cells, show where actions lead
    print(f"    Transition sample (first 3 cells):", flush=True)
    for cell in sorted(g.edges.keys(), key=lambda k: k[0])[:6]:
        src, act = cell
        dest_dist = g.edges[cell]
        dest_str = " ".join(f"c{k}x{v}" for k, v in sorted(dest_dist.items())[:4])
        print(f"      cell{src} act{act} -> {dest_str}", flush=True)

    return {'levels': lvls, 'cells': len(g.cells_seen), 'n_c': n_c,
            'pairs': n_pairs, 'max_pairs': max_pairs,
            'transitions': n_transitions, 'saturation': g.saturation_step,
            'streak': g.max_same_cell_streak, 'reward_events': reward_events,
            'max_reward': max_reward, 'go': go}


def main():
    import arc_agi
    n_seeds = 3
    print(f"Step 499: FT09 full diagnostic. {n_seeds} seeds, {MAX_STEPS//1000}K steps.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ft09 = next((g for g in games if 'ft09' in g.game_id.lower()), None)
    if not ft09:
        print("SKIP — FT09 not found"); return

    t0 = time.time()

    # Inspect action space
    na, complex_count = inspect_action_space(arc, ft09.game_id)

    # Check reward API
    has_reward, has_score = check_reward_api(arc, ft09.game_id)

    print(f"\nRunning {n_seeds} seeds...", flush=True)
    results = []
    for seed in range(n_seeds):
        r = run_seed(arc, ft09.game_id, seed=seed, na=na,
                     has_reward=has_reward, has_score=has_score)
        results.append(r)

    wins = sum(1 for r in results if r['levels'] > 0)
    any_reward = any(r['reward_events'] > 0 for r in results)
    avg_transitions = sum(r['transitions'] for r in results) / n_seeds

    print(f"\n{'='*50}", flush=True)
    print(f"STEP 499 SUMMARY:", flush=True)
    print(f"  Action space: {na} total, {complex_count} complex", flush=True)
    print(f"  Navigation: {wins}/{n_seeds}", flush=True)
    print(f"  Reward signal: {'YES' if any_reward else 'NONE in {MAX_STEPS} steps'}", flush=True)
    print(f"  Avg transitions: {avg_transitions:.0f} unique triples", flush=True)
    avg_pairs = sum(r['pairs'] for r in results) / n_seeds
    max_pairs = results[0]['max_pairs']
    print(f"  Avg (cell,action) pairs: {avg_pairs:.0f}/{max_pairs}", flush=True)

    print(f"\nVERDICT:", flush=True)
    if not any_reward:
        print(f"  NO REWARD IN {MAX_STEPS//1000}K STEPS. Win condition likely requires:", flush=True)
        if complex_count == 0:
            print(f"  - Complex/click actions NOT in action space ({na} discrete only).", flush=True)
            print(f"  - Avir: codebook used 69-class CLICK — FT09 may need spatial clicks.", flush=True)
            print(f"  - The 6 discrete actions may be an INCOMPLETE action space for FT09.", flush=True)
        else:
            print(f"  - {complex_count} complex actions exist but argmin not using them effectively.", flush=True)
    elif wins > 0:
        print(f"  NAVIGATION ACHIEVED: {wins}/{n_seeds}", flush=True)
    else:
        print(f"  Reward exists but navigation fails. Transition graph:", flush=True)
        print(f"  {avg_transitions:.0f} unique triples / {max_pairs * results[0]['n_c']} max possible", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
