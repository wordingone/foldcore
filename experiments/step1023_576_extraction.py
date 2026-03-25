"""
Step 1023 — 576-Extraction: Mode Map + CC Discovery WITHOUT Graph

DEBATE EXPERIMENT #1 (prosecution, D2-grounded framework).

Hypothesis: Step 576's discovery mechanism (mode map → isolated CC) is separable
from graph navigation. The graph provided exploration bias during warmup; random
exploration should accumulate the same mode map because 5000 random clicks on 64
positions gives ~78 clicks each — plenty for frequency statistics.

Method:
  Phase 1 (warmup, 5000 steps): Click random grid positions. Accumulate per-pixel
  color frequency. Compute mode map at step 5000.
  Phase 2 (discover): Find isolated connected components (size 4-60, non-background).
  Phase 3 (navigate): Cycle through discovered cluster centroids with BURST=5 clicks
  each. Track level transitions.

Three conditions:
  A) 576-original: Graph argmin exploration + CC discovery (CONTROL — should match 576)
  B) 576-no-graph: Random exploration + CC discovery (TEST — is graph needed?)
  C) 576-delta: 800b delta-EMA exploration + CC discovery (D2-grounded substrate)

All conditions: no graph ban violations (B and C use no per-(state,action) storage).
Condition A uses graph for comparison only.

RHAE measurement: prescription_length / actual_steps for each level solved.
VC33 L1 prescription = 3 clicks. If L1 solved at step N, RHAE_L1 = 3/N.

Kill: B and C both 0/5 → discovery needs graph, separability fails
Signal: B or C ≥ 3/5 → discovery works without graph
"""
import os, sys, time
os.environ['PYTHONUTF8'] = '1'
import logging
logging.getLogger().setLevel(logging.WARNING)
import numpy as np
from scipy.ndimage import label as ndlabel

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import arc_agi
from arcengine import GameAction, GameState

# ─── Constants ───
MODE_WARMUP = 5000
MIN_CLUSTER = 4
MAX_CLUSTER = 60
MAX_STEPS = 50_000
TIME_CAP = 60
BURST = 5
BG_COLOR = 0
N_GRID = 64
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]
ALPHA_DELTA = 0.1
EPSILON_RANDOM = 0.20

# VC33 L1 prescription (from D2 Step 1020)
VC33_L1_PRESCRIPTION = 3  # 3 clicks at (62,34)

# ─── Mode map + CC discovery (from Step 576) ───

def update_freq(freq_arr, frame):
    arr = np.array(frame[0], dtype=np.int32)
    r = np.arange(64)[:, None]
    c = np.arange(64)[None, :]
    freq_arr[r, c, arr] += 1

def compute_mode(freq_arr):
    return np.argmax(freq_arr, axis=2).astype(np.int32)

def find_isolated_clusters(mode_arr, min_sz=MIN_CLUSTER, max_sz=MAX_CLUSTER):
    clusters = []
    for color in range(1, 16):
        mask = (mode_arr == color)
        if not mask.any():
            continue
        labeled, n = ndlabel(mask)
        for cid in range(1, n + 1):
            region = (labeled == cid)
            sz = int(region.sum())
            if min_sz <= sz <= max_sz:
                ys, xs = np.where(region)
                clusters.append({
                    'cy': float(ys.mean()), 'cx': float(xs.mean()),
                    'color': int(color), 'size': sz,
                    'cy_int': int(round(ys.mean())),
                    'cx_int': int(round(xs.mean())),
                })
    return clusters

# ─── Encoding (from 800b / 674) ───

def enc_frame(frame):
    """avgpool16 → 256D → center."""
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()  # 256D
    return x

# ─── Exploration strategies ───

class GraphArgmin:
    """576-style graph argmin (CONTROL — uses banned per-(state,action) storage)."""
    def __init__(self):
        self.G = {}
        self._prev_hash = None
        self._prev_action = None

    def observe(self, frame):
        h = hash(frame[0].tobytes())
        if self._prev_hash is not None:
            d = self.G.setdefault((self._prev_hash, self._prev_action), {})
            d[h] = d.get(h, 0) + 1
        self._prev_hash = h

    def select(self, n_actions):
        counts = [sum(self.G.get((self._prev_hash, a), {}).values())
                  for a in range(n_actions)]
        min_c = min(counts)
        candidates = [a for a, c in enumerate(counts) if c == min_c]
        a = candidates[int(np.random.randint(len(candidates)))]
        self._prev_action = a
        return a

    def reset(self):
        self._prev_hash = None

class RandomExplore:
    """Uniform random click selection (TEST — no state storage)."""
    def observe(self, frame):
        pass
    def select(self, n_actions):
        return int(np.random.randint(n_actions))
    def reset(self):
        pass

class DeltaExplore:
    """800b-style delta EMA (D2-GROUNDED — no per-(state,action) storage)."""
    def __init__(self, n_actions):
        self.delta = np.zeros(n_actions, dtype=np.float32)
        self.prev_enc = None
        self.prev_action = None

    def observe(self, frame):
        enc = enc_frame(frame)
        if self.prev_enc is not None and self.prev_action is not None:
            change = float(np.linalg.norm(enc - self.prev_enc))
            a = self.prev_action
            self.delta[a] = (1 - ALPHA_DELTA) * self.delta[a] + ALPHA_DELTA * change
        self.prev_enc = enc

    def select(self, n_actions):
        if np.random.rand() < EPSILON_RANDOM or self.delta.max() == 0:
            a = int(np.random.randint(n_actions))
        else:
            a = int(np.argmax(self.delta[:n_actions]))
        self.prev_action = a
        return a

    def reset(self):
        self.prev_enc = None

# ─── Run one seed ───

def run_seed(arc, game_id, seed, explorer_factory, label):
    """Run one seed with given exploration strategy. Returns result dict."""
    np.random.seed(seed * 1000 + hash(label) % 1000)

    env = arc.make(game_id)
    action6 = list(GameAction)[6]  # ACTION6 = click

    freq = np.zeros((64, 64, 16), dtype=np.int32)
    clusters = []
    phase = 'explore'
    nav_cluster = 0
    nav_burst = 0

    explorer = explorer_factory()
    obs = env.reset()
    steps = 0
    levels = 0
    l1_step = None
    t0 = time.time()

    while steps < MAX_STEPS:
        if time.time() - t0 > TIME_CAP:
            break
        if obs is None:
            obs = env.reset()
            explorer.reset()
            nav_burst = 0
            continue
        if obs.state == GameState.GAME_OVER:
            obs = env.reset()
            explorer.reset()
            nav_burst = 0
            continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset()
            continue

        # Accumulate mode map
        update_freq(freq, obs.frame)
        explorer.observe(obs.frame)

        # Check for mode map computation
        if steps == MODE_WARMUP and phase == 'explore':
            mode = compute_mode(freq)
            clusters = find_isolated_clusters(mode)
            if clusters:
                phase = 'navigate'

        # Action selection
        if phase == 'explore':
            a = explorer.select(N_GRID)
            cx, cy = CLICK_GRID[a]
        else:
            if clusters:
                if nav_burst <= 0:
                    nav_cluster = (nav_cluster + 1) % len(clusters)
                    nav_burst = BURST
                cx = clusters[nav_cluster]['cx_int']
                cy = clusters[nav_cluster]['cy_int']
                nav_burst -= 1
            else:
                cx, cy = CLICK_GRID[0]

        lvls_before = obs.levels_completed
        obs = env.step(action6, data={"x": cx, "y": cy})
        steps += 1

        if obs is None:
            break
        if obs.levels_completed > lvls_before:
            levels = obs.levels_completed
            if l1_step is None:
                l1_step = steps

    elapsed = time.time() - t0
    cluster_info = [(c['cx_int'], c['cy_int'], c['color'], c['size']) for c in clusters]

    return {
        'seed': seed, 'label': label, 'levels': levels,
        'l1_step': l1_step, 'steps': steps, 'elapsed': round(elapsed, 1),
        'n_clusters': len(clusters), 'clusters': cluster_info,
        'phase_at_end': phase,
    }

# ─── Main ───

def main():
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    vc33 = next((e for e in envs if 'vc33' in e.game_id.lower()), None)
    if vc33 is None:
        print("SKIP — VC33 not found")
        return

    print("=== Step 1023: 576-Extraction (D2 Debate Experiment #1) ===")
    print(f"Game: {vc33.game_id}")
    print(f"Mode map warmup: {MODE_WARMUP} steps")
    print(f"Max steps: {MAX_STEPS}, time cap: {TIME_CAP}s/seed")
    print()

    conditions = {
        'A_graph': lambda: GraphArgmin(),
        'B_random': lambda: RandomExplore(),
        'C_delta': lambda: DeltaExplore(N_GRID),
    }

    all_results = {}
    N_SEEDS = 5

    for cond_name, factory in conditions.items():
        print(f"--- Condition {cond_name} ---")
        results = []
        for seed in range(N_SEEDS):
            r = run_seed(arc, vc33.game_id, seed, factory, cond_name)
            status = f"L{r['levels']}@{r['l1_step']}" if r['levels'] > 0 else "FAIL"
            print(f"  s{seed}: {status:12s}  clusters={r['n_clusters']}  "
                  f"phase={r['phase_at_end']}  steps={r['steps']}  {r['elapsed']}s")
            if r['clusters']:
                print(f"    cluster positions: {r['clusters']}")
            results.append(r)
        all_results[cond_name] = results
        wins = sum(1 for r in results if r['levels'] > 0)
        discovered = sum(1 for r in results if r['n_clusters'] > 0)
        print(f"  {cond_name}: {wins}/{N_SEEDS} wins, {discovered}/{N_SEEDS} discovered clusters")
        print()

    # ─── RHAE calculation ───
    print("=== RHAE Summary ===")
    print(f"{'Condition':<12} {'Wins':>5} {'Discovery':>10} {'Avg L1 step':>12} {'RHAE_L1':>10}")
    for cond_name, results in all_results.items():
        wins = sum(1 for r in results if r['levels'] > 0)
        disc = sum(1 for r in results if r['n_clusters'] > 0)
        l1_steps = [r['l1_step'] for r in results if r['l1_step'] is not None]
        avg_l1 = np.mean(l1_steps) if l1_steps else float('inf')
        rhae_l1 = VC33_L1_PRESCRIPTION / avg_l1 if l1_steps else 0.0
        print(f"{cond_name:<12} {wins:>5} {disc:>10} {avg_l1:>12.0f} {rhae_l1:>10.4f}")

    # ─── Verdict ───
    print()
    a_wins = sum(1 for r in all_results['A_graph'] if r['levels'] > 0)
    b_wins = sum(1 for r in all_results['B_random'] if r['levels'] > 0)
    c_wins = sum(1 for r in all_results['C_delta'] if r['levels'] > 0)

    print("=== Debate Verdict ===")
    if b_wins >= 3 or c_wins >= 3:
        print(f"SEPARABILITY CONFIRMED: Discovery works without graph.")
        print(f"  B (random): {b_wins}/5, C (delta): {c_wins}/5")
        print(f"  Graph was needed for navigation bias, not discovery.")
        print(f"  D2-grounded framework: non-zero RHAE on click game WITHOUT banned components.")
    elif b_wins > 0 or c_wins > 0:
        print(f"PARTIAL SIGNAL: Some seeds work without graph.")
        print(f"  B (random): {b_wins}/5, C (delta): {c_wins}/5")
    else:
        print(f"SEPARABILITY FAILS: Discovery needs graph exploration bias.")
        print(f"  B (random): {b_wins}/5, C (delta): {c_wins}/5")
        if a_wins > 0:
            print(f"  A (graph control): {a_wins}/5 — graph IS needed.")

    # Discovery analysis
    print()
    print("=== Discovery Analysis ===")
    # Compare discovered cluster positions across conditions
    for cond_name, results in all_results.items():
        for r in results:
            if r['clusters']:
                # Check if any cluster is near (62,34) — VC33 L1 target
                for cx, cy, color, size in r['clusters']:
                    dist = ((cx - 62)**2 + (cy - 34)**2)**0.5
                    if dist < 5:
                        print(f"  {cond_name} s{r['seed']}: cluster at ({cx},{cy}) "
                              f"= {dist:.1f}px from L1 target (62,34)")

    print("\nStep 1023 DONE")

if __name__ == "__main__":
    main()
