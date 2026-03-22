"""
Step 654 — Hidden state MI diagnostic.

Does the graph implicitly encode the game's hidden state variables?

Run standard LSH k=12 argmin on LS20, 1 seed, 50K steps.
Record: current cell + game's hidden state variables (snw, tmx, tuv or equiv).

Compute:
  MI(cell_id, hidden_state) — does cell identity predict hidden state?
  MI(cell_visit_pattern, hidden_state) — does temporal pattern predict it?

If MI(pattern) > MI(cell), temporal structure encodes state flat cell IDs miss.
"""
import numpy as np
import sys
import time
from collections import defaultdict

K = 12
DIM = 256
N_A = 4
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MAX_STEPS = 50_001
SEED = 0


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class Recode:
    def __init__(self, k=K, dim=DIM, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.ns = 0
        self.dim = dim

    def _hash(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        x = enc_frame(frame)
        n = self._node(x)
        self.live.add(n)
        self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)
        self._px = x
        self._cn = n
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

    def on_reset(self):
        self._pn = None

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d or sum(d.values()) < 4:
            return 0.0
        v = np.array(list(d.values()), np.float64)
        p = v / v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self):
        did = 0
        for (n, a), d in list(self.G.items()):
            if n not in self.live or n in self.ref:
                continue
            if len(d) < 2 or sum(d.values()) < MIN_OBS:
                continue
            if self._h(n, a) < H_SPLIT:
                continue
            top = sorted(d, key=d.get, reverse=True)[:2]
            r0 = self.C.get((n, a, top[0]))
            r1 = self.C.get((n, a, top[1]))
            if r0 is None or r1 is None or r0[1] < 3 or r1[1] < 3:
                continue
            diff = (r0[0] / r0[1]) - (r1[0] / r1[1])
            nm = np.linalg.norm(diff)
            if nm < 1e-8:
                continue
            self.ref[n] = (diff / nm).astype(np.float32)
            self.live.discard(n)
            self.ns += 1
            did += 1
            if did >= 3:
                break


def probe_game_state(env, info):
    """Probe env and info for hidden state variables. Returns dict of available vars."""
    state = {}

    # Check info dict for any non-level keys
    if isinstance(info, dict):
        for k, v in info.items():
            if k != 'level' and isinstance(v, (int, float, np.integer, np.floating)):
                state[f'info.{k}'] = float(v)

    # Try common game object patterns
    for attr in ['game', '_game', 'game_state', '_state', 'state']:
        obj = getattr(env, attr, None)
        if obj is not None and not callable(obj):
            for var in ['snw', 'tmx', 'tuv', 'snow', 'temp', 'temperature',
                        'humidity', 'wind', 'energy', 'water', 'fire',
                        'x', 'y', 'pos', 'score', 'lives']:
                val = getattr(obj, var, None)
                if val is not None and not callable(val):
                    try:
                        state[f'{attr}.{var}'] = float(val)
                    except (TypeError, ValueError):
                        pass

    return state


def compute_mi_discrete_continuous(cell_ids, state_values, bins=10):
    """
    Compute MI(cell_id, discretized_state_value).
    cell_ids: list of hashable cell identifiers
    state_values: list of floats (same length)
    """
    if not state_values or len(set(state_values)) < 2:
        return 0.0

    # Discretize state values
    arr = np.array(state_values, dtype=np.float64)
    y_min, y_max = arr.min(), arr.max()
    if y_max == y_min:
        return 0.0
    y_disc = ((arr - y_min) / (y_max - y_min + 1e-10) * bins).astype(int).clip(0, bins - 1)

    # Map cell_ids to ints
    cell_map = {}
    x_int = []
    for c in cell_ids:
        if c not in cell_map:
            cell_map[c] = len(cell_map)
        x_int.append(cell_map[c])

    n = len(x_int)
    from collections import Counter
    xy_cnt = Counter(zip(x_int, y_disc.tolist()))
    x_cnt = Counter(x_int)
    y_cnt = Counter(y_disc.tolist())

    mi = 0.0
    for (x, y), c_xy in xy_cnt.items():
        p_xy = c_xy / n
        p_x = x_cnt[x] / n
        p_y = y_cnt[y] / n
        mi += p_xy * np.log2(p_xy / (p_x * p_y + 1e-15) + 1e-15)

    return max(0.0, mi)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    env = mk()
    sub = Recode(seed=SEED * 1000)
    obs = env.reset(seed=SEED)
    level = 0
    l1 = None
    go = 0
    t_start = time.time()

    # Data collection: (cell_id, pattern_3bit, {state_var: value})
    trajectory = []         # list of (cell, pattern, state_dict)
    recent_cells = []       # last 3 cells for pattern computation

    # Probe state on first step
    state_keys = None
    state_probed = False

    # Pattern: 3-bit recency register (bit 0 = this step, bit 1 = prev, bit 2 = 2 steps ago)
    # Computed lazily from recent_cells
    step_idx = 0

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=SEED)
            sub.on_reset()
            recent_cells = []
            continue

        cell = sub.observe(obs)

        # Compute 3-bit pattern for current cell
        pattern = 0
        for age in range(min(3, len(recent_cells))):
            if recent_cells[-(age + 1)] == cell:
                pattern |= (1 << age)

        action = sub.act()
        obs, reward, done, info = env.step(action)

        # Probe hidden state
        if not state_probed:
            state = probe_game_state(env, info)
            if state:
                state_keys = list(state.keys())
                print(f"  Hidden state vars found: {state_keys}", flush=True)
            else:
                # Report what IS accessible
                print(f"  No numeric hidden state found in env/info.", flush=True)
                print(f"  info keys: {list(info.keys()) if isinstance(info, dict) else type(info)}")
                env_attrs = [a for a in dir(env) if not a.startswith('_')]
                print(f"  env attrs (non-private): {env_attrs[:20]}")
                state_keys = []
            state_probed = True

        # Record current step
        cur_state = {}
        if state_keys:
            for attr in ['game', '_game', 'game_state', '_state', 'state']:
                obj = getattr(env, attr, None)
                if obj is not None:
                    for var in ['snw', 'tmx', 'tuv', 'snow', 'temp', 'energy']:
                        val = getattr(obj, var, None)
                        if val is not None and not callable(val):
                            try:
                                cur_state[f'{attr}.{var}'] = float(val)
                            except (TypeError, ValueError):
                                pass
            # Also from info
            if isinstance(info, dict):
                for k, v in info.items():
                    if k != 'level' and isinstance(v, (int, float, np.integer, np.floating)):
                        cur_state[f'info.{k}'] = float(v)

        trajectory.append((cell, pattern, cur_state))
        recent_cells.append(cell)
        if len(recent_cells) > 3:
            recent_cells.pop(0)

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
            level = cl
            sub.on_reset()

        if done:
            go += 1
            obs = env.reset(seed=SEED)
            sub.on_reset()
            recent_cells = []

    elapsed = time.time() - t_start
    print(f"\n  L1={l1} steps={step} t={elapsed:.1f}s", flush=True)

    print(f"\n{'='*60}")
    print(f"Trajectory length: {len(trajectory)}")
    cells = [t[0] for t in trajectory]
    patterns = [t[1] for t in trajectory]
    print(f"Unique cells: {len(set(cells))}")
    print(f"Pattern distribution: {dict(sorted({p: patterns.count(p) for p in set(patterns)}.items()))}")

    if not trajectory:
        print("No data collected.")
        return

    # If we have hidden state vars, compute MI
    all_state_keys = set()
    for _, _, s in trajectory:
        all_state_keys.update(s.keys())

    if not all_state_keys:
        print("\nNo hidden state vars accessible — MI computation skipped.")
        print("CONCLUSION: Cannot verify composition hypothesis from outside the game.")
        print("Recommend: inspect LS20 game source for accessible state variables.")
        return

    print(f"\nHidden state vars: {sorted(all_state_keys)}")
    print(f"\nMI Results (bits):")
    print(f"  {'Variable':<25} {'MI(cell,var)':>15} {'MI(pattern,var)':>17}")
    print(f"  {'-'*57}")

    for var in sorted(all_state_keys):
        var_vals = [s.get(var, 0.0) for _, _, s in trajectory]
        if len(set(var_vals)) < 2:
            print(f"  {var:<25} {'(constant)':>15} {'(constant)':>17}")
            continue
        mi_cell = compute_mi_discrete_continuous(cells, var_vals)
        mi_pat = compute_mi_discrete_continuous(patterns, var_vals)
        winner = "PATTERN>" if mi_pat > mi_cell * 1.1 else ("CELL>" if mi_cell > mi_pat * 1.1 else "≈")
        print(f"  {var:<25} {mi_cell:>15.4f} {mi_pat:>17.4f}  {winner}")

    # Summary
    higher_pattern = sum(
        1 for var in sorted(all_state_keys)
        if len(set(s.get(var, 0.0) for _, _, s in trajectory)) >= 2 and
        compute_mi_discrete_continuous(patterns, [s.get(var, 0.0) for _, _, s in trajectory]) >
        compute_mi_discrete_continuous(cells, [s.get(var, 0.0) for _, _, s in trajectory]) * 1.1
    )
    total_vars = len([v for v in all_state_keys
                      if len(set(s.get(v, 0.0) for _, _, s in trajectory)) >= 2])

    if total_vars > 0:
        if higher_pattern > 0:
            print(f"\nCOMPOSITION SIGNAL: pattern MI > cell MI for {higher_pattern}/{total_vars} vars")
        else:
            print(f"\nNO COMPOSITION SIGNAL: cell MI >= pattern MI for all {total_vars} vars")


if __name__ == "__main__":
    main()
