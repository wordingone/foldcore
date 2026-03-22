"""
Step 575 -- FT09 pure LSH k=12 baseline. 69-action argmin.

Control experiment for cross-game navigation. No mode map, no BFS, no state
estimation. Just LSH k=12 + avgpool16 + centered_enc + argmin over 69 actions.

Step 503: k-means (n=300) + 69 actions won FT09 (action coverage mechanism).
Does LSH k=12 replicate? LSH is immediate (no warmup), finer-grained.

Action space (69):
  0-63: ACTION6 click at (gx*8+4, gy*8+4), gy,gx = divmod(id, 8)  [8x8 grid]
  64-68: ACTION1-5 (simple non-complex)

Kill:  0/5 at 50K -> FT09 requires codebook-style learning or larger action expansion.
Signal: >=3/5 -> LSH k=12 replicates k-means mechanism. Step 576: mode map on FT09.

Protocol: 5 seeds, 50K steps max, 60s per seed (5-min total).
"""
import time
import logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)

K = 12
DIM = 256        # avgpool16: 16x16 = 256
N_ACTIONS = 69   # 64 click + 5 simple
MAX_STEPS = 50_000
TIME_CAP = 60    # seconds per seed


# ── encoding ──────────────────────────────────────────────────────────────────

def encode(frame, H):
    """frame[0] is 64x64 uint8 [0-15] -> avgpool16 -> center -> k-bit hash (int)."""
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()  # 16x16 = 256
    x -= x.mean()
    bits = (H @ x > 0).astype(np.uint8)
    return int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)


# ── substrate ─────────────────────────────────────────────────────────────────

class SubLSH69:
    def __init__(self, k=K, dim=DIM, n_actions=N_ACTIONS, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.G = {}
        self._pn = self._pa = self._cn = None
        self.cells = set()
        self.n_actions = n_actions

    def observe(self, frame):
        n = encode(frame, self.H)
        self.cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values())
                  for a in range(self.n_actions)]
        min_c = min(counts)
        candidates = [a for a, c in enumerate(counts) if c == min_c]
        action_id = candidates[int(np.random.randint(len(candidates)))]
        self._pn = self._cn
        self._pa = action_id
        return action_id

    def on_reset(self):
        self._pn = None


def action_id_to_env(action_id, action_space):
    """Map 0-68 to (env_action, data)."""
    if action_id < 64:
        gy, gx = divmod(action_id, 8)
        cx, cy = gx * 8 + 4, gy * 8 + 4
        return action_space[5], {"x": cx, "y": cy}  # ACTION6 click
    else:
        return action_space[action_id - 64], {}  # ACTION1-5


# ── tests ─────────────────────────────────────────────────────────────────────

def t0():
    H = np.random.RandomState(0).randn(K, DIM).astype(np.float32)
    frame = [np.random.randint(0, 16, (64, 64), dtype=np.uint8)]
    n = encode(frame, H)
    assert isinstance(n, int)

    sub = SubLSH69(seed=0)
    sub.observe(frame)
    a = sub.act()
    assert 0 <= a < N_ACTIONS
    print("T0 PASS")


# ── experiment ────────────────────────────────────────────────────────────────

def run_seed(arc, game_id, seed):
    from arcengine import GameState
    np.random.seed(seed)

    env = arc.make(game_id)
    action_space = env.action_space
    sub = SubLSH69(k=K, dim=DIM, n_actions=N_ACTIONS, seed=seed * 1000)
    obs = env.reset()

    ts = go = lvls = 0
    level_step = None
    t0 = time.time()

    while ts < MAX_STEPS:
        if obs is None:
            obs = env.reset()
            sub.on_reset()
            continue
        if obs.state == GameState.GAME_OVER:
            go += 1
            obs = env.reset()
            sub.on_reset()
            continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset()
            sub.on_reset()
            continue

        sub.observe(obs.frame)
        action_id = sub.act()
        action, data = action_id_to_env(action_id, action_space)

        lvls_before = obs.levels_completed
        obs = env.step(action, data=data)
        ts += 1

        if obs is None:
            break
        if obs.levels_completed > lvls_before:
            lvls = obs.levels_completed
            if level_step is None:
                level_step = ts
                print(f"  s{seed} WIN@{ts} go={go}", flush=True)
        if time.time() - t0 > TIME_CAP:
            break

    elapsed = time.time() - t0
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    print(f"  s{seed}: {status:12s}  cells={len(sub.cells):4d}  go={go}  "
          f"steps={ts}  {elapsed:.0f}s", flush=True)
    return dict(seed=seed, levels=lvls, level_step=level_step,
                cells=len(sub.cells), go=go, steps=ts)


def main():
    t0()

    import arc_agi
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    ft09 = next((e for e in envs if 'ft09' in e.game_id.lower()), None)
    if ft09 is None:
        print("SKIP -- FT09 not found")
        return
    print(f"FT09: {ft09.game_id}", flush=True)

    results = []
    t_total = time.time()

    for seed in range(5):
        if time.time() - t_total > 290:
            print("TOTAL TIME CAP HIT")
            break
        r = run_seed(arc, ft09.game_id, seed)
        results.append(r)

    wins = sum(1 for r in results if r['levels'] > 0)
    avg_cells = float(np.mean([r['cells'] for r in results])) if results else 0

    print(f"\n{'='*50}")
    print(f"STEP 575: {wins}/{len(results)} wins  avg_cells={avg_cells:.0f}")
    print(f"Step 503 baseline: 3/3 wins (k-means n=300)")

    if wins == 0:
        print("U19 CONFIRMED for FT09: LSH dynamics insufficient. Need codebook-style learning.")
    elif wins >= 3:
        print(f"LSH k=12 REPLICATES k-means mechanism: {wins}/{len(results)} wins.")
        print("Step 576 direction: mode map + isolated CC on FT09.")
    else:
        print(f"PARTIAL: {wins}/{len(results)}. LSH works but less reliably than k-means.")


if __name__ == "__main__":
    main()
