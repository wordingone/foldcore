"""
Step 585 — Cross-game: 581d on FT09 + VC33.

Tests whether permanent soft death penalty transfers to different games.
Same mechanism as 581d (PENALTY=100, permanent soft penalty on death edges).
Flexible encoding: adapts to per-game frame size automatically.

5 seeds per game. Compare vs argmin baseline on same game.
5-min cap per seed.

Output: SIGNAL/NEUTRAL/FAIL per game, cells comparison.
"""
import time
import numpy as np
import sys

K = 12
DIM = 256
N_A = 4
MAX_STEPS = 10_000   # 10K per runtime cap; increase to 50K with Jun approval
TIME_CAP = 60        # 5 min per seed (60s for 10K matches LS20 ratio)
N_SEEDS = 5
PENALTY = 100


# ── Flexible LSH encoding ─────────────────────────────────────────────────────
# Handles any frame size by downsampling to DIM=256 elements.

def encode(frame, H):
    arr = np.array(frame[0], dtype=np.float32).flatten()
    n = arr.size

    if n == DIM:
        x = arr / 15.0
    elif n > DIM:
        # Downsample: split into DIM chunks, average each
        # Fast path for sizes that are clean multiples
        if n % DIM == 0:
            x = arr.reshape(DIM, n // DIM).mean(axis=1) / 15.0
        else:
            # General case: use np.array_split
            chunks = np.array_split(arr, DIM)
            x = np.array([c.mean() for c in chunks]) / 15.0
    else:
        # Upsample: pad with zeros
        x = np.zeros(DIM, dtype=np.float32)
        x[:n] = arr / 15.0

    x -= x.mean()
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(K)))


# ── Soft penalty substrate (identical to 581d, flexible encode) ───────────────

class SoftPenaltySub:
    def __init__(self, lsh_seed=0):
        self.H = np.random.RandomState(lsh_seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self.death_edges = set()
        self._prev_node = None
        self._prev_action = None
        self.cells = set()
        self.total_deaths = 0

    def observe(self, frame):
        node = encode(frame, self.H)
        self.cells.add(node)
        self._curr_node = node
        if self._prev_node is not None:
            d = self.G.setdefault((self._prev_node, self._prev_action), {})
            d[node] = d.get(node, 0) + 1

    def on_death(self):
        if self._prev_node is not None:
            self.death_edges.add((self._prev_node, self._prev_action))
            self.total_deaths += 1

    def act(self):
        node = self._curr_node
        counts = np.array([sum(self.G.get((node, a), {}).values()) for a in range(N_A)],
                          dtype=np.float64)
        penalized = counts.copy()
        for a in range(N_A):
            if (node, a) in self.death_edges:
                penalized[a] += PENALTY
        action = int(np.argmin(penalized))
        self._prev_node = node
        self._prev_action = action
        return action

    def on_reset(self):
        self._prev_node = None
        self._prev_action = None


# ── Argmin baseline ───────────────────────────────────────────────────────────

class ArgminSub:
    def __init__(self, lsh_seed=0):
        self.H = np.random.RandomState(lsh_seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self._prev_node = None
        self._prev_action = None
        self.cells = set()

    def observe(self, frame):
        node = encode(frame, self.H)
        self.cells.add(node)
        self._curr_node = node

    def act(self):
        node = self._curr_node
        counts = [sum(self.G.get((node, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        if self._prev_node is not None:
            d = self.G.setdefault((self._prev_node, self._prev_action), {})
            d[node] = d.get(node, 0) + 1
        self._prev_node = node
        self._prev_action = action
        return action

    def on_reset(self):
        self._prev_node = None
        self._prev_action = None

    def on_death(self):
        pass


# ── Seed runner ───────────────────────────────────────────────────────────────

def run_seed(mk, seed, SubClass, time_cap=TIME_CAP):
    env = mk()
    sub = SubClass(lsh_seed=seed * 100 + 7)
    obs = env.reset(seed=seed)
    sub.on_reset()

    prev_cl = 0; fresh = True
    l1 = l2 = go = step = 0
    t0 = time.time()

    while step < MAX_STEPS and time.time() - t0 < time_cap:
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue

        sub.observe(obs)
        action = sub.act()
        obs, _, done, info = env.step(action)
        step += 1

        if done:
            sub.on_death()
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if fresh:
            prev_cl = cl; fresh = False
        elif cl >= 1 and prev_cl < 1:
            l1 += 1
            if l1 <= 2:
                print(f"    s{seed} L1@{step}", flush=True)
        elif cl >= 2 and prev_cl < 2:
            l2 += 1
        prev_cl = cl

    elapsed = time.time() - t0
    cells = len(sub.cells)
    deaths = getattr(sub, 'total_deaths', 0)
    print(f"  s{seed}: L1={l1} L2={l2} go={go} step={step} cells={cells} deaths={deaths} {elapsed:.0f}s",
          flush=True)
    return dict(seed=seed, l1=l1, l2=l2, go=go, steps=step, cells=cells)


def run_game(game_name, mk, label=""):
    sp_results = []
    am_results = []
    t_total = time.time()

    print(f"\n{'='*50}", flush=True)
    print(f"Game: {game_name} {label}", flush=True)
    print(f"{'='*50}", flush=True)

    print("\n-- SoftPenalty --", flush=True)
    for seed in range(N_SEEDS):
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(mk, seed, SoftPenaltySub)
        sp_results.append(r)

    print("\n-- Argmin --", flush=True)
    for seed in range(N_SEEDS):
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(mk, seed, ArgminSub)
        am_results.append(r)

    sp_l1 = sum(r['l1'] for r in sp_results)
    sp_seeds = sum(1 for r in sp_results if r['l1'] > 0)
    am_l1 = sum(r['l1'] for r in am_results)
    am_seeds = sum(1 for r in am_results if r['l1'] > 0)
    elapsed = time.time() - t_total

    print(f"\n--- {game_name} Summary ---")
    print(f"  SoftPenalty: {sp_seeds}/{N_SEEDS} seeds L1, total L1={sp_l1}")
    for r in sp_results:
        print(f"    s{r['seed']}: L1={r['l1']} cells={r['cells']}")
    print(f"  Argmin:      {am_seeds}/{N_SEEDS} seeds L1, total L1={am_l1}")
    for r in am_results:
        print(f"    s{r['seed']}: L1={r['l1']} cells={r['cells']}")

    sp_cells_avg = np.mean([r['cells'] for r in sp_results])
    am_cells_avg = np.mean([r['cells'] for r in am_results])
    print(f"  Cells avg: SP={sp_cells_avg:.0f} AM={am_cells_avg:.0f}")

    if sp_l1 > am_l1:
        verdict = f"SIGNAL: SP({sp_l1}) > AM({am_l1})"
    elif sp_l1 == am_l1:
        if sp_cells_avg > am_cells_avg * 1.05:
            verdict = f"NEUTRAL+: SP({sp_l1}) == AM({am_l1}), cells SP>{am_cells_avg:.0f}"
        else:
            verdict = f"NEUTRAL: SP({sp_l1}) == AM({am_l1})"
    else:
        verdict = f"FAIL: SP({sp_l1}) < AM({am_l1})"
    print(f"  {verdict}")
    print(f"  Elapsed: {elapsed:.0f}s")
    return dict(game=game_name, sp_l1=sp_l1, sp_seeds=sp_seeds,
                am_l1=am_l1, am_seeds=am_seeds,
                sp_cells=sp_cells_avg, am_cells=am_cells_avg,
                verdict=verdict)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 585: Cross-game transfer — FT09 + VC33", flush=True)
    print(f"  K={K} MAX_STEPS={MAX_STEPS} PENALTY={PENALTY} TIME_CAP={TIME_CAP}s/seed", flush=True)

    game_results = []

    # FT09
    try:
        mk_ft09 = lambda: arcagi3.make("FT09")
        r = run_game("FT09", mk_ft09)
        game_results.append(r)
    except Exception as e:
        print(f"FT09 failed: {e}", flush=True)

    # VC33
    try:
        mk_vc33 = lambda: arcagi3.make("VC33")
        r = run_game("VC33", mk_vc33)
        game_results.append(r)
    except Exception as e:
        print(f"VC33 failed: {e}", flush=True)

    print(f"\n{'='*60}")
    print(f"Step 585: Cross-game transfer summary")
    for r in game_results:
        print(f"  {r['game']}: {r['verdict']}")

    # Transfer verdict
    signals = sum(1 for r in game_results if r['verdict'].startswith('SIGNAL'))
    neutrals = sum(1 for r in game_results if r['verdict'].startswith('NEUTRAL'))
    fails = sum(1 for r in game_results if r['verdict'].startswith('FAIL'))
    print(f"\n  Transfer: SIGNAL={signals} NEUTRAL={neutrals} FAIL={fails}")
    if signals >= 1:
        print("  Death penalty TRANSFERS to at least one game.")
    elif fails == 0:
        print("  Death penalty NEUTRAL across all games (no harm, no boost).")
    else:
        print("  Death penalty FAILS on some games — game-specific phenomenon.")


if __name__ == "__main__":
    main()
