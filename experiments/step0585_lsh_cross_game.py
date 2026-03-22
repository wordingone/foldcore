"""
Step 585 — Cross-game: 581d soft death penalty on FT09 + VC33.

Uses arc_agi directly (same interface as steps 575/576).
FT09: 69 actions (64 click positions 8x8 grid + ACTION1-5)
VC33: 64 actions (click positions only, ACTION6)

Compare: SoftPenalty (PENALTY=100) vs ArgminSub baseline on each game.
5 seeds each, TIME_CAP=60s/seed.

Transfer test: does permanent soft death penalty help on click-based games?
"""
import time
import logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)

K = 12
DIM = 256
PENALTY = 100
TIME_CAP = 60
MAX_STEPS = 50_000

# FT09: 64 click + 5 simple = 69
# VC33: 64 click only = 64
GAME_CONFIGS = {
    'ft09': {'n_actions': 69, 'has_simple': True},
    'vc33': {'n_actions': 64, 'has_simple': False},
}


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode(frame, H):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    x -= x.mean()
    bits = (H @ x > 0).astype(np.uint8)
    return int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)


# ── Action mapping ────────────────────────────────────────────────────────────

def action_to_env(action_id, action_space, has_simple):
    """Map integer action_id to (env_action, data)."""
    if action_id < 64:
        gy, gx = divmod(action_id, 8)
        cx, cy = gx * 8 + 4, gy * 8 + 4
        # Find ACTION6 (click action) in action_space
        click_action = next((a for a in action_space if a.is_complex()), action_space[-1])
        return click_action, {"x": cx, "y": cy}
    elif has_simple:
        simple_actions = [a for a in action_space if not a.is_complex()]
        idx = action_id - 64
        if idx < len(simple_actions):
            return simple_actions[idx], {}
        return simple_actions[-1], {}
    return action_space[0], {}


# ── Soft penalty substrate ────────────────────────────────────────────────────

class SoftPenaltySub:
    def __init__(self, n_actions, lsh_seed=0):
        self.H = np.random.RandomState(lsh_seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self.death_edges = set()
        self.n_actions = n_actions
        self._pn = self._pa = self._cn = None
        self.cells = set()
        self.total_deaths = 0

    def observe(self, frame):
        n = encode(frame, self.H)
        self.cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def on_death(self):
        if self._pn is not None:
            self.death_edges.add((self._pn, self._pa))
            self.total_deaths += 1

    def act(self):
        counts = np.array([sum(self.G.get((self._cn, a), {}).values())
                           for a in range(self.n_actions)], dtype=np.float64)
        for a in range(self.n_actions):
            if (self._cn, a) in self.death_edges:
                counts[a] += PENALTY
        action_id = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action_id
        return action_id

    def on_reset(self):
        self._pn = None


class ArgminSub:
    def __init__(self, n_actions, lsh_seed=0):
        self.H = np.random.RandomState(lsh_seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self.n_actions = n_actions
        self._pn = self._pa = self._cn = None
        self.cells = set()

    def observe(self, frame):
        n = encode(frame, self.H)
        self.cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def on_death(self):
        pass

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values())
                  for a in range(self.n_actions)]
        min_c = min(counts)
        candidates = [a for a, c in enumerate(counts) if c == min_c]
        action_id = int(np.random.choice(candidates))
        self._pn = self._cn
        self._pa = action_id
        return action_id

    def on_reset(self):
        self._pn = None


# ── Seed runner ───────────────────────────────────────────────────────────────

def run_seed(arc, game_id, seed, SubClass, n_actions, has_simple):
    from arcengine import GameState
    np.random.seed(seed)

    env = arc.make(game_id)
    action_space = env.action_space
    sub = SubClass(n_actions=n_actions, lsh_seed=seed * 1000 + 7)
    obs = env.reset()
    sub.on_reset()

    ts = go = l1 = l2 = 0
    t0 = time.time()

    while ts < MAX_STEPS and time.time() - t0 < TIME_CAP:
        if obs is None:
            obs = env.reset()
            sub.on_reset()
            go += 1
            continue

        frame = obs.frame if hasattr(obs, 'frame') else obs
        if frame is None:
            obs = env.reset()
            sub.on_reset()
            go += 1
            continue

        sub.observe(frame)
        action_id = sub.act()
        env_action, data = action_to_env(action_id, action_space, has_simple)
        obs = env.step(env_action, data=data)
        ts += 1

        if obs is None:
            sub.on_death()
            obs = env.reset()
            sub.on_reset()
            go += 1
            continue

        done = obs.state in (GameState.GAME_OVER, GameState.WIN)
        lvl = obs.levels_completed if hasattr(obs, 'levels_completed') else 0
        if done:
            if lvl >= 1: l1 += 1
            if lvl >= 2: l2 += 1
            sub.on_death()
            obs = env.reset()
            sub.on_reset()
            go += 1

    elapsed = time.time() - t0
    cells = len(sub.cells)
    deaths = getattr(sub, 'total_deaths', 0)
    print(f"  s{seed}: L1={l1} L2={l2} go={go} step={ts} cells={cells} deaths={deaths} {elapsed:.0f}s",
          flush=True)
    return dict(seed=seed, l1=l1, l2=l2, go=go, steps=ts, cells=cells, deaths=deaths)


def run_game(arc, game_name, game_id, n_actions, has_simple, n_seeds=5):
    sp_results = []
    am_results = []
    t0 = time.time()

    print(f"\n{'='*50}", flush=True)
    print(f"Game: {game_name} (N_A={n_actions})", flush=True)
    print(f"{'='*50}", flush=True)

    print("\n-- SoftPenalty --", flush=True)
    for seed in range(n_seeds):
        print(f"\nseed {seed}:", flush=True)
        try:
            r = run_seed(arc, game_id, seed, SoftPenaltySub, n_actions, has_simple)
            sp_results.append(r)
        except Exception as e:
            print(f"  FAIL: {e}", flush=True)

    print("\n-- Argmin --", flush=True)
    for seed in range(n_seeds):
        print(f"\nseed {seed}:", flush=True)
        try:
            r = run_seed(arc, game_id, seed, ArgminSub, n_actions, has_simple)
            am_results.append(r)
        except Exception as e:
            print(f"  FAIL: {e}", flush=True)

    if not sp_results or not am_results:
        print(f"  INSUFFICIENT DATA for {game_name}", flush=True)
        return None

    sp_l1 = sum(r['l1'] for r in sp_results)
    sp_s = sum(1 for r in sp_results if r['l1'] > 0)
    am_l1 = sum(r['l1'] for r in am_results)
    am_s = sum(1 for r in am_results if r['l1'] > 0)
    sp_d = sum(r['deaths'] for r in sp_results)
    am_d = sum(r['deaths'] for r in am_results)
    elapsed = time.time() - t0

    print(f"\n--- {game_name} Summary ---")
    print(f"  SoftPenalty: {sp_s}/{len(sp_results)} L1={sp_l1} deaths={sp_d}")
    print(f"  Argmin:      {am_s}/{len(am_results)} L1={am_l1} deaths={am_d}")

    if sp_l1 > am_l1:
        verdict = f"SIGNAL: SP({sp_l1}) > AM({am_l1})"
    elif sp_l1 == am_l1:
        verdict = f"NEUTRAL: SP({sp_l1}) == AM({am_l1})"
    else:
        verdict = f"FAIL: SP({sp_l1}) < AM({am_l1})"
    print(f"  {verdict}  [{elapsed:.0f}s]")
    return dict(game=game_name, sp_l1=sp_l1, am_l1=am_l1, verdict=verdict, deaths_sp=sp_d)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import arc_agi
    arc = arc_agi.Arcade()

    print(f"Step 585: Cross-game transfer — FT09 + VC33", flush=True)
    print(f"  K={K} MAX_STEPS={MAX_STEPS} PENALTY={PENALTY} TIME_CAP={TIME_CAP}s/seed", flush=True)

    results = []
    for game_name, cfg in GAME_CONFIGS.items():
        try:
            r = run_game(arc, game_name.upper(), game_name,
                         cfg['n_actions'], cfg['has_simple'])
            if r:
                results.append(r)
        except Exception as e:
            print(f"{game_name} FAIL: {e}", flush=True)

    print(f"\n{'='*60}")
    print(f"Step 585: Cross-game transfer summary")
    for r in results:
        print(f"  {r['game']}: {r['verdict']} (SP_deaths={r['deaths_sp']})")

    signals = sum(1 for r in results if r['verdict'].startswith('SIGNAL'))
    neutrals = sum(1 for r in results if r['verdict'].startswith('NEUTRAL'))
    fails = sum(1 for r in results if r['verdict'].startswith('FAIL'))
    print(f"\n  SIGNAL={signals} NEUTRAL={neutrals} FAIL={fails}")
    if signals >= 1:
        print("  Death penalty TRANSFERS to click-based games.")
    elif fails == 0:
        print("  Death penalty NEUTRAL on click-based games.")
    else:
        print("  Death penalty does not transfer uniformly.")


if __name__ == "__main__":
    main()
