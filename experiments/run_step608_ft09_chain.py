"""
Step 608 -- FT09 full deterministic chain: L0 through L6 (game completion).

Builds on step 604 (FT09 L2 SOLVED: 5/5). Chain all 6 levels:
  L0 (THR):  4 clicks -> L1
  L1 (hxv):  7 clicks -> L2  (confirmed step 604)
  L2 (Fmh): 14 clicks -> L3
  L3 (oea): 16 clicks -> L4  (some walls need 2 clicks: gqb=[9,8,12])
  L4 (INW): 21 clicks -> L5  (NTi cascade: clicking NTi also cycles 4 neighbors)
  L5 (DFx): 13 clicks -> L6 (WIN)  (ZkU cascade: clicking ZkU also cycles wall above)

Solutions derived from ft09_level_solver.py + NTi cascade analysis.
Click coordinates: grid_pos * 2 (Camera width=32, scale=2 mapping).
NTi sprite pixels[j][i]==6 expands the click pattern to neighboring walls.
ZkU (tagged NTi) pixel (0,1)==6 means each click also cycles the wall above.

R3 note: FT09 = "frozen frame" per Leo's Prop 10.
All solutions are deterministic (same clicks across all seeds after reset).

Protocol: 5 seeds x 50K steps (60s/seed cap).
"""
import time
import logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)

MAX_STEPS = 50_000
TIME_CAP = 60

# Deterministic solutions per level (display pixel coords)
# Level index = levels_completed when entering that level
SOLUTIONS = {
    0: [(36,36),(36,44),(36,52),(52,44)],           # THR: 4 clicks
    1: [(20,14),(20,22),(20,30),(20,46),(28,46),(36,22),(36,30)],  # hxv: 7 clicks
    2: [(12,20),(12,28),(20,4),(20,12),(20,44),(20,52),
        (28,4),(28,20),(28,36),(28,52),(36,4),(36,52),(44,28),(44,36)],  # Fmh: 14 clicks
    3: [(20,14),(20,14),(20,30),(20,30),(20,46),(20,46),
        (28,14),(28,22),(28,30),(28,46),(28,46),(36,30),(36,46),(36,46),
        (44,14),(44,22)],                            # oea: 16 clicks (multi-click)
    4: [(30,36),(30,20),(30,52),(46,36),(14,20),(30,4),(14,36),(14,52),(46,20),
        (22,12),(22,4),(14,12),(30,12),  # NTi(11,6) + cascade fix: (11,2),(7,6),(15,6)
        (22,28),(14,28),(30,28),          # NTi(11,14) + cascade fix: (7,14),(15,14)
        (38,44),(38,36),(30,44),(46,44),(38,52)],  # NTi(19,22) + cascade fix: (19,18),(15,22),(23,22),(19,26)
                                          # INW: 21 clicks total
    5: [(4,14),(4,6),                     # ZkU(2,7)+fix(2,3)
        (12,22),(12,30),                  # ZkU(6,11),(6,15)
        (20,14),(20,22),(20,38),          # ZkU(10,7),(10,11),(10,19)
        (28,30),                          # ZkU(14,15)
        (36,14),(36,30),                  # ZkU(18,7),(18,15)
        (44,30),(44,38),                  # ZkU(22,15),(22,19)
        (52,38)],                         # ZkU(26,19): DFx 13 clicks total
}


def t0():
    """Verify solution data is self-consistent."""
    total_clicks = sum(len(v) for v in SOLUTIONS.values())
    assert total_clicks == 4 + 7 + 14 + 16 + 21 + 13 == 75, f"Expected 75, got {total_clicks}"
    print("T0 PASS", flush=True)


class SubFT09Chain:
    def __init__(self):
        self._queue = []        # pending click coords
        self._level = 0         # current level (= levels_completed at entry)
        self._loaded_up_to = -1  # which levels we've queued
        self._argmin_G = {}     # fallback argmin graph
        self._pn = self._pa = self._cn = None
        self.H = np.random.RandomState(42).randn(12, 256).astype(np.float32)

    def _encode(self, frame):
        arr = np.array(frame[0], dtype=np.float32) / 15.0
        x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
        x -= x.mean()
        bits = (self.H @ x > 0).astype(np.uint8)
        return int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)

    def on_level_up(self, new_lvl):
        self._level = new_lvl
        # Load the solution for the new level
        if new_lvl in SOLUTIONS and new_lvl > self._loaded_up_to:
            self._queue = list(SOLUTIONS[new_lvl])
            self._loaded_up_to = new_lvl

    def on_start(self):
        """Called once at episode start (level 0 = THR)."""
        self._level = 0
        self._queue = list(SOLUTIONS[0])
        self._loaded_up_to = 0
        self._pn = None

    def observe(self, frame):
        n = self._encode(frame)
        if self._pn is not None:
            d = self._argmin_G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def act(self):
        if self._queue:
            cx, cy = self._queue.pop(0)
            self._pa = None  # non-argmin action
            return cx, cy
        # Fallback: argmin
        counts = [sum(self._argmin_G.get((self._cn, a), {}).values())
                  for a in range(64)]
        a = int(np.argmin(counts))
        cx = (a % 8) * 8 + 4
        cy = (a // 8) * 8 + 4
        self._pn = self._cn
        self._pa = a
        return cx, cy

    def on_reset(self):
        self._queue = []
        self._level = 0
        self._loaded_up_to = -1
        self._pn = None


def run_seed(arc, game_id, seed):
    from arcengine import GameState
    np.random.seed(seed)

    env = arc.make(game_id)
    action6 = env.action_space[0]
    sub = SubFT09Chain()
    sub.on_start()
    obs = env.reset()

    ts = go = 0
    prev_lvls = 0
    level_steps = {}
    t_start = time.time()
    last_queue_note = -1

    while ts < MAX_STEPS:
        if obs is None:
            obs = env.reset(); sub.on_reset(); sub.on_start(); prev_lvls = 0; go += 1; continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); sub.on_reset(); sub.on_start(); prev_lvls = 0; continue
        if obs.state == GameState.WIN:
            lvl = obs.levels_completed
            if lvl not in level_steps:
                level_steps[lvl] = ts
                print(f"  s{seed} WIN@{ts} L{lvl}!!", flush=True)
            break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset(); sub.on_reset(); sub.on_start(); prev_lvls = 0; continue

        sub.observe(obs.frame)
        cx, cy = sub.act()

        lvls_before = obs.levels_completed
        obs = env.step(action6, data={"x": cx, "y": cy})
        ts += 1

        if obs is None:
            break

        if obs.levels_completed > lvls_before:
            new_lvl = obs.levels_completed
            sub.on_level_up(new_lvl)
            if new_lvl not in level_steps:
                level_steps[new_lvl] = ts
                print(f"  s{seed} L{new_lvl}@{ts} queue={len(sub._queue)}", flush=True)
            prev_lvls = new_lvl

        if time.time() - t_start > TIME_CAP:
            print(f"  s{seed} cap@{ts} go={go} lvls_reached={sorted(level_steps.keys())} "
                  f"queue={len(sub._queue)}", flush=True)
            break

    max_lvl = max(level_steps.keys()) if level_steps else 0
    status = f"L{max_lvl}@{level_steps[max_lvl]}" if level_steps else "---"
    print(f"  s{seed}: {status}  go={go}  levels={sorted(level_steps.keys())}",
          flush=True)
    return dict(seed=seed, level_steps=level_steps, go=go, ts=ts,
                max_lvl=max_lvl)


def main():
    t0()

    import arc_agi
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    ft09 = next((e for e in envs if 'ft09' in e.game_id.lower()), None)
    if ft09 is None:
        print("SKIP -- FT09 not found"); return

    print(f"Step 608: FT09 full chain (L0 to L6)", flush=True)
    print(f"  game={ft09.game_id}  6 levels  75 total clicks", flush=True)
    print(f"  Solutions from cgj() offline analysis", flush=True)

    results = []
    t_total = time.time()

    for seed in range(5):
        if time.time() - t_total > 295:
            print("TOTAL TIME CAP"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(arc, ft09.game_id, seed)
        results.append(r)

    print(f"\n{'='*60}", flush=True)
    print(f"Step 608: FT09 chain (L0 to L6)", flush=True)

    for lvl in range(7):
        wins = sum(1 for r in results if r['max_lvl'] >= lvl)
        step_avg = np.mean([r['level_steps'][lvl] for r in results
                           if lvl in r['level_steps']]) if any(lvl in r['level_steps'] for r in results) else None
        avg_str = f"avg@{step_avg:.0f}" if step_avg else "---"
        print(f"  L{lvl}: {wins}/{len(results)}  {avg_str}", flush=True)

    max_wins = sum(1 for r in results if r['max_lvl'] >= 6)
    if max_wins >= 3:
        print("  SIGNAL: FT09 complete chain works. Game solvable.", flush=True)
    elif max_wins > 0:
        print(f"  PARTIAL: {max_wins}/5 complete. Remaining levels need analysis.", flush=True)
    else:
        print("  L3 not reached. Solution has errors.", flush=True)


if __name__ == "__main__":
    main()
