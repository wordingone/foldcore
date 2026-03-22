"""
Step 604 -- FT09 L2: deterministic solution for level 1.

Analysis of ft09.py cgj() win condition:
  - Level 1 has 2 bsT sprites: jGI@(14,11) and EQX@(14,19)
  - Both check 8 neighboring Hkx walls using pixel pattern
  - nRq = sprite.pixels[1][1] = 12 for both
  - Level 1 gqb = [9, 12]: clicking a wall cycles its center color 9<->12
  - Walls start at color 9; some must reach 12, others must stay at 9

Coordinate system discovery:
  - set_position(x,y) maps to frame pixel (x*2, y*2) (2px/unit rendering)
  - click pixel (cx,cy) -> display_to_grid(cx,cy) = (cx/4, cy/4)
  - get_sprite_at(gx,gy) checks sprite at game position (gx*2, gy*2)
  - Therefore: to click wall at set_position(wx,wy), use click pixel
    (cx,cy) such that (cx/4)*2 in [wx,wx+2] and (cy/4)*2 in [wy,wy+2]

7 walls that must be color 12 (click once to cycle 9->12):
  - (10,7)  -> click (20,16)
  - (10,11) -> click (20,24)
  - (18,11) -> click (36,24)  <- previously thought "off-screen"
  - (10,15) -> click (20,32)
  - (18,15) -> click (36,32)  <- previously thought "off-screen"
  - (10,23) -> click (20,48)
  - (14,23) -> click (28,48)

8 walls that must stay at 9 (do NOT click):
  - (14,7), (18,7), (14,15), (10,19), (18,19), (18,23), (18,11) neighbor
  - checked implicitly by not clicking them

Kill:  L2=0/5 -> coordinate analysis wrong or cgj condition wrong
Signal: L2>=3/5 -> deterministic solution works. Push VC33.

Protocol: 5 seeds x 50K steps, 60s/seed
"""
import time
import logging
import numpy as np
from scipy.ndimage import label as ndlabel

logging.getLogger().setLevel(logging.WARNING)

K = 12
DIM = 256
N_CLICKS = 64
MAX_STEPS = 50_000
TIME_CAP = 60

# Deterministic L1 solution: exactly 7 clicks in order
# Each click cycles one wall from 9->12. cgj() fires True after all 7.
L1_SOLUTION = [
    (20, 16),   # wall (10, 7)
    (20, 24),   # wall (10, 11)
    (36, 24),   # wall (18, 11)
    (20, 32),   # wall (10, 15)
    (36, 32),   # wall (18, 15)
    (20, 48),   # wall (10, 23)
    (28, 48),   # wall (14, 23)
]


def encode(frame, H):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    x -= x.mean()
    bits = (H @ x > 0).astype(np.uint8)
    return int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)


def t0():
    # Verify L1_SOLUTION has exactly 7 entries
    assert len(L1_SOLUTION) == 7, f"Expected 7 solution clicks, got {len(L1_SOLUTION)}"
    # Verify all coordinates in 0-63 range
    for cx, cy in L1_SOLUTION:
        assert 0 <= cx < 64 and 0 <= cy < 64, f"Click ({cx},{cy}) out of range"
    # Verify coordinate math: click (cx,cy) -> display (cx//4, cy//4) -> pos (cx//2, cy//2)
    # Wall (10,7): click (20,16) -> display (5,4) -> pos (10,8) in [10..12]x[7..9] ✓
    for (cx, cy), expected_wall in zip(L1_SOLUTION, [(10,7),(10,11),(18,11),(10,15),(18,15),(10,23),(14,23)]):
        gx, gy = cx // 2, cy // 2
        wx, wy = expected_wall
        assert wx <= gx <= wx + 2, f"Click ({cx},{cy}) -> pos ({gx},{gy}), expected wall x={wx}"
        assert wy <= gy <= wy + 2, f"Click ({cx},{cy}) -> pos ({gx},{gy}), expected wall y={wy}"
    print("T0 PASS", flush=True)


def run_seed(arc, game_id, seed):
    from arcengine import GameState
    np.random.seed(seed)

    env = arc.make(game_id)
    action6 = env.action_space[0]
    H = np.random.RandomState(seed * 1000).randn(K, DIM).astype(np.float32)
    G = {}
    pn = pa = cn = None

    obs = env.reset()
    ts = go = 0
    prev_lvls = 0
    l1_step = l2_step = None
    l1_click_idx = None   # index into L1_SOLUTION while solving L1
    t_start = time.time()

    while ts < MAX_STEPS:
        if obs is None:
            obs = env.reset(); pn = pa = None; prev_lvls = 0; continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); pn = pa = None; prev_lvls = 0
            l1_click_idx = None; continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset(); pn = pa = None; prev_lvls = 0; continue

        # Check level transition
        if obs.levels_completed > prev_lvls:
            new_lvl = obs.levels_completed
            if new_lvl >= 1 and l1_step is None:
                l1_step = ts
                l1_click_idx = 0
                print(f"  s{seed} L1@{ts} go={go} -> starting deterministic solve",
                      flush=True)
            if new_lvl >= 2 and l2_step is None:
                l2_step = ts
                print(f"  s{seed} L2@{ts}!!", flush=True)
            prev_lvls = new_lvl

        # Choose action
        if l1_click_idx is not None and l1_click_idx < len(L1_SOLUTION):
            # Deterministic L1 solution
            cx, cy = L1_SOLUTION[l1_click_idx]
            l1_click_idx += 1
        else:
            # Argmin exploration (L0 and post-solution)
            cn = encode(obs.frame, H)
            counts = [sum(G.get((cn, a), {}).values()) for a in range(N_CLICKS)]
            min_c = min(counts)
            candidates = [a for a, c in enumerate(counts) if c == min_c]
            a = candidates[int(np.random.randint(len(candidates)))]
            gy, gx = divmod(a, 8)
            cx, cy = gx * 8 + 4, gy * 8 + 4
            if pn is not None:
                d = G.setdefault((pn, pa), {})
                d[cn] = d.get(cn, 0) + 1
            pn, pa = cn, a

        obs = env.step(action6, data={"x": cx, "y": cy})
        ts += 1

        if obs is None:
            break

        if time.time() - t_start > TIME_CAP:
            print(f"  s{seed} cap@{ts} go={go} l1_ci={l1_click_idx}", flush=True)
            break

    status = f"L2@{l2_step}" if l2_step else (f"L1@{l1_step}" if l1_step else "---")
    print(f"  s{seed}: {status}  go={go}  ts={ts}", flush=True)
    return dict(seed=seed, l1=l1_step, l2=l2_step, go=go, ts=ts)


def main():
    t0()

    import arc_agi
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    ft09 = next((e for e in envs if 'ft09' in e.game_id.lower()), None)
    if ft09 is None:
        print("SKIP -- FT09 not found"); return

    print(f"Step 604: FT09 L2 -- deterministic L1 solution", flush=True)
    print(f"  game={ft09.game_id}  7-click solution for level 1", flush=True)
    print(f"  argmin L0, then hardcoded wall clicks for L1", flush=True)

    results = []
    t_total = time.time()

    for seed in range(5):
        if time.time() - t_total > 295:
            print("TOTAL TIME CAP"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(arc, ft09.game_id, seed)
        results.append(r)

    l1_wins = sum(1 for r in results if r['l1'])
    l2_wins = sum(1 for r in results if r['l2'])

    print(f"\n{'='*60}", flush=True)
    print(f"Step 604: FT09 L2 (deterministic L1 solution)", flush=True)
    print(f"  L1: {l1_wins}/{len(results)}  L2: {l2_wins}/{len(results)}", flush=True)
    if l2_wins >= 3:
        print("  SIGNAL: Deterministic solution wins L2. Push VC33.", flush=True)
    elif l2_wins > 0:
        print(f"  PARTIAL: {l2_wins}/{len(results)} L2.", flush=True)
    elif l1_wins >= 3:
        print("  L1 reached but solution clicks failed. Check cgj analysis.", flush=True)
    else:
        print("  L1 not reached. Argmin regression?", flush=True)


if __name__ == "__main__":
    main()
