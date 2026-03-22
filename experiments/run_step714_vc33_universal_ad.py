"""
Step 714 — Universal Action Discovery on VC33, raw 64x64.

R3 hypothesis: action pruning discovers VC33's magic click positions autonomously.
VC33 magic pixels: (62,26) and (62,34).
8x8 grid centers: (gx*8+4, gy*8+4). Magic cells: (7,3)→(60,28), (7,4)→(60,36).
Action indices in GRID_ACTIONS: these map to UNIVERSAL_ACTIONS[4+3*8+7]=UNIVERSAL_ACTIONS[35]
and UNIVERSAL_ACTIONS[4+4*8+7]=UNIVERSAL_ACTIONS[43] (gx=7, gy=3 and gx=7, gy=4).

If VC33 responds to clicks at (60,28) and (60,36), those actions should be LIVE.
If 0/5 bootloader pass AND live_click=0 → VC33 needs zone discovery beyond click detection.
If 0/5 bootloader pass BUT live_click>0 → clicks detected but zone navigation still broken.

Kill criterion: 0/5 bootloader pass.
"""
import numpy as np
import time
import sys

K_NAV = 12
K_FINE = 20
DIM = 4096   # raw 64x64
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MIN_VISITS_ALIAS = 3
WARMUP_STEPS = 500
DEAD_THRESHOLD = 5
MAX_STEPS = 120_001
N_SEEDS = 5

DIR_ACTIONS = [0, 1, 2, 3]
GRID_ACTIONS = [(gx * 8 + 4) + (gy * 8 + 4) * 64
                for gy in range(8) for gx in range(8)]
UNIVERSAL_ACTIONS = DIR_ACTIONS + GRID_ACTIONS
N_UNIV = len(UNIVERSAL_ACTIONS)  # 68

# Magic click grid indices for VC33 (gx=7, gy=3) and (gx=7, gy=4)
# GRID_ACTIONS index = gy*8 + gx
MAGIC_GRID_IDX_A = 3 * 8 + 7  # gy=3, gx=7 → (60,28) — near (62,26)
MAGIC_GRID_IDX_B = 4 * 8 + 7  # gy=4, gx=7 → (60,36) — near (62,34)
MAGIC_UNIV_IDX_A = 4 + MAGIC_GRID_IDX_A
MAGIC_UNIV_IDX_B = 4 + MAGIC_GRID_IDX_B


def enc_raw(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    return a.flatten()  # 64x64 = 4096D


def obs_changed(frame_before, frame_after, tol=0.5):
    try:
        a = np.array(frame_before[0], dtype=np.float32)
        b = np.array(frame_after[0], dtype=np.float32)
        return not np.allclose(a, b, atol=tol)
    except Exception:
        return True


class UniversalAD_Raw:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self.ref = {}; self.G = {}; self.C = {}; self.live_nodes = set()
        self.G_fine = {}; self.aliased = set()
        self._pn = self._pa = self._px = None; self._pfn = None
        self.t = 0; self.dim = DIM; self._cn = None; self._fn = None
        self._mu = np.zeros(DIM, dtype=np.float32); self._mu_n = 0
        self.live_actions = set(range(N_UNIV))
        self.dead_actions = set()
        self.probe_count = [0] * N_UNIV
        self.change_count = [0] * N_UNIV
        self._probe_ptr = 0
        self.steps = 0

    def _hash_nav(self, x):
        return int(np.packbits((self.H_nav @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def _hash_fine(self, x):
        return int(np.packbits((self.H_fine @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash_nav(x)
        while n in self.ref: n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        x_raw = enc_raw(frame)
        self._mu_n += 1
        self._mu = self._mu + (x_raw - self._mu) / self._mu_n
        x = x_raw - self._mu
        n = self._node(x); fn = self._hash_fine(x)
        self.live_nodes.add(n); self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)
            succ = self.G.get((self._pn, self._pa), {})
            if sum(succ.values()) >= MIN_VISITS_ALIAS and len(succ) >= 2:
                self.aliased.add(self._pn)
            if self._pn in self.aliased and self._pfn is not None:
                d2 = self.G_fine.setdefault((self._pfn, self._pa), {})
                d2[fn] = d2.get(fn, 0) + 1
        self._px = x; self._cn = n; self._fn = fn
        if self.t % REFINE_EVERY == 0: self._refine()
        return n

    def act(self):
        self.steps += 1
        live_sorted = sorted(self.live_actions) or [4]
        if self.steps <= WARMUP_STEPS:
            idx = live_sorted[self._probe_ptr % len(live_sorted)]
            self._probe_ptr += 1
        else:
            if self._cn in self.aliased and self._fn is not None:
                idx = min(live_sorted, key=lambda a: sum(self.G_fine.get((self._fn, a), {}).values()))
            else:
                idx = min(live_sorted, key=lambda a: sum(self.G.get((self._cn, a), {}).values()))
        self._pn = self._cn; self._pfn = self._fn; self._pa = idx
        return idx

    def report_change(self, action_idx, changed):
        self.probe_count[action_idx] += 1
        if changed:
            self.change_count[action_idx] += 1
        if (self.probe_count[action_idx] >= DEAD_THRESHOLD
                and self.change_count[action_idx] == 0):
            self.live_actions.discard(action_idx)
            self.dead_actions.add(action_idx)

    def on_reset(self):
        self._pn = None; self._pfn = None
        self._mu = np.zeros(self.dim, dtype=np.float32); self._mu_n = 0

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d or sum(d.values()) < 4: return 0.0
        v = np.array(list(d.values()), np.float64); p = v/v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self):
        did = 0
        for (n, a), d in list(self.G.items()):
            if n not in self.live_nodes or n in self.ref: continue
            if len(d) < 2 or sum(d.values()) < MIN_OBS: continue
            if self._h(n, a) < H_SPLIT: continue
            top = sorted(d, key=d.get, reverse=True)[:2]
            r0 = self.C.get((n, a, top[0])); r1 = self.C.get((n, a, top[1]))
            if r0 is None or r1 is None or r0[1] < 3 or r1[1] < 3: continue
            diff = (r0[0]/r0[1]) - (r1[0]/r1[1]); nm = np.linalg.norm(diff)
            if nm < 1e-8: continue
            self.ref[n] = (diff/nm).astype(np.float32); self.live_nodes.discard(n); did += 1
            if did >= 3: break


def run(seed, make):
    env = make()
    sub = UniversalAD_Raw(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0; l1 = None
    t_start = time.time()

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset(); continue
        sub.observe(obs)
        action_idx = sub.act()
        action_int = UNIVERSAL_ACTIONS[action_idx]
        try:
            obs_new, reward, done, info = env.step(action_int)
            changed = obs_changed(obs, obs_new) if obs_new is not None else False
        except Exception:
            sub.live_actions.discard(action_idx)
            sub.dead_actions.add(action_idx)
            obs_new = obs; done = False; info = {}; changed = False
        sub.report_change(action_idx, changed)
        obs = obs_new
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None: l1 = step
            level = cl; sub.on_reset()
        if done:
            obs = env.reset(seed=seed); sub.on_reset()

    elapsed = time.time() - t_start
    live_dir = len([a for a in sub.live_actions if a < 4])
    live_click = len([a for a in sub.live_actions if a >= 4])
    magic_a_live = MAGIC_UNIV_IDX_A in sub.live_actions
    magic_b_live = MAGIC_UNIV_IDX_B in sub.live_actions
    bootloader = "PASS" if l1 else "FAIL"
    print(f"  s{seed:2d}: {bootloader} live={len(sub.live_actions)}(dir={live_dir},click={live_click}) "
          f"magic=({'Y' if magic_a_live else 'N'},{'Y' if magic_b_live else 'N'}) "
          f"dead={len(sub.dead_actions)} aliased={len(sub.aliased)} t={elapsed:.1f}s", flush=True)
    return dict(seed=seed, l1=l1, live_dir=live_dir, live_click=live_click,
                dead=len(sub.dead_actions), aliased=len(sub.aliased),
                magic_a=magic_a_live, magic_b=magic_b_live)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("VC33")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    t_start = time.time()
    print(f"Step 714: Universal AD on VC33, {N_SEEDS} seeds, {MAX_STEPS-1} steps")
    print(f"Encoding: raw 64x64 = {DIM}D. R3: autonomous zone discovery via click detection.")
    print(f"Magic click indices: UNIV[{MAGIC_UNIV_IDX_A}]->({UNIVERSAL_ACTIONS[MAGIC_UNIV_IDX_A]}), "
          f"UNIV[{MAGIC_UNIV_IDX_B}]->({UNIVERSAL_ACTIONS[MAGIC_UNIV_IDX_B]})")
    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    boot_n = sum(1 for r in results if r['l1'])
    elapsed = time.time() - t_start
    print(f"Bootloader: {boot_n}/{N_SEEDS}  total_time={elapsed:.1f}s")
    for r in results:
        status = "PASS" if r['l1'] else "FAIL"
        print(f"  s{r['seed']:2d}: {status} live_dir={r['live_dir']} live_click={r['live_click']} "
              f"magic=({'Y' if r['magic_a'] else 'N'},{'Y' if r['magic_b'] else 'N'}) dead={r['dead']}")
    magic_detected = sum(1 for r in results if r['magic_a'] or r['magic_b'])
    print(f"R3 result: magic clicks detected in {magic_detected}/{N_SEEDS} seeds")
    if boot_n >= 3:
        print(f"PASS: VC33 solvable with universal AD — proceed to chain (step 715)")
    elif magic_detected >= 3:
        print(f"PARTIAL: Magic clicks detected but bootloader fails — zone nav broken, not click detection")
    else:
        print(f"KILL: Neither magic clicks detected nor bootloader passes — try 16x16 grid (714b)")


if __name__ == "__main__":
    main()
