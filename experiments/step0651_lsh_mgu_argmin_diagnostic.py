"""
Step 651 — Passive L2 diagnostic: what does the mgu pipeline DO that argmin doesn't?

Run SubDual (from 572u) and argmin baseline (Recode) in separate envs with same seed=0.
Both record full transition logs: (step, shared_cell, action, level).
Shared encoder (k=12 avgpool) assigns comparable cell IDs to both trajectories.

Report:
(1) Cells mgu visits that argmin never reaches (mgu_exclusive)
(2) (cell, action) pairs mgu takes that argmin never tries
(3) First step mgu visits a cell argmin never visits

LS20, seed=0 only, 90s cap per pipeline (~50K steps). Under 5 min total.
"""
import numpy as np
import time
import sys

K = 12
DIM = 256
N_A = 4
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
PER_SEED_TIME = 90  # per pipeline cap
MAX_STEPS = 500_001


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def make_shared_encoder(seed=999):
    H = np.random.RandomState(seed).randn(K, DIM).astype(np.float32)

    def encode(frame):
        x = enc_frame(frame)
        return int(np.packbits(
            (H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    return encode


class Recode:
    """Argmin baseline with k=12."""

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


def run_argmin(mk, seed, encode):
    """Run argmin baseline, return log of (step, shared_cell, action, level)."""
    env = mk()
    sub = Recode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = l2 = None
    go = 0
    t_start = time.time()
    log = []  # [(step, shared_cell, action, level)]

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            continue

        sc = encode(obs)
        sub.observe(obs)
        action = sub.act()
        log.append((step, sc, action, level))
        obs, reward, done, info = env.step(action)

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
            if cl == 2 and l2 is None:
                l2 = step
            level = cl
            sub.on_reset()

        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset()

        if time.time() - t_start > PER_SEED_TIME:
            break

    print(f"  argmin: L1={l1} L2={l2} go={go} unique={len(sub.live)} "
          f"steps={step} t={time.time()-t_start:.1f}s", flush=True)
    return log, dict(l1=l1, l2=l2, go=go, unique=len(sub.live))


def run_mgu(mk, seed, encode, SubDual, fire_level_handlers):
    """Run mgu pipeline, return log of (step, shared_cell, action, level)."""
    env = mk()
    sub = SubDual(seed=seed * 1000)
    obs = env.reset(seed=seed)
    l1_step = [None]; l2_first = [None]; l3_steps = []
    go = 0
    prev_cl = 0
    t_start = time.time()
    log = []

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = -1
            go += 1
            continue

        sc = encode(obs)
        sub.observe(obs)
        action = sub.act()
        current_level = sub.game_level
        log.append((step, sc, action, current_level))
        obs, reward, done, info = env.step(action)

        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = -1
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if prev_cl == -1:
            fire_level_handlers(sub, cl, -1, step, seed, go, l1_step, l2_first, l3_steps)
            prev_cl = cl
        elif cl > prev_cl:
            fire_level_handlers(sub, cl, prev_cl, step, seed, go, l1_step, l2_first, l3_steps)
            prev_cl = cl
        else:
            prev_cl = cl

        if time.time() - t_start > PER_SEED_TIME:
            break

    print(f"  mgu: L1={l1_step[0]} L2={l2_first[0]} L3={len(l3_steps)} "
          f"go={go} steps={step} t={time.time()-t_start:.1f}s", flush=True)
    return log, dict(l1=l1_step[0], l2=l2_first[0], l3=len(l3_steps), go=go)


def compare_logs(argmin_log, mgu_log):
    """Compare argmin vs mgu trajectories."""
    argmin_cells = set(sc for _, sc, _, _ in argmin_log)
    mgu_cells = set(sc for _, sc, _, _ in mgu_log)

    mgu_exclusive = mgu_cells - argmin_cells
    argmin_exclusive = argmin_cells - mgu_cells
    common = mgu_cells & argmin_cells

    # (cell, action) pairs
    argmin_pairs = set((sc, a) for _, sc, a, _ in argmin_log)
    mgu_pairs = set((sc, a) for _, sc, a, _ in mgu_log)
    mgu_excl_pairs = mgu_pairs - argmin_pairs

    # First step mgu visits an exclusive cell
    first_exclusive_step = None
    for step, sc, _, _ in mgu_log:
        if sc in mgu_exclusive:
            first_exclusive_step = step
            break

    print(f"\n--- Comparison ---")
    print(f"  argmin cells: {len(argmin_cells)}")
    print(f"  mgu cells:    {len(mgu_cells)}")
    print(f"  common:       {len(common)}")
    print(f"  mgu_exclusive (cells argmin never reaches): {len(mgu_exclusive)}")
    print(f"  argmin_exclusive (cells mgu never reaches): {len(argmin_exclusive)}")
    print(f"  mgu_excl_transitions (cell,action) pairs mgu takes that argmin never does: {len(mgu_excl_pairs)}")
    print(f"  first step mgu visits exclusive cell: {first_exclusive_step}")

    # Action distribution comparison (global)
    argmin_adist = [0] * 4
    mgu_adist = [0] * 4
    for _, _, a, _ in argmin_log:
        argmin_adist[a] += 1
    for _, _, a, _ in mgu_log:
        mgu_adist[a] += 1
    n_a = sum(argmin_adist)
    n_m = sum(mgu_adist)
    print(f"\n  argmin action dist: {[f'{100*c/n_a:.0f}%' for c in argmin_adist]}")
    print(f"  mgu action dist:    {[f'{100*c/n_m:.0f}%' for c in mgu_adist]}")

    # Level distribution of mgu_exclusive cells (at what level does mgu reach exclusive cells?)
    excl_levels = [lv for _, sc, _, lv in mgu_log if sc in mgu_exclusive]
    if excl_levels:
        from collections import Counter
        lv_dist = Counter(excl_levels)
        print(f"\n  mgu_exclusive cells reached at levels: {dict(lv_dist)}")

    return dict(
        argmin_cells=len(argmin_cells),
        mgu_cells=len(mgu_cells),
        common=len(common),
        mgu_exclusive=len(mgu_exclusive),
        argmin_exclusive=len(argmin_exclusive),
        mgu_excl_transitions=len(mgu_excl_pairs),
        first_exclusive_step=first_exclusive_step,
    )


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    # Import SubDual from 572u
    try:
        from run_step572u_boost1 import SubDual, fire_level_handlers
        print("SubDual imported from 572u", flush=True)
    except ImportError as e:
        print(f"Cannot import SubDual: {e}")
        return

    seed = 0
    encode = make_shared_encoder(seed=999)

    print(f"\n--- Running argmin baseline (seed={seed}) ---", flush=True)
    argmin_log, argmin_stats = run_argmin(mk, seed, encode)

    print(f"\n--- Running mgu pipeline (seed={seed}) ---", flush=True)
    mgu_log, mgu_stats = run_mgu(mk, seed, encode, SubDual, fire_level_handlers)

    compare_logs(argmin_log, mgu_log)

    print(f"\n--- Level summary ---")
    print(f"  argmin: L1={argmin_stats['l1']} L2={argmin_stats['l2']}")
    print(f"  mgu:    L1={mgu_stats['l1']} L2={mgu_stats['l2']} L3={mgu_stats['l3']}")


if __name__ == "__main__":
    main()
