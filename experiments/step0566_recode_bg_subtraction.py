"""
Step 566 — Background subtraction encoding for object detection.

Running mode map per pixel -> foreground mask -> LSH on 4096D binary vector.
Background mode = static level (walls, floors, palettes).
Foreground = agent sprite + anything that moved.

Hypothesis: palettes are static -> IN background -> invisible in foreground.
L1 question: does sprite-position-based hashing support navigation?

Kill: L1 < 2/5 -> foreground hash doesn't support navigation.
5-min cap. LS20. 5 seeds. 50K steps per seed.
"""
import numpy as np
import time
import sys

N_A = 4
K = 16
FG_DIM = 4096    # 64*64 binary foreground mask
REFINE_EVERY = 2000
MIN_OBS = 4
H_SPLIT = 0.05
MODE_EVERY = 200   # recompute mode every N steps
WARMUP = 30        # frames before mode is reliable enough to use


class RecodeBackground:
    """Recode with background-subtracted foreground encoding."""

    def __init__(self, k=K, seed=0):
        self.H = np.random.RandomState(seed).randn(k, FG_DIM).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self._cn = None
        self.t = 0
        self._last_visit = {}
        # Background model
        self.freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.mode = np.zeros((64, 64), dtype=np.int32)
        self.n_frames = 0
        self._mode_dirty = False
        # Stats
        self.fg_density = []  # track avg foreground density

    def _update_bg(self, arr):
        """Update pixel frequency map."""
        r = np.arange(64)[:, None]
        c = np.arange(64)[None, :]
        self.freq[r, c, arr] += 1
        self.n_frames += 1
        self._mode_dirty = True

    def _get_mode(self):
        """Recompute mode if dirty and due."""
        if self._mode_dirty and self.n_frames % MODE_EVERY == 0:
            self.mode = np.argmax(self.freq, axis=2).astype(np.int32)
            self._mode_dirty = False
        return self.mode

    def _fg_enc(self, arr):
        """Compute foreground binary mask as float32 vector."""
        mode = self._get_mode()
        fg = (arr != mode).astype(np.float32).flatten()  # (4096,)
        return fg

    def _base(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._base(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        arr = np.array(frame[0], dtype=np.int32)
        self._update_bg(arr)

        # Use foreground encoding (only after warmup)
        if self.n_frames < WARMUP:
            x = arr.astype(np.float32).flatten() / 15.0  # fallback: raw pixels
            x = x - x.mean()
        else:
            x = self._fg_enc(arr)
            self.fg_density.append(float(x.sum()))

        n = self._node(x)
        self.live.add(n)
        self.t += 1
        self._last_visit[n] = self.t

        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k_key = (self._pn, self._pa, n)
            s, c = self.C.get(k_key, (np.zeros(FG_DIM, np.float64), 0))
            self.C[k_key] = (s + x.astype(np.float64), c + 1)
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
        # Keep freq/mode — cross-episode background model

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d or sum(d.values()) < 4:
            return 0.0
        v = np.array(list(d.values()), np.float64)
        p = v / v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self):
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
            if r0 is None or r1 is None or r0[1] < 2 or r1[1] < 2:
                continue
            diff = (r0[0] / r0[1]) - (r1[0] / r1[1])
            nm = np.linalg.norm(diff)
            if nm < 1e-8:
                continue
            self.ref[n] = (diff / nm).astype(np.float32)
            self.live.discard(n)

    def stats(self):
        return len(self.live), len(self.ref), len(self.G)

    def avg_fg_density(self):
        if not self.fg_density:
            return 0.0
        return float(np.mean(self.fg_density[-1000:]))


def t0():
    rng = np.random.RandomState(42)
    sub = RecodeBackground(seed=0)

    # Generate 50 synthetic frames (warmup period)
    frames = [[rng.randint(0, 16, (64, 64))] for _ in range(60)]
    for f in frames[:WARMUP + 5]:
        sub.observe(f)
        sub.act()

    # After warmup: mode should be set
    assert sub.n_frames >= WARMUP
    assert sub.mode is not None

    # fg_enc: identical frame to mode -> all zeros foreground
    # Force mode to known state
    sub.mode = np.zeros((64, 64), dtype=np.int32)
    test_arr = np.zeros((64, 64), dtype=np.int32)
    fg = sub._fg_enc(test_arr)
    assert fg.sum() == 0, f"All-background frame should have 0 foreground: {fg.sum()}"

    # One pixel different -> 1 foreground pixel
    test_arr2 = np.zeros((64, 64), dtype=np.int32)
    test_arr2[10, 10] = 5
    fg2 = sub._fg_enc(test_arr2)
    assert fg2.sum() == 1, f"1 different pixel -> fg=1: {fg2.sum()}"

    # on_reset preserves freq
    freq_before = sub.freq.sum()
    sub.on_reset()
    assert sub.freq.sum() == freq_before, "on_reset should keep freq"

    print("T0 PASS")


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}")
        return

    n_seeds = 5
    global_cap = 280
    R = []
    t_start = time.time()

    for seed in range(n_seeds):
        elapsed = time.time() - t_start
        if elapsed > global_cap - 10:
            print(f"\nGlobal cap hit at seed {seed}", flush=True)
            break
        seeds_left = n_seeds - seed
        budget = (global_cap - elapsed) / seeds_left
        print(f"\nseed {seed} (budget={budget:.0f}s):", flush=True)

        env = mk()
        sub = RecodeBackground(seed=seed * 1000)
        obs = env.reset(seed=seed)
        level = 0
        l1 = l2 = None
        go = 0
        deadline = time.time() + budget

        for step in range(1, 500_001):
            if obs is None:
                obs = env.reset(seed=seed)
                sub.on_reset()
                continue

            sub.observe(obs)
            action = sub.act()
            obs, reward, done, info = env.step(action)

            if done:
                go += 1
                obs = env.reset(seed=seed)
                sub.on_reset()

            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                level = cl
                sub.on_reset()
                if cl == 1 and l1 is None:
                    l1 = step
                    nc, ns, ne = sub.stats()
                    print(f"  s{seed} L1@{step} c={nc} go={go} "
                          f"fg={sub.avg_fg_density():.1f}px", flush=True)
                if cl == 2 and l2 is None:
                    l2 = step
                    nc, ns, ne = sub.stats()
                    print(f"  s{seed} L2@{step} c={nc} go={go}", flush=True)

            if step % 25_000 == 0:
                nc, ns, ne = sub.stats()
                el = time.time() - t_start
                print(f"  s{seed} @{step} c={nc} go={go} "
                      f"fg={sub.avg_fg_density():.1f}px {el:.0f}s", flush=True)

            if time.time() > deadline:
                break

        nc, ns, ne = sub.stats()
        R.append(dict(seed=seed, l1=l1, l2=l2, cells=nc, go=go,
                      steps=step, level=level,
                      fg_density=sub.avg_fg_density()))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"\nResults (background subtraction encoding):")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        print(f"  s{r['seed']}: {tag:>3}  c={r['cells']:>5}  go={r['go']:>4}  "
              f"steps={r['steps']:>7}  fg={r['fg_density']:.1f}px")

    l1n = sum(1 for r in R if r['l1'])
    l2n = sum(1 for r in R if r['l2'])
    if not R:
        print("No results.")
        return

    l1_steps = [r['l1'] for r in R if r['l1']]
    avg_l1 = np.mean(l1_steps) if l1_steps else None
    avg_fg = np.mean([r['fg_density'] for r in R])

    print(f"\nL1={l1n}/{len(R)}  L2={l2n}/{len(R)}")
    print(f"Avg foreground density: {avg_fg:.1f} pixels/frame")
    print(f"Baseline (Step 554): L1=3/3 at ~15K steps")
    if avg_l1:
        print(f"Avg L1: {avg_l1:.0f} steps")

    if l2n > 0:
        print(f"\nFIND: L2={l2n}/{len(R)}. Background subtraction reveals palettes!")
    elif l1n >= 2:
        print(f"\nL1={l1n}/{len(R)}, L2=0: Foreground supports navigation but palettes invisible.")
        print(f"Palettes are in background (static) -> enc can't see them.")
        print(f"Foreground = agent sprite only (avg {avg_fg:.0f}px). L2 needs palette detection.")
    else:
        print(f"\nKILL: L1={l1n}/{len(R)} < 2. Foreground hash doesn't support navigation.")
        print("Agent sprite alone doesn't provide enough spatial signal.")


if __name__ == "__main__":
    main()
