"""
Step 578 -- U5 validation: proportional action selection in Recode on LS20.

U5 (provisional): "Sparse selection over global aggregation."
Current evidence: argmin (hard, sparse) works. Soft blending might not.

Test: Recode k=16 on LS20, softmax(1/count) instead of hard argmin.
Softmax temperature inversely proportional to count: low-count actions get HIGH
probability (same direction as argmin, but probabilistic not deterministic).

Prediction (U8-linked hypothesis): soft selection works if temperature is high
(approaches argmin). Fails if too soft (U8: hard selection over soft blending).

Protocol: 3 seeds x 50K steps, 5-min total cap.
Compare to argmin baseline: 3/3 L1 at 50K (step 546 clean).

If softmax 3/3: U5 CHALLENGED (soft proportional works)
If softmax 0/3: U5 + U8 CONFIRMED (soft kills navigation)
"""
import time
import numpy as np
import sys

K = 16
DIM = 256
N_A = 4
TEMP = 1.0         # softmax temperature on action counts

REFINE_EVERY = 5000
MIN_OBS = 8
MODE_EVERY = 200
WARMUP = 100
MIN_CLUSTER = 2
MAX_CLUSTER = 60
VISIT_DIST = 4

from scipy.ndimage import label as ndlabel


def enc_ls20(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    return a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def find_isolated_clusters(mode_arr):
    clusters = []
    for color in range(16):
        mask = (mode_arr == color)
        if not mask.any():
            continue
        labeled, n = ndlabel(mask)
        for cid in range(1, n + 1):
            region = (labeled == cid)
            sz = int(region.sum())
            if MIN_CLUSTER <= sz <= MAX_CLUSTER:
                ys, xs = np.where(region)
                clusters.append({'cy': float(ys.mean()), 'cx': float(xs.mean()),
                                 'color': int(color), 'size': sz})
    return clusters


def dir_action(ty, tx, ay, ax):
    dy, dx = ty - ay, tx - ax
    if abs(dy) >= abs(dx):
        return 0 if dy < 0 else 1
    return 2 if dx < 0 else 3


class RecodeU5:
    """Recode with softmax(1/count) action selection (U5 test)."""

    def __init__(self, k=K, dim=DIM, seed=0, temp=TEMP):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = self._cn = None
        self.t = 0
        self.ns = 0
        self.dim = dim
        self.temp = temp
        self._mu = np.zeros(dim, dtype=np.float32)
        self._mu_n = 0
        # Mode map
        self.freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.mode = np.zeros((64, 64), dtype=np.int32)
        self.n_frames = 0
        self.targets = []
        self._steps_since_detect = 99999
        self.agent_yx = None
        self.prev_arr = None
        self.visited = []

    def _base(self, x):
        return int(np.packbits((self.H @ x > 0).astype(np.uint8),
                               bitorder='big').tobytes().hex(), 16)

    def _node(self, x):
        n = self._base(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        arr = np.array(frame[0], dtype=np.int32)
        r = np.arange(64)[:, None]; c = np.arange(64)[None, :]
        self.freq[r, c, arr] += 1; self.n_frames += 1
        if self.n_frames % MODE_EVERY == 0:
            self.mode = np.argmax(self.freq, axis=2).astype(np.int32)
        if self.prev_arr is not None:
            diff = np.abs(arr - self.prev_arr) > 0
            nc = int(diff.sum())
            if 1 <= nc < 200:
                ys, xs = np.where(diff)
                self.agent_yx = (float(ys.mean()), float(xs.mean()))
        self.prev_arr = arr.copy()
        x_raw = enc_ls20(frame)
        x = x_raw - self._mu
        self._mu_n += 1
        self._mu += (x_raw - self._mu) / self._mu_n
        n = self._node(x)
        self.live.add(n); self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c2 = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c2 + 1)
        self._px = x; self._cn = n
        if self.t % REFINE_EVERY == 0:
            self._refine()
        self._steps_since_detect += 1

    def act(self):
        if self._steps_since_detect >= 500 and self.n_frames >= WARMUP:
            self.targets = find_isolated_clusters(self.mode)
            self._steps_since_detect = 0
        if self.targets and self.agent_yx is not None:
            ay, ax = self.agent_yx
            best = None; best_d = 1e9
            for t in self.targets:
                if any(((t['cy']-vy)**2 + (t['cx']-vx)**2) < VISIT_DIST**2
                       for vy, vx in self.visited):
                    continue
                d = ((t['cy']-ay)**2 + (t['cx']-ax)**2)**0.5
                if d < best_d: best_d = d; best = t
            if best is not None:
                if best_d < VISIT_DIST:
                    self.visited.append((best['cy'], best['cx']))
                else:
                    a = dir_action(best['cy'], best['cx'], ay, ax)
                    self._pn = self._cn; self._pa = a; return a
        # Softmax over inverse counts (high probability for low-count actions)
        counts = np.array([sum(self.G.get((self._cn, a), {}).values())
                           for a in range(N_A)], dtype=np.float32)
        # Invert: score = -count. Softmax with temperature.
        scores = -counts / self.temp
        scores -= scores.max()  # numerical stability
        probs = np.exp(scores)
        probs /= probs.sum()
        a = int(np.random.choice(N_A, p=probs))
        self._pn = self._cn; self._pa = a; return a

    def _refine(self):
        for (pn, pa, n), (s, c) in list(self.C.items()):
            if c < MIN_OBS or n in self.ref:
                continue
            d = self.G.get((pn, pa), {})
            if len(d) < 2:
                continue
            mu = (s / c).astype(np.float32)
            v = mu - mu.mean()
            norm = np.linalg.norm(v)
            if norm < 1e-8:
                continue
            self.ref[n] = (v / norm).astype(np.float32)
            self.ns += 1

    def on_reset(self):
        self._pn = None; self.agent_yx = None
        self.prev_arr = None; self.visited = []
        self._steps_since_detect = 99999
        self._mu = np.zeros(self.dim, dtype=np.float32); self._mu_n = 0


# ── tests ─────────────────────────────────────────────────────────────────────

def t0():
    sub = RecodeU5(seed=0, temp=1.0)
    frame = [np.random.RandomState(1).randint(0, 16, (64, 64), dtype=np.uint8)]
    sub.observe(frame)
    a = sub.act()
    assert 0 <= a < 4
    # Softmax probabilities: with only one cell (no G entries), all actions have
    # count=0 so scores = [0,0,0,0]/1 -> all equal -> uniform distribution
    counts = np.zeros(4, dtype=np.float32)
    scores = -counts / 1.0
    scores -= scores.max()
    probs = np.exp(scores)
    probs /= probs.sum()
    assert abs(probs[0] - 0.25) < 0.01
    print("T0 PASS")


# ── experiment ────────────────────────────────────────────────────────────────

def run_seed(mk, seed, time_cap=80):
    env = mk()
    sub = RecodeU5(seed=seed * 1000, temp=TEMP)
    obs = env.reset(seed=seed)
    l1 = l2 = go = 0
    prev_cl = 0; t_start = time.time()
    step = 0

    while time.time() - t_start < time_cap:
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset(); prev_cl = 0; go += 1; continue

        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)
        step += 1

        if done:
            go += 1; obs = env.reset(seed=seed); sub.on_reset(); prev_cl = 0; continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl >= 1 and prev_cl < 1:
            l1 += 1
            print(f"  s{seed} L1@{step} go={go}", flush=True)
        if cl >= 2 and prev_cl < 2:
            l2 += 1
        prev_cl = cl

    cells, edges = len(sub.live), len(sub.G)
    elapsed = time.time() - t_start
    print(f"  s{seed}: L1={l1} L2={l2} go={go} cells={cells} edges={edges} "
          f"splits={sub.ns} steps={step} {elapsed:.0f}s", flush=True)
    return dict(seed=seed, l1=l1, l2=l2, steps=step, cells=cells, splits=sub.ns)


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    results = []
    t_total = time.time()
    for seed in range(3):
        if time.time() - t_total > 280:
            print("TOTAL TIME CAP HIT"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(mk, seed, time_cap=90)
        results.append(r)

    any_l1 = sum(1 for r in results if r['l1'] > 0)
    avg_cells = float(np.mean([r['cells'] for r in results])) if results else 0

    print(f"\n{'='*50}")
    for r in results:
        print(f"  s{r['seed']}: L1={r['l1']} cells={r['cells']} splits={r['splits']}")
    print(f"\nSTEP 578: {any_l1}/{len(results)} seeds reach L1  avg_cells={avg_cells:.0f}")
    print(f"Baseline (step 546 argmin): 2/3 L1 on chain, 5/5 clean")
    print(f"Softmax temp={TEMP}")

    if any_l1 == 0:
        print("U5+U8 CONFIRMED: Soft selection fails. Hard argmin is required.")
    elif any_l1 >= 2:
        print(f"U5 CHALLENGED: Softmax({TEMP}) navigates {any_l1}/{len(results)}.")
    else:
        print(f"PARTIAL: {any_l1}/{len(results)}. Softmax degrades but doesn't kill.")


if __name__ == "__main__":
    main()
