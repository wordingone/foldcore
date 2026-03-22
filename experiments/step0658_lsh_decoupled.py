"""
Step 658 — Does the interpreter decompose? Decoupled compare vs select.

Normal:    hash(frame) -> cell. argmin over cell's edges. increment.
           compare and select use SAME cell.

Decoupled: hash(frame) -> cell_compare (for STORING/incrementing)
           hash(delta_frame = frame XOR prev_frame) -> cell_select (for SELECTING)
           argmin over cell_select's edges.

If >= 3/10 L1: compare and select are genuinely independent.
If 0/10 L1: compare and select must be coupled (interpreter is ONE operation).
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
MAX_STEPS = 500_001
PER_SEED_TIME = 25
N_SEEDS = 10


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def enc_delta(frame, prev_frame):
    """Encode XOR of two frames."""
    a = np.array(frame[0], dtype=np.uint8)
    b = np.array(prev_frame[0], dtype=np.uint8)
    delta = (a ^ b).astype(np.float32) / 15.0
    x = delta.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    x = x - x.mean()
    return x


class DecopledRecode:
    """
    STORES at cell_compare = hash(frame).
    SELECTS from cell_select = hash(frame XOR prev_frame).
    """

    def __init__(self, k=K, dim=DIM, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self._prev_frame = None
        self._cn_compare = None
        self._cn_select = None
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
        n_compare = self._node(x)
        self.live.add(n_compare)
        self.t += 1

        # Delta cell for selection
        if self._prev_frame is not None:
            x_delta = enc_delta(frame, self._prev_frame)
            norm = np.linalg.norm(x_delta)
            if norm > 1e-8:
                x_delta = x_delta / norm
            n_select = self._node(x_delta)
        else:
            n_select = n_compare  # fallback

        # STORE edges at compare cell
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n_compare] = d.get(n_compare, 0) + 1
            k = (self._pn, self._pa, n_compare)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)

        self._px = x
        self._cn_compare = n_compare
        self._cn_select = n_select
        self._prev_frame = frame

        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()

        return n_compare

    def act(self):
        # SELECT from delta cell
        counts = [sum(self.G.get((self._cn_select, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn_compare  # store prev as compare cell
        self._pa = action
        return action

    def on_reset(self):
        self._pn = None
        self._prev_frame = None

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


BASELINE_L1 = [1362, 3270, 48391, 62727, 846, None, None, None, None, None]


def run(seed, make):
    env = make()
    sub = DecopledRecode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = l2 = None
    go = 0
    t_start = time.time()

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            continue

        sub.observe(obs)
        action = sub.act()
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

    bsl = BASELINE_L1[seed]
    if l1 and bsl:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    elif l1:
        spd = "no baseline"
    else:
        spd = "NO_L1"

    print(f"  s{seed}: L1={l1} ({spd}) L2={l2} go={go}", flush=True)
    return dict(seed=seed, l1=l1, l2=l2, go=go)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Decoupled compare/select: {N_SEEDS} seeds, {PER_SEED_TIME}s cap")
    print("STORE at compare cell (hash(frame))")
    print("SELECT from delta cell (hash(frame XOR prev_frame))")

    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1_n = sum(1 for r in results if r['l1'])
    l2_n = sum(1 for r in results if r['l2'])
    print(f"L1={l1_n}/10  L2={l2_n}/10")

    if l1_n >= 3:
        print("FINDING: compare and select are INDEPENDENT — decoupling preserves navigation")
    elif l1_n == 0:
        print("FINDING: compare and select are COUPLED — interpreter is one operation")
    else:
        print(f"MARGINAL: {l1_n}/10 — partial coupling")


if __name__ == "__main__":
    main()
