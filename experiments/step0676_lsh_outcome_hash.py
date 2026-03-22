"""
Step 676 — Outcome-hash refinement (fix Step 667 sparse key problem).

Step 667 failed: raw (cell, prev_outcome) tuples too sparse.
Fix: HASH the outcome tuple to 2 bits = 4 outcome classes.

key = (current_cell, outcome_class)
outcome_class = 2-bit hash of (prev_cell, prev_action, current_cell)

4x key expansion (4 outcome classes x N cells).
Different arrival paths trigger different action selections.

10 seeds, 25s cap.
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
N_OUTCOME_CLASSES = 4  # 2-bit hash -> 4 classes

BASELINE_L1 = {0: 1362, 1: 3270, 2: 48391, 3: 62727, 4: 846}


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def outcome_class(prev_cell, prev_action, current_cell):
    """Hash (prev_cell, prev_action, current_cell) to 0-3."""
    h = hash((prev_cell, prev_action, current_cell))
    return h % N_OUTCOME_CLASSES


class OutcomeHashRecode:
    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(K, DIM).astype(np.float32)
        self.ref = {}
        self.G = {}   # ((cell, oc), action) -> {successor: count}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self._poc = None   # previous outcome class
        self.t = 0
        self.dim = DIM
        self._cn = None
        self._coc = None   # current outcome class

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

        # Compute outcome class: hash of (prev_cell, prev_action, current_cell)
        if self._pn is not None:
            oc = outcome_class(self._pn, self._pa, n)
        else:
            oc = 0  # no previous context -> class 0

        # Key = (current_cell, outcome_class)
        key = (n, oc)

        if self._pn is not None:
            prev_key = (self._pn, self._poc)
            d = self.G.setdefault((prev_key, self._pa), {})
            d[key] = d.get(key, 0) + 1
            k = (prev_key, self._pa, key)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)

        self._px = x
        self._cn = n
        self._coc = oc

        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return (n, oc)

    def act(self):
        key = (self._cn, self._coc)
        best_a, best_s = 0, float('inf')
        for a in range(N_A):
            s = sum(self.G.get((key, a), {}).values())
            if s < best_s:
                best_s = s
                best_a = a
        self._pn = self._cn
        self._poc = self._coc
        self._pa = best_a
        return best_a

    def on_reset(self):
        self._pn = None
        self._poc = None

    def _h_key(self, key, a):
        d = self.G.get((key, a))
        if not d or sum(d.values()) < 4:
            return 0.0
        v = np.array(list(d.values()), np.float64)
        p = v / v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self):
        # Refinement on the raw cell hash (not the keyed graph)
        # Use G to find refineable nodes (by raw cell)
        did = 0
        # Collect raw cell transition stats
        raw_G = {}  # (raw_cell, action) -> {raw_successor: count}
        for (key, a), d in self.G.items():
            raw_n = key[0]
            for succ_key, cnt in d.items():
                raw_succ = succ_key[0]
                rk = (raw_n, a)
                raw_G.setdefault(rk, {})
                raw_G[rk][raw_succ] = raw_G[rk].get(raw_succ, 0) + cnt

        for (n, a), d in raw_G.items():
            if n not in self.live or n in self.ref:
                continue
            if len(d) < 2 or sum(d.values()) < MIN_OBS:
                continue
            v = np.array(list(d.values()), np.float64)
            p = v / v.sum()
            h = float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))
            if h < H_SPLIT:
                continue
            # Build centroid vectors from C
            top = sorted(d, key=d.get, reverse=True)[:2]
            # Look up in C (raw key)
            r0 = self.C.get(((n, 0), a, (top[0], 0)))
            if r0 is None:
                # Try all outcome classes
                for oc in range(N_OUTCOME_CLASSES):
                    r0 = self.C.get(((n, oc), a, (top[0], oc)))
                    if r0 is not None:
                        break
            r1 = self.C.get(((n, 0), a, (top[1], 0)))
            if r1 is None:
                for oc in range(N_OUTCOME_CLASSES):
                    r1 = self.C.get(((n, oc), a, (top[1], oc)))
                    if r1 is not None:
                        break
            if r0 is None or r1 is None or r0[1] < 3 or r1[1] < 3:
                continue
            diff = (r0[0] / r0[1]) - (r1[0] / r1[1])
            nm = np.linalg.norm(diff)
            if nm < 1e-8:
                continue
            self.ref[n] = (diff / nm).astype(np.float32)
            self.live.discard(n)
            did += 1
            if did >= 3:
                break


def run(seed, make):
    env = make()
    sub = OutcomeHashRecode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = l2 = None
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
            obs = env.reset(seed=seed)
            sub.on_reset()
        if time.time() - t_start > PER_SEED_TIME:
            break

    bsl = BASELINE_L1.get(seed)
    if l1 and bsl:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    elif l1:
        spd = "no_baseline"
    else:
        spd = "NO_L1"

    n_keys = len(sub.G)
    print(f"  s{seed:2d}: L1={l1} ({spd}) L2={l2} n_keys={n_keys}", flush=True)
    return dict(seed=seed, l1=l1, l2=l2)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Outcome-hash refinement ({N_OUTCOME_CLASSES} classes): {N_SEEDS} seeds, {PER_SEED_TIME}s cap")

    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1_n = sum(1 for r in results if r['l1'])
    l2_n = sum(1 for r in results if r['l2'])
    l1_valid = [(r['l1'], BASELINE_L1[r['seed']]) for r in results
                if r['l1'] and BASELINE_L1.get(r['seed'])]
    avg_ratio = float(np.mean([b / l for l, b in l1_valid])) if l1_valid else 0.0

    print(f"L1={l1_n}/{N_SEEDS}  L2={l2_n}/{N_SEEDS}  avg_speedup={avg_ratio:.2f}x")

    if l2_n > 0:
        print("BREAKTHROUGH: L2 reached — outcome context resolves POMDP")
    elif l1_n >= 8:
        print("STRONG FINDING: Outcome-hash context works")
    elif l1_n >= 6 and avg_ratio > 2.0:
        print("FINDING: Outcome hashing improves over 667")
    elif l1_n < 3:
        print("KILL: Outcome-hash fails — context still too sparse")
    else:
        print(f"MARGINAL: L1={l1_n}/{N_SEEDS}")


if __name__ == "__main__":
    main()
