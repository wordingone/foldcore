"""
Step 640 — Meta-graph tie-breaking via transition profile similarity.

For each cell N, profile v(N) = [count(N,a0),...,count(N,a3)].
When argmin has ties, break using action least-tried across k=5 nearest
neighbors (cosine similarity on profile vectors).

Transfers experience: cells with similar exploration patterns share
action recommendations. The meta-graph IS the world model.

5 seeds × 60s, LS20, k=16 LSH. Compare L1 to 620 baseline.
"""
import numpy as np
import time
import sys

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
K_NEIGHBORS = 5
PER_SEED_TIME = 60

BASELINE_L1 = [1362, 3270, 48391, 62727, 846]


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class Recode:

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
        # Stats
        self.tie_count = 0
        self.tie_changed = 0   # times tie-break chose different action than first tie
        self.total_act = 0

    def _hash(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def _profile(self, n):
        """Action count profile for cell n."""
        v = np.array([sum(self.G.get((n, a), {}).values()) for a in range(N_A)],
                     dtype=np.float32)
        return v

    def _neighbors(self, n):
        """k nearest cells by cosine similarity on profile vectors.
        Returns list of cell ids (may be fewer than k if not enough cells with nonzero profile).
        """
        target = self._profile(n)
        tn = np.linalg.norm(target)
        if tn < 1e-8:
            return []  # no data on current cell yet

        candidates = []
        for cell in self.live:
            if cell == n:
                continue
            v = self._profile(cell)
            vn = np.linalg.norm(v)
            if vn < 1e-8:
                continue
            sim = float(np.dot(target, v) / (tn * vn))
            candidates.append((sim, cell))

        candidates.sort(reverse=True, key=lambda x: x[0])
        return [cell for _, cell in candidates[:K_NEIGHBORS]]

    def _tiebreak(self, tied_actions):
        """Break ties using neighbor action profiles."""
        neighbors = self._neighbors(self._cn)
        if not neighbors:
            return int(tied_actions[0])  # no data: pick first

        # Sum neighbor counts per action
        neighbor_counts = np.zeros(N_A, dtype=np.float32)
        for nb in neighbors:
            neighbor_counts += self._profile(nb)

        # Among tied actions, pick the one with lowest neighbor count
        best_a = tied_actions[0]
        best_cnt = neighbor_counts[best_a]
        for a in tied_actions[1:]:
            if neighbor_counts[a] < best_cnt:
                best_cnt = neighbor_counts[a]
                best_a = a
        return int(best_a)

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
        min_count = min(counts)
        tied = [a for a, c in enumerate(counts) if c == min_count]
        self.total_act += 1

        if len(tied) > 1:
            self.tie_count += 1
            action = self._tiebreak(tied)
            if action != tied[0]:
                self.tie_changed += 1
        else:
            action = tied[0]

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


def t0():
    sub = Recode(seed=0)

    # Profile of unknown cell = all zeros
    p = sub._profile(99)
    assert np.all(p == 0), "unknown cell should have zero profile"

    # No neighbors when profiles are zero
    sub.live.add(99)
    nbrs = sub._neighbors(99)
    assert nbrs == [], "no neighbors before any data"

    # Tiebreak with all tied → returns first (no neighbor data)
    sub._cn = 1
    sub.live.add(1)
    action = sub._tiebreak([0, 1, 2, 3])
    assert action == 0, "no data: should return first tied action"

    print("T0 PASS")


def run(seed, make):
    env = make()
    sub = Recode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = l2 = None
    go = 0
    t_start = time.time()

    for step in range(1, 500_001):
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

    tie_rate = 100.0 * sub.tie_count / sub.total_act if sub.total_act else 0.0
    changed_rate = (100.0 * sub.tie_changed / sub.tie_count
                    if sub.tie_count > 0 else 0.0)
    bsl = BASELINE_L1[seed]
    if l1:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    else:
        spd = "NO_L1"
    print(f"  s{seed}: L1={l1} ({spd}) go={go} "
          f"tie_rate={tie_rate:.1f}% changed={changed_rate:.0f}% "
          f"unique={len(sub.live)}", flush=True)

    return dict(seed=seed, l1=l1, l2=l2, go=go,
                tie_rate=tie_rate, changed_rate=changed_rate,
                unique=len(sub.live))


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        print("\nDry run only. T0 passed.")
        return

    R = []
    for seed in range(5):
        print(f"\nseed {seed}:", flush=True)
        R.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1n = sum(1 for r in R if r['l1'])
    l1_valid = [(r['l1'], BASELINE_L1[r['seed']]) for r in R if r['l1']]
    avg_ratio = np.mean([b / l for l, b in l1_valid]) if l1_valid else 0.0
    avg_tie = np.mean([r['tie_rate'] for r in R])
    avg_changed = np.mean([r['changed_rate'] for r in R if r['tie_rate'] > 0])

    for r in R:
        bsl = BASELINE_L1[r['seed']]
        if r['l1']:
            ratio = bsl / r['l1']
            spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
        else:
            spd = "NO_L1"
        print(f"  s{r['seed']}: L1={r['l1']} ({spd}) "
              f"tie_rate={r['tie_rate']:.1f}% changed={r['changed_rate']:.0f}%")

    print(f"\nL1={l1n}/5  avg_speedup={avg_ratio:.2f}x  "
          f"avg_tie_rate={avg_tie:.1f}%  avg_changed={avg_changed:.0f}%")

    if l1n >= 4 and avg_ratio > 1.0:
        print("SIGNAL: meta-graph tie-breaking faster than baseline.")
    elif l1n < 3:
        print("KILL: L1 < 3/5.")
    else:
        print(f"MARGINAL: L1={l1n}/5 avg_speedup={avg_ratio:.2f}x")


if __name__ == "__main__":
    main()
