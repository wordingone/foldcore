"""
Step 634 — Binary mask with more bits, separate delta H matrix.

Variant A: Pack ALL 256 bits of binary mask → hash with kd=4 (16 delta cells).
Variant B: Pack ALL 256 bits → hash with kd=8 (256 delta cells).

CRITICAL: separate H matrix for delta hashing (independent of state H).
State hashing: k=16 rows. Delta hashing: kd rows (separate random seed).

PREFER_BONUS = 10 (down from 50), AVOID_PENALTY = 20 (down from 100).
Softer bias for context-sensitivity with more cells.

5 seeds × 60s per variant on LS20. Compare L1 to 620 baseline.
"""
import numpy as np
import time
import sys
from collections import Counter, defaultdict

N_A = 4
K_STATE = 16   # state hashing fixed at k=16
DIM = 256      # 16x16 avgpool grid
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
WARMUP = 500
STALE_UPDATE_EVERY = 1000
EPSILON_WARMUP = 1000
PER_SEED_TIME = 60
AVOID_PENALTY = 20   # softened from 100
PREFER_BONUS = 10    # softened from 50

BASELINE_L1 = [1362, 3270, 48391, 62727, 846]
KD_VALUES = [4, 8]


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def enc_delta_raw(curr_frame, prev_frame):
    """Return raw (uncentered) avgpool delta."""
    a = np.array(curr_frame[0], dtype=np.float32) / 15.0
    b = np.array(prev_frame[0], dtype=np.float32) / 15.0
    return (a - b).reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


class Recode:

    def __init__(self, kd=4, dim=DIM, seed=0):
        # State hashing: k=16, seed=seed
        self.H_state = np.random.RandomState(seed).randn(K_STATE, dim).astype(np.float32)
        # Delta hashing: kd rows, SEPARATE seed (seed + 999999)
        self.H_delta = np.random.RandomState(seed + 999999).randn(kd, dim).astype(np.float32)
        self.kd = kd
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self._prev_frame = None
        self._last_dc = None
        self.t = 0
        self.ns = 0
        self.dim = dim
        # Delta tracking
        self.delta_by_action = [Counter() for _ in range(N_A)]
        self.delta_by_na = defaultdict(Counter)
        self.total_transitions = 0
        # Epsilon estimation (adaptive)
        self._delta_abs_sum = 0.0
        self._delta_abs_count = 0
        self.epsilon = None
        # Action selection
        self.productive_set = set()
        self.stale_set = set()
        self.op_counts = [0, 0, 0]

    def _hash_state(self, x):
        return int(np.packbits(
            (self.H_state @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _hash_delta(self, mask):
        """Hash full 256-bit binary mask using separate delta H matrix.
        mask: bool array of shape (DIM,).
        Project mask (as float) through H_delta → kd-bit hash.
        """
        v = mask.astype(np.float32) - 0.5  # center: True=+0.5, False=-0.5
        return int(np.packbits(
            (self.H_delta @ v > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash_state(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def _binary_mask(self, raw_delta):
        if self.epsilon is None:
            return None
        return np.abs(raw_delta) > self.epsilon

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

            if self._prev_frame is not None:
                raw_d = enc_delta_raw(frame, self._prev_frame)
                # Update epsilon estimate
                self._delta_abs_sum += float(np.mean(np.abs(raw_d)))
                self._delta_abs_count += 1
                if self._delta_abs_count == EPSILON_WARMUP and self.epsilon is None:
                    self.epsilon = self._delta_abs_sum / self._delta_abs_count

                mask = self._binary_mask(raw_d)
                if mask is not None:
                    dc = self._hash_delta(mask)
                    self._last_dc = dc
                    self.delta_by_action[self._pa][dc] += 1
                    self.delta_by_na[(self._pn, self._pa)][dc] += 1
                    self.total_transitions += 1

        self._px = x
        self._cn = n
        self._prev_frame = frame

        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        if self.t >= WARMUP and self.t % STALE_UPDATE_EVERY == 0:
            self._update_stale()
        return n

    def tag_productive(self):
        if self._last_dc is not None:
            self.productive_set.add(self._last_dc)

    def _update_stale(self):
        if self.total_transitions == 0:
            return
        combined = Counter()
        for a in range(N_A):
            combined.update(self.delta_by_action[a])
        total = sum(combined.values())
        self.stale_set = {dc for dc, cnt in combined.items() if cnt / total > 0.8}

    def act(self):
        counts = []
        for a in range(N_A):
            base_count = sum(self.G.get((self._cn, a), {}).values())
            na_ctr = self.delta_by_na.get((self._cn, a))
            if na_ctr:
                top_dc = na_ctr.most_common(1)[0][0]
                if top_dc in self.productive_set:
                    modified = max(0, base_count - PREFER_BONUS)
                    self.op_counts[2] += 1
                elif top_dc in self.stale_set:
                    modified = base_count + AVOID_PENALTY
                    self.op_counts[1] += 1
                else:
                    modified = base_count
                    self.op_counts[0] += 1
            else:
                modified = base_count
                self.op_counts[0] += 1
            counts.append(modified)
        action = int(np.argmin(counts))
        self._pn = self._cn
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

    def unique_delta_cells(self):
        combined = Counter()
        for a in range(N_A):
            combined.update(self.delta_by_action[a])
        return len(combined)

    def op_stats(self):
        total = sum(self.op_counts)
        if total == 0:
            return 0.0, 0.0, 100.0
        return (
            100.0 * self.op_counts[1] / total,
            100.0 * self.op_counts[2] / total,
            100.0 * self.op_counts[0] / total,
        )


def t0():
    sub4 = Recode(kd=4, seed=0)
    sub8 = Recode(kd=8, seed=0)

    # Separate H matrices — different seeds means different values
    assert not np.allclose(sub4.H_state[:4, :], sub4.H_delta), \
        "H_state and H_delta must be independent"

    # Full 256-bit mask hashing
    rng = np.random.RandomState(42)
    mask1 = rng.rand(DIM) > 0.5
    mask2 = mask1.copy()
    assert sub4._hash_delta(mask1) == sub4._hash_delta(mask2), \
        "same mask should give same cell"

    # Different masks should (usually) differ
    mask3 = ~mask1
    # Not guaranteed but very likely for opposite mask
    dc1 = sub4._hash_delta(mask1)
    dc3 = sub4._hash_delta(mask3)
    # kd=4 → at most 16 cells
    cells = set(sub4._hash_delta(rng.rand(DIM) > 0.5) for _ in range(200))
    assert len(cells) <= 16, f"kd=4 should give <=16 cells, got {len(cells)}"

    cells8 = set(sub8._hash_delta(rng.rand(DIM) > 0.5) for _ in range(500))
    assert len(cells8) <= 256, f"kd=8 should give <=256 cells, got {len(cells8)}"

    # Epsilon starts None
    assert sub4.epsilon is None

    # tag_productive
    sub4._last_dc = 42
    sub4.tag_productive()
    assert 42 in sub4.productive_set

    print("T0 PASS")


def run(seed, kd, make):
    env = make()
    sub = Recode(kd=kd, seed=seed * 1000)
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
                sub.tag_productive()
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

    av, pr, nt = sub.op_stats()
    prod = len(sub.productive_set)
    stale = len(sub.stale_set)
    unique = sub.unique_delta_cells()
    eps_str = f"{sub.epsilon:.4f}" if sub.epsilon else "not_set"
    bsl = BASELINE_L1[seed]
    if l1:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    else:
        spd = "NO_L1"
    print(f"  kd={kd} s{seed}: L1={l1} ({spd}) go={go} eps={eps_str} "
          f"unique_dc={unique} prod={prod} stale={stale} "
          f"ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%", flush=True)

    return dict(seed=seed, kd=kd, l1=l1, l2=l2, go=go,
                unique=unique, prod=prod, stale=stale,
                avoid_pct=av, prefer_pct=pr, epsilon=sub.epsilon)


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

    results = {}
    for kd in KD_VALUES:
        print(f"\n--- kd={kd} (Variant {'A' if kd == 4 else 'B'}) ---", flush=True)
        R = []
        for seed in range(5):
            print(f"\nseed {seed}:", flush=True)
            R.append(run(seed, kd, mk))
        results[kd] = R

    print(f"\n{'='*60}")
    for kd in KD_VALUES:
        R = results[kd]
        l1n = sum(1 for r in R if r['l1'])
        l1_valid = [(r['l1'], BASELINE_L1[r['seed']]) for r in R if r['l1']]
        avg_ratio = np.mean([b / l for l, b in l1_valid]) if l1_valid else 0.0
        avg_unique = np.mean([r['unique'] for r in R])
        avg_prod = np.mean([r['prod'] for r in R])
        avg_stale = np.mean([r['stale'] for r in R])
        avg_av = np.mean([r['avoid_pct'] for r in R])
        avg_pr = np.mean([r['prefer_pct'] for r in R])
        avg_eps = np.mean([r['epsilon'] for r in R if r['epsilon']])
        print(f"  kd={kd}: L1={l1n}/5  avg_speedup={avg_ratio:.2f}x  "
              f"avg_unique={avg_unique:.1f}  avg_prod={avg_prod:.1f}  "
              f"avg_stale={avg_stale:.1f}  ops=A{avg_av:.0f}%P{avg_pr:.0f}%  "
              f"avg_eps={avg_eps:.4f}")

    print()
    for kd in KD_VALUES:
        R = results[kd]
        l1n = sum(1 for r in R if r['l1'])
        l1_valid = [(r['l1'], BASELINE_L1[r['seed']]) for r in R if r['l1']]
        avg_ratio = np.mean([b / l for l, b in l1_valid]) if l1_valid else 0.0
        avg_stale = np.mean([r['stale'] for r in R])
        avg_av = np.mean([r['avoid_pct'] for r in R])

        if avg_stale > 0 and avg_av > 1 and l1n >= 4 and avg_ratio > 1.0:
            print(f"kd={kd}: SIGNAL — stale_set non-empty AND L1 faster than baseline")
        elif avg_stale > 0 and avg_av > 1:
            print(f"kd={kd}: PARTIAL — stale_set non-empty, delta ops active, "
                  f"L1={l1n}/5 avg_speedup={avg_ratio:.2f}x")
        elif l1n >= 4 and avg_ratio > 1.0:
            print(f"kd={kd}: SIGNAL — L1 faster than baseline (stale_set empty)")
        elif l1n < 3:
            print(f"kd={kd}: KILL — L1 < 3/5")
        else:
            print(f"kd={kd}: INERT — delta not propagating "
                  f"(stale={avg_stale:.1f} av={avg_av:.0f}%)")


if __name__ == "__main__":
    main()
