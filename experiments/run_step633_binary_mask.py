"""
Step 633 — Binary change mask delta encoding.

Instead of LSH on avgpool-delta, threshold |delta| > epsilon per 16x16 block
→ binary "changed/unchanged" vector → hash that with k=16.

epsilon = mean(|delta|) across first EPSILON_WARMUP transitions (adaptive).

Hypothesis: binary mask groups "same pattern of changes" together,
giving far fewer unique delta_cells than continuous LSH.

5 seeds × 60s on LS20. Compare L1 steps to 620 baseline.
"""
import numpy as np
import time
import sys
from collections import Counter, defaultdict

N_A = 4
K = 16
DIM = 256  # 16x16 avgpool grid
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
WARMUP = 500
STALE_UPDATE_EVERY = 1000
EPSILON_WARMUP = 1000
PER_SEED_TIME = 60
AVOID_PENALTY = 100
PREFER_BONUS = 50

BASELINE_L1 = [1362, 3270, 48391, 62727, 846]


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

    def __init__(self, k=K, dim=DIM, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
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
        self.epsilon = None  # set after EPSILON_WARMUP transitions
        # Action selection
        self.productive_set = set()
        self.stale_set = set()
        self.op_counts = [0, 0, 0]

    def _hash_state(self, x):
        """Hash state vector for node identity (k=16 LSH)."""
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _hash_mask(self, mask):
        """Hash binary change mask → delta_cell.
        mask: bool array of shape (DIM,), True = changed block.
        Pack bits directly → up to 2^DIM cells, but typically very few unique.
        Use first K bits for compactness.
        """
        bits = mask[:K].astype(np.uint8)
        return int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash_state(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def _binary_mask(self, raw_delta):
        """Convert raw delta to binary changed/unchanged mask."""
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
                    dc = self._hash_mask(mask)
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
    rng = np.random.RandomState(42)
    sub = Recode(seed=0)

    # Test _hash_mask: same mask → same cell
    mask1 = np.array([True, False, True, False] + [False] * (K - 4), dtype=bool)
    mask2 = np.array([True, False, True, False] + [False] * (K - 4), dtype=bool)
    assert sub._hash_mask(mask1) == sub._hash_mask(mask2), "same mask should give same cell"

    # Test that k=16 binary mask gives at most 2^16 = 65536 cells (in practice much fewer)
    # All-True mask
    all_true = np.ones(K, dtype=bool)
    all_false = np.zeros(K, dtype=bool)
    dc_true = sub._hash_mask(all_true)
    dc_false = sub._hash_mask(all_false)
    assert dc_true != dc_false, "all-changed vs no-change should differ"

    # Test epsilon warmup: before EPSILON_WARMUP, epsilon is None
    assert sub.epsilon is None

    # Test tag_productive
    sub._last_dc = 42
    sub.tag_productive()
    assert 42 in sub.productive_set

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
    print(f"  s{seed}: L1={l1} ({spd}) go={go} eps={eps_str} "
          f"unique_dc={unique} prod={prod} stale={stale} "
          f"ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%", flush=True)

    return dict(seed=seed, l1=l1, l2=l2, go=go,
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

    R = []
    for seed in range(5):
        print(f"\nseed {seed}:", flush=True)
        R.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1n = sum(1 for r in R if r['l1'])
    l1_valid = [(r['l1'], BASELINE_L1[r['seed']]) for r in R if r['l1']]
    avg_ratio = np.mean([b / l for l, b in l1_valid]) if l1_valid else 0.0
    avg_unique = np.mean([r['unique'] for r in R])
    avg_prod = np.mean([r['prod'] for r in R])
    avg_stale = np.mean([r['stale'] for r in R])
    avg_av = np.mean([r['avoid_pct'] for r in R])
    avg_pr = np.mean([r['prefer_pct'] for r in R])
    avg_eps = np.mean([r['epsilon'] for r in R if r['epsilon']])

    for r in R:
        bsl = BASELINE_L1[r['seed']]
        if r['l1']:
            ratio = bsl / r['l1']
            spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
        else:
            spd = "NO_L1"
        print(f"  s{r['seed']}: L1={r['l1']} ({spd}) unique={r['unique']} "
              f"prod={r['prod']} stale={r['stale']} ops=A{r['avoid_pct']:.0f}%P{r['prefer_pct']:.0f}%")

    print(f"\nL1={l1n}/5  avg_speedup={avg_ratio:.2f}x  avg_unique={avg_unique:.0f}  "
          f"avg_prod={avg_prod:.1f}  avg_stale={avg_stale:.1f}  "
          f"ops=A{avg_av:.0f}%P{avg_pr:.0f}%  avg_eps={avg_eps:.4f}")

    if avg_stale > 0 and avg_av > 1:
        print("SIGNAL: stale_set non-empty, delta ops active — binary mask has structure.")
    elif l1n >= 4 and avg_ratio > 1.0:
        print("SIGNAL: L1 faster than baseline.")
    elif l1n < 3:
        print("KILL: L1 < 3/5.")
    else:
        print("INERT: binary mask delta still not propagating.")


if __name__ == "__main__":
    main()
