"""
Step 639 + 639b — large_diff per-edge penalty.

639:  PENALTY=100 on large_diff edges (frame_diff > 0.082).
      Same mechanism as 581d death penalty, different trigger.

639b: Combined large_diff + game_over.
      PENALTY=100 for game_over, PENALTY=50 for large_diff.
      Key diagnostic: overlap fraction between large_diff and game_over.

5 seeds × 60s, LS20, k=16 LSH.
Compare L1 to 620 baseline [1362, 3270, 48391, 62727, 846].
"""
import numpy as np
import time
import sys
from collections import defaultdict

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
FRAME_DIFF_THRESH = 0.082
PER_SEED_TIME = 60

BASELINE_L1 = [1362, 3270, 48391, 62727, 846]

MODES = ['639', '639b']


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def frame_diff(curr_frame, prev_frame):
    a = np.array(curr_frame[0], dtype=np.float32) / 15.0
    b = np.array(prev_frame[0], dtype=np.float32) / 15.0
    return float(np.mean(np.abs(a - b)))


class Recode:

    def __init__(self, k=K, dim=DIM, seed=0, mode='639'):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.ns = 0
        self.dim = dim
        self.mode = mode
        # Per-edge penalty accumulator (sum of penalties applied)
        self.edge_penalty = defaultdict(float)
        # Stats
        self.large_diff_count = 0
        self.game_over_count = 0
        self.both_count = 0   # game_over AND large_diff on same step
        self.total_steps = 0

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

    def on_large_diff(self):
        """Call after env.step() when frame_diff > threshold.
        Penalizes the edge (pn, pa) that produced the large visual change.
        """
        if self._pn is None:
            return
        self.large_diff_count += 1
        if self.mode == '639':
            self.edge_penalty[(self._pn, self._pa)] += 100
        elif self.mode == '639b':
            self.edge_penalty[(self._pn, self._pa)] += 50

    def on_game_over(self):
        """Call when done=True. Penalizes the last edge taken."""
        if self._pn is None:
            return
        self.game_over_count += 1
        if self.mode == '639b':
            self.edge_penalty[(self._pn, self._pa)] += 100

    def on_both(self):
        """Track co-occurrence."""
        self.both_count += 1

    def act(self):
        counts = []
        for a in range(N_A):
            base_count = sum(self.G.get((self._cn, a), {}).values())
            penalty = self.edge_penalty.get((self._cn, a), 0.0)
            counts.append(base_count + penalty)
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

    def on_reset(self):
        self._pn = None

    def penalized_edge_count(self):
        return len(self.edge_penalty)

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
    sub639 = Recode(seed=0, mode='639')
    sub639b = Recode(seed=0, mode='639b')

    # 639: large_diff adds 100 to edge penalty
    sub639._pn = 1
    sub639._pa = 0
    sub639.on_large_diff()
    assert sub639.edge_penalty[(1, 0)] == 100.0, "639: large_diff should add 100"
    sub639.on_game_over()  # game_over should NOT add penalty in 639
    assert sub639.edge_penalty[(1, 0)] == 100.0, "639: game_over should not add penalty"

    # 639b: large_diff adds 50, game_over adds 100
    sub639b._pn = 1
    sub639b._pa = 0
    sub639b.on_large_diff()
    assert sub639b.edge_penalty[(1, 0)] == 50.0, "639b: large_diff should add 50"
    sub639b.on_game_over()
    assert sub639b.edge_penalty[(1, 0)] == 150.0, "639b: game_over should add 100 more"

    print("T0 PASS")


def run(seed, mode, make):
    env = make()
    sub = Recode(seed=seed * 1000, mode=mode)
    obs = env.reset(seed=seed)
    level = 0
    l1 = l2 = None
    go = 0
    prev_obs = None
    t_start = time.time()

    for step in range(1, 500_001):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_obs = None
            continue

        sub.observe(obs)
        action = sub.act()
        prev_obs = obs
        obs, reward, done, info = env.step(action)
        sub.total_steps += 1

        # Detect large_diff on this transition
        is_large_diff = False
        if prev_obs is not None:
            fd = frame_diff(obs, prev_obs)
            if fd > FRAME_DIFF_THRESH:
                is_large_diff = True

        # Detect game_over
        is_game_over = done

        # Apply penalties
        if is_large_diff and is_game_over:
            sub.on_both()
        if is_large_diff:
            sub.on_large_diff()
        if is_game_over:
            sub.on_game_over()

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
            prev_obs = None

        if time.time() - t_start > PER_SEED_TIME:
            break

    ts = sub.total_steps
    ld_rate = 100.0 * sub.large_diff_count / ts if ts else 0.0
    go_rate = 100.0 * sub.game_over_count / ts if ts else 0.0
    overlap = (100.0 * sub.both_count / sub.large_diff_count
               if sub.large_diff_count > 0 else 0.0)
    pen_edges = sub.penalized_edge_count()
    bsl = BASELINE_L1[seed]
    if l1:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    else:
        spd = "NO_L1"
    print(f"  [{mode}] s{seed}: L1={l1} ({spd}) go={go} "
          f"ld={ld_rate:.1f}% go_rate={go_rate:.1f}% overlap={overlap:.0f}% "
          f"pen_edges={pen_edges}", flush=True)

    return dict(seed=seed, mode=mode, l1=l1, l2=l2, go=go,
                ld_rate=ld_rate, go_rate=go_rate, overlap=overlap,
                pen_edges=pen_edges)


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
    for mode in MODES:
        print(f"\n{'='*50}")
        print(f"--- Mode: {mode} ---", flush=True)
        R = []
        for seed in range(5):
            print(f"\nseed {seed}:", flush=True)
            R.append(run(seed, mode, mk))
        results[mode] = R

    print(f"\n{'='*60}")
    for mode in MODES:
        R = results[mode]
        l1n = sum(1 for r in R if r['l1'])
        l1_valid = [(r['l1'], BASELINE_L1[r['seed']]) for r in R if r['l1']]
        avg_ratio = np.mean([b / l for l, b in l1_valid]) if l1_valid else 0.0
        avg_ld = np.mean([r['ld_rate'] for r in R])
        avg_go = np.mean([r['go_rate'] for r in R])
        avg_overlap = np.mean([r['overlap'] for r in R])
        avg_pen = np.mean([r['pen_edges'] for r in R])
        print(f"  {mode}: L1={l1n}/5  avg_speedup={avg_ratio:.2f}x  "
              f"avg_ld={avg_ld:.1f}%  avg_go={avg_go:.1f}%  "
              f"overlap={avg_overlap:.0f}%  avg_pen_edges={avg_pen:.0f}")

    print()
    # Signal assessment per mode
    for mode in MODES:
        R = results[mode]
        l1n = sum(1 for r in R if r['l1'])
        l1_valid = [(r['l1'], BASELINE_L1[r['seed']]) for r in R if r['l1']]
        avg_ratio = np.mean([b / l for l, b in l1_valid]) if l1_valid else 0.0
        if l1n >= 4 and avg_ratio > 1.0:
            print(f"{mode}: SIGNAL — L1 faster than baseline.")
        elif l1n < 3:
            print(f"{mode}: KILL — L1 < 3/5.")
        else:
            print(f"{mode}: MARGINAL — L1={l1n}/5 avg_speedup={avg_ratio:.2f}x")


if __name__ == "__main__":
    main()
