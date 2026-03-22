"""
Step 638 — Environmental event diagnostic (pure logging, no mechanism).

Run pure argmin baseline (Step 620 setup) for 5 seeds × 60s.
Log EVERY detectable environmental event:

1. frame_diff > 0.082 (bimodal gap from step 558) — large visual change events
2. game-over events (done=True, env.reset() called)
3. level transitions (info['level'] increases)
4. new cell discovery (first visit to an LSH cell) — rate over time
5. observation repetition (same frame hash seen before)
6. reward signal (if any non-zero reward occurs)

Report: event counts per seed, rates in first 5K vs last 5K steps.
Target: identify events firing at 1-10% rate (sparsity sweet spot from 581d).
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
FRAME_DIFF_THRESH = 0.082  # bimodal gap from step 558
RATE_WINDOW = 5000
PER_SEED_TIME = 60

BASELINE_L1 = [1362, 3270, 48391, 62727, 846]


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def frame_hash_raw(frame):
    """Exact frame hash for repetition detection (not LSH — exact pixel hash)."""
    return hash(np.array(frame[0], dtype=np.uint8).tobytes())


def frame_diff(curr_frame, prev_frame):
    """Mean absolute pixel difference (normalized to [0,1])."""
    a = np.array(curr_frame[0], dtype=np.float32) / 15.0
    b = np.array(prev_frame[0], dtype=np.float32) / 15.0
    return float(np.mean(np.abs(a - b)))


class Recode:
    """Pure argmin substrate (step 620 baseline)."""

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
        self.known_cells = set()

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
        is_new = n not in self.known_cells
        self.known_cells.add(n)

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
        return n, is_new

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


def t0():
    sub = Recode(seed=0)
    rng = np.random.RandomState(42)
    x = rng.randn(DIM).astype(np.float32)
    assert isinstance(sub._hash(x), int)
    print("T0 PASS")


def run(seed, make):
    env = make()
    sub = Recode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = l2 = None
    go = 0
    step = 0
    t_start = time.time()

    # Event counters
    ev = defaultdict(int)
    # Rate tracking: (count_first5k, count_last5k) per event type
    first5k = defaultdict(int)
    last5k = defaultdict(int)
    total_steps = 0

    # Observation history for repetition tracking
    seen_frames = set()
    prev_frame = None
    prev_reward = None

    for step in range(1, 500_001):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            continue

        _, is_new = sub.observe(obs)
        action = sub.act()

        # Frame diff event
        if prev_frame is not None:
            fd = frame_diff(obs, prev_frame)
            if fd > FRAME_DIFF_THRESH:
                ev['large_diff'] += 1
                if step <= RATE_WINDOW:
                    first5k['large_diff'] += 1
                if time.time() - t_start > PER_SEED_TIME - 5:
                    last5k['large_diff'] += 1

        # Observation repetition
        fh = frame_hash_raw(obs)
        if fh in seen_frames:
            ev['repeat_obs'] += 1
            if step <= RATE_WINDOW:
                first5k['repeat_obs'] += 1
            if time.time() - t_start > PER_SEED_TIME - 5:
                last5k['repeat_obs'] += 1
        seen_frames.add(fh)

        # New cell discovery
        if is_new:
            ev['new_cell'] += 1
            if step <= RATE_WINDOW:
                first5k['new_cell'] += 1
            if time.time() - t_start > PER_SEED_TIME - 5:
                last5k['new_cell'] += 1

        prev_frame = obs
        obs, reward, done, info = env.step(action)
        total_steps += 1

        # Reward signal
        if reward != 0 and reward != prev_reward:
            ev['reward_change'] += 1
            if step <= RATE_WINDOW:
                first5k['reward_change'] += 1
        prev_reward = reward

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
            if cl == 2 and l2 is None:
                l2 = step
            level = cl
            ev['level_transition'] += 1
            if step <= RATE_WINDOW:
                first5k['level_transition'] += 1
            sub.on_reset()

        if done:
            go += 1
            ev['game_over'] += 1
            if step <= RATE_WINDOW:
                first5k['game_over'] += 1
            if time.time() - t_start > PER_SEED_TIME - 5:
                last5k['game_over'] += 1
            obs = env.reset(seed=seed)
            sub.on_reset()

        if time.time() - t_start > PER_SEED_TIME:
            break

    bsl = BASELINE_L1[seed]
    spd = "NO_L1"
    if l1:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"

    print(f"\n  s{seed}: L1={l1} ({spd}) total_steps={total_steps} go={go}", flush=True)

    # Report event rates
    event_types = ['large_diff', 'game_over', 'level_transition',
                   'new_cell', 'repeat_obs', 'reward_change']
    for et in event_types:
        cnt = ev[et]
        rate = 100.0 * cnt / total_steps if total_steps else 0.0
        f5 = 100.0 * first5k[et] / min(total_steps, RATE_WINDOW) if total_steps else 0.0
        l5 = 100.0 * last5k[et] / min(total_steps, RATE_WINDOW) if total_steps else 0.0
        sweet = " *** SWEET SPOT (1-10%)" if 1.0 <= rate <= 10.0 else ""
        print(f"    {et:<20}: total={cnt:6d} rate={rate:5.1f}%  "
              f"first5k={f5:5.1f}%  last5k_approx={l5:5.1f}%{sweet}", flush=True)

    return dict(seed=seed, l1=l1, go=go, total_steps=total_steps,
                events=dict(ev), first5k=dict(first5k), last5k=dict(last5k))


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
        print(f"\n{'='*50}")
        print(f"seed {seed}:", flush=True)
        R.append(run(seed, mk))

    # Summary: average rates across seeds
    print(f"\n{'='*60}")
    print("AVERAGE RATES ACROSS 5 SEEDS:")
    event_types = ['large_diff', 'game_over', 'level_transition',
                   'new_cell', 'repeat_obs', 'reward_change']
    for et in event_types:
        rates = []
        for r in R:
            cnt = r['events'].get(et, 0)
            ts = r['total_steps']
            if ts > 0:
                rates.append(100.0 * cnt / ts)
        if rates:
            avg_rate = np.mean(rates)
            sweet = " *** SWEET SPOT (1-10%)" if 1.0 <= avg_rate <= 10.0 else ""
            print(f"  {et:<20}: avg_rate={avg_rate:5.1f}%{sweet}")

    print("\nSWEET SPOT events (1-10% rate) are candidates for sparse negative bias.")


if __name__ == "__main__":
    main()
