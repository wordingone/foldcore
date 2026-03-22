"""
Step 652 — Exit cell visit count diagnostic.

Is L1 about EXPLORATION (reaching the exit cell) or
RECOGNITION (being in the right hidden state when at the exit cell)?

Instrument standard LSH k=12 argmin on LS20.
For each seed, track which cell triggers L1, how many times visited before.

exit_visits=1 → exploration (first visit = trigger)
exit_visits>>1 → recognition (agent returns many times before trigger)
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
PER_SEED_TIME = 30  # 10 seeds × 30s = 300s < 5 min
N_SEEDS = 10


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


def run(seed, make):
    env = make()
    sub = Recode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = None
    go = 0
    t_start = time.time()

    # Visit log: cell -> sorted list of step numbers
    visit_log = {}
    exit_cell = None
    exit_visits = None

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            continue

        cell = sub.observe(obs)
        visit_log.setdefault(cell, []).append(step)

        action = sub.act()
        obs, reward, done, info = env.step(action)

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
                exit_cell = cell
                exit_visits = list(visit_log.get(cell, []))
            level = cl
            sub.on_reset()

        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset()

        if time.time() - t_start > PER_SEED_TIME:
            break

    elapsed = time.time() - t_start

    if l1 is not None:
        # visits_before = visits strictly before the trigger step
        visits_before = [v for v in exit_visits if v < l1]
        n_before = len(visits_before)
        first_visit = exit_visits[0]
        gap = l1 - first_visit
        print(f"  s{seed}: L1={l1} visits_before_trigger={n_before} "
              f"first_visit={first_visit} gap={gap} t={elapsed:.1f}s", flush=True)
    else:
        print(f"  s{seed}: L1=None t={elapsed:.1f}s", flush=True)

    return dict(
        seed=seed, l1=l1,
        n_before=len([v for v in exit_visits if v < l1]) if exit_visits else None,
        first_visit=exit_visits[0] if exit_visits else None,
        gap=(l1 - exit_visits[0]) if exit_visits else None,
    )


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Exit cell diagnostic: {N_SEEDS} seeds, {PER_SEED_TIME}s cap")

    results = []
    for seed in range(N_SEEDS):
        print(f"\nseed {seed}:", flush=True)
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1_r = [r for r in results if r['l1'] is not None]
    print(f"L1 reached: {len(l1_r)}/{N_SEEDS}")

    if l1_r:
        nb = [r['n_before'] for r in l1_r]
        gaps = [r['gap'] for r in l1_r]
        print(f"visits_before_trigger: avg={np.mean(nb):.1f} min={min(nb)} max={max(nb)}")
        print(f"gap (first_visit to trigger): avg={np.mean(gaps):.1f} min={min(gaps)} max={max(gaps)}")

        if max(nb) == 0:
            verdict = "PURE EXPLORATION: first visit always triggers L1"
        elif np.mean(nb) > 10:
            verdict = f"RECOGNITION: avg {np.mean(nb):.1f} prior visits before trigger"
        elif np.mean(nb) > 2:
            verdict = f"MIXED: avg {np.mean(nb):.1f} prior visits"
        else:
            verdict = f"NEAR-EXPLORATION: avg {np.mean(nb):.1f} prior visits"
        print(f"\n{verdict}")


if __name__ == "__main__":
    main()
