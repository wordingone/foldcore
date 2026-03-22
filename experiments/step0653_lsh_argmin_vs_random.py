"""
Step 653 — Seed-matched argmin vs random walk.

Does argmin PREVENT some solutions that random walk reaches?
Run both on same 20 seeds, compare L1 success/failure.

argmin_only: seeds where argmin succeeds and random fails
random_only: seeds where random succeeds and argmin fails
both:        seeds where both succeed
neither:     seeds where both fail
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
PER_SEED_TIME = 5   # 20 seeds × 2 methods × 5s = 200s < 5 min
N_SEEDS = 20


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


def run_argmin(seed, make):
    env = make()
    sub = Recode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = None
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
            level = cl
            sub.on_reset()

        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset()

        if time.time() - t_start > PER_SEED_TIME:
            break

    return l1


def run_random(seed, make):
    env = make()
    rng = np.random.RandomState(seed * 777 + 42)  # deterministic, different from argmin seed
    obs = env.reset(seed=seed)
    level = 0
    l1 = None
    go = 0
    t_start = time.time()

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed)
            continue

        action = int(rng.randint(N_A))
        obs, reward, done, info = env.step(action)

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
            level = cl

        if done:
            go += 1
            obs = env.reset(seed=seed)

        if time.time() - t_start > PER_SEED_TIME:
            break

    return l1


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Argmin vs random: {N_SEEDS} seeds, {PER_SEED_TIME}s per method per seed")

    results = []
    for seed in range(N_SEEDS):
        al1 = run_argmin(seed, mk)
        rl1 = run_random(seed, mk)
        tag = "both" if (al1 and rl1) else \
              "argmin_only" if (al1 and not rl1) else \
              "random_only" if (not al1 and rl1) else "neither"
        print(f"  s{seed:2d}: argmin={al1} random={rl1} -> {tag}", flush=True)
        results.append(dict(seed=seed, argmin=al1, random=rl1, tag=tag))

    print(f"\n{'='*60}")
    from collections import Counter
    tag_counts = Counter(r['tag'] for r in results)
    print(f"  argmin_only: {tag_counts['argmin_only']}")
    print(f"  random_only: {tag_counts['random_only']}")
    print(f"  both:        {tag_counts['both']}")
    print(f"  neither:     {tag_counts['neither']}")

    if tag_counts['random_only'] > 0:
        print(f"\nFINDING: argmin PREVENTS {tag_counts['random_only']} solutions random finds")
    elif tag_counts['argmin_only'] > tag_counts['both']:
        print(f"\nFINDING: argmin provides genuine advantage over random")
    elif tag_counts['argmin_only'] + tag_counts['both'] == tag_counts['both'] + tag_counts['random_only']:
        print(f"\nFINDING: argmin and random reach L1 via similar paths")
    else:
        print(f"\nFINDING: mixed — both methods succeed/fail differently")


if __name__ == "__main__":
    main()
