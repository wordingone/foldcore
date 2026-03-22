"""
Step 670 — Alternating exploration (argmin/random every 5 episodes).

Standard LSH k=12 graph. Action selection ALTERNATES:
  episodes 1-5:  argmin
  episodes 6-10: uniform random
  episodes 11-15: argmin
  ... repeat

Both modes use the SAME graph (counts accumulate across both).
Argmin builds the count gradient. Random breaks conjunction lock.

653 showed argmin and random each unlock 3/20 seeds the other misses.
If alternating captures BOTH sets: alternation helps.
If same as argmin: random episodes are wasted.

20 seeds, 5s cap (same as 653).
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
PER_SEED_TIME = 5
N_SEEDS = 20
EPISODE_BLOCK = 5  # switch mode every 5 episodes

ARGMIN_653 = {3: 220, 4: 132, 6: 950, 10: 900}
RANDOM_653 = {7: 8959, 10: 9130, 11: 1034, 15: 3453}


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class AlternatingRecode:
    def __init__(self, k=K, dim=DIM, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.rng = np.random.RandomState(seed * 777 + 42)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.dim = dim
        self._cn = None
        self.episode_count = 0

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
        # Determine mode: block 0,2,4... = argmin; block 1,3,5... = random
        block = (self.episode_count // EPISODE_BLOCK) % 2
        if block == 1:
            action = int(self.rng.randint(N_A))
        else:
            best_a, best_s = 0, float('inf')
            for a in range(N_A):
                s = sum(self.G.get((self._cn, a), {}).values())
                if s < best_s:
                    best_s = s
                    best_a = a
            action = best_a
        self._pn = self._cn
        self._pa = action
        return action

    def on_reset(self):
        self._pn = None
        self.episode_count += 1

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
            did += 1
            if did >= 3:
                break


def run(seed, make):
    env = make()
    sub = AlternatingRecode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = None
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
            obs = env.reset(seed=seed)
            sub.on_reset()
        if time.time() - t_start > PER_SEED_TIME:
            break

    print(f"  s{seed:2d}: L1={l1}", flush=True)
    return dict(seed=seed, l1=l1)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Alternating argmin/random: {N_SEEDS} seeds, {PER_SEED_TIME}s cap, "
          f"block={EPISODE_BLOCK} eps")

    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    alt_success = {r['seed'] for r in results if r['l1']}
    argmin_success = set(ARGMIN_653.keys())
    random_success = set(RANDOM_653.keys())
    union_success = argmin_success | random_success

    alt_only = alt_success - union_success
    union_only = union_success - alt_success
    both = alt_success & union_success

    print(f"alternating: {sorted(alt_success)}")
    print(f"argmin_653:  {sorted(argmin_success)}")
    print(f"random_653:  {sorted(random_success)}")
    print(f"union:       {sorted(union_success)}")
    print(f"alt_only: {sorted(alt_only)}, union_only: {sorted(union_only)}, both: {sorted(both)}")

    if len(alt_success) >= len(union_success):
        print(f"\nFINDING: Alternating captures BOTH argmin and random seeds ({len(alt_success)} >= {len(union_success)})")
    elif len(alt_success) > len(argmin_success):
        print(f"\nFINDING: Alternating beats pure argmin ({len(alt_success)} > {len(argmin_success)}) but not full union")
    elif alt_success == argmin_success:
        print(f"\nFINDING: Alternating = argmin — random episodes wasted")
    else:
        print(f"\nFINDING: Alternating mode INTERFERES ({len(alt_success)} seeds, lost {len(argmin_success - alt_success)} from argmin)")


if __name__ == "__main__":
    main()
