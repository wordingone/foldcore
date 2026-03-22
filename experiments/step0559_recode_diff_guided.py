"""
Step 559 — Frame-diff guided Recode: skip blocked (node, action) pairs.

Step 558 finding: frame_diff < 0.082 = blocked (agent against wall, wasted step).
37% of steps are wasted. Skip them -> more efficient exploration within 129-step budget.

Modified act(): argmin over NON-blocked actions. Fallback to pure argmin if all blocked.
5-min global cap. LS20. Up to 5 seeds.

Predictions:
  L1: 5/5 (fewer wasted steps, reaches exit faster)
  L2: 1/5 (maybe — more efficient exploration within energy budget)
  L1 step count: < 15K (vs 15164 baseline — skip-blocked saves steps)

Kill: L1 < 3/5 -> filter hurts navigation.
Find: L2 >= 1/5 -> efficiency gain enough to reach energy palettes.
"""
import numpy as np
import time
import sys

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 2000
MIN_OBS = 4
H_SPLIT = 0.05
DIFF_THRESH = 0.082  # from Step 558: valley between populations


def enc(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class RecodeGuided:
    """Recode with frame-diff blocked-action avoidance."""

    def __init__(self, dim=DIM, k=K, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self._cn = None
        self.t = 0
        self.ns = 0
        self.dim = dim
        self._last_visit = {}
        self.blocked = set()      # (node, action) marked as wall-hits
        self.block_count = 0      # total marks applied
        self.fallback_count = 0   # times all actions were blocked

    def _base(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._base(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        x = enc(frame)
        n = self._node(x)
        self.live.add(n)
        self.t += 1
        self._last_visit[n] = self.t
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k_key = (self._pn, self._pa, n)
            s, c = self.C.get(k_key, (np.zeros(self.dim, np.float64), 0))
            self.C[k_key] = (s + self._px.astype(np.float64), c + 1)
        self._px = x
        self._cn = n
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        """Argmin over non-blocked actions. Fallback to all if all blocked."""
        unblocked = [a for a in range(N_A) if (self._cn, a) not in self.blocked]
        if not unblocked:
            unblocked = list(range(N_A))
            self.fallback_count += 1
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in unblocked]
        action = unblocked[int(np.argmin(counts))]
        self._pn = self._cn
        self._pa = action
        return action

    def mark_blocked(self, node, action):
        if (node, action) not in self.blocked:
            self.blocked.add((node, action))
            self.block_count += 1

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
            if r0 is None or r1 is None or r0[1] < 2 or r1[1] < 2:
                continue
            diff = (r0[0] / r0[1]) - (r1[0] / r1[1])
            nm = np.linalg.norm(diff)
            if nm < 1e-8:
                continue
            self.ref[n] = (diff / nm).astype(np.float32)
            self.live.discard(n)
            self.ns += 1

    def active_set(self, window=100_000):
        cutoff = self.t - window
        return sum(1 for v in self._last_visit.values() if v >= cutoff)

    def stats(self):
        return len(self.live), self.ns, len(self.G)


def t0():
    rng = np.random.RandomState(0)
    f1 = [rng.randint(0, 16, (64, 64))]
    f2 = [rng.randint(0, 16, (64, 64))]

    sub = RecodeGuided(seed=0)

    # observe returns node
    n1 = sub.observe(f1)
    assert n1 is not None
    assert sub._px is not None and sub._px.shape == (256,)

    # act works
    sub.act()

    # mark_blocked works
    node = sub._cn
    sub.mark_blocked(node, 0)
    assert (node, 0) in sub.blocked
    assert sub.block_count == 1

    # act() skips blocked action
    sub._cn = node
    for _ in range(20):
        a = sub.act()
        # Should not pick action 0 (blocked) unless all blocked
        if (node, 1) not in sub.blocked and (node, 2) not in sub.blocked and (node, 3) not in sub.blocked:
            assert a != 0, f"Should skip blocked action 0, got {a}"

    # fallback: mark all actions blocked
    for a in range(N_A):
        sub.mark_blocked(node, a)
    sub._cn = node
    fc_before = sub.fallback_count
    action = sub.act()
    assert sub.fallback_count == fc_before + 1, "Should increment fallback"
    assert action in range(N_A), "Fallback returns valid action"

    # enc diff
    x1 = enc(f1)
    x2 = enc(f2)
    diff = float(np.linalg.norm(x2 - x1))
    assert diff > 0.0

    print("T0 PASS")


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}")
        return

    n_seeds = 5
    global_cap = 280
    R = []
    t_start = time.time()
    for seed in range(n_seeds):
        elapsed = time.time() - t_start
        if elapsed > global_cap - 10:
            print(f"\nGlobal cap hit at seed {seed}", flush=True)
            break
        remaining = global_cap - elapsed
        # Allocate remaining time equally across remaining seeds
        seeds_left = n_seeds - seed
        budget = remaining / seeds_left
        print(f"\nseed {seed} (budget={budget:.0f}s):", flush=True)

        env = mk()
        sub = RecodeGuided(seed=seed * 1000)
        obs = env.reset(seed=seed)
        level = 0
        l1 = l2 = None
        go = 0
        deadline = time.time() + budget

        for step in range(1, 500_001):
            if obs is None:
                obs = env.reset(seed=seed)
                sub.on_reset()
                continue

            sub.observe(obs)
            prev_node = sub._cn
            prev_x = sub._px
            action = sub.act()

            obs, reward, done, info = env.step(action)

            if obs is not None and prev_x is not None:
                x_new = enc(obs)
                diff = float(np.linalg.norm(x_new - prev_x))
                if diff < DIFF_THRESH:
                    sub.mark_blocked(prev_node, action)

            if done:
                go += 1
                obs = env.reset(seed=seed)
                sub.on_reset()

            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                level = cl
                sub.on_reset()
                if cl == 1 and l1 is None:
                    l1 = step
                    nc, ns, ne = sub.stats()
                    print(f"  s{seed} L1@{step} c={nc} sp={ns} go={go} "
                          f"blocked={sub.block_count} fb={sub.fallback_count}", flush=True)
                if cl == 2 and l2 is None:
                    l2 = step
                    nc, ns, ne = sub.stats()
                    print(f"  s{seed} L2@{step} c={nc} sp={ns} go={go} "
                          f"blocked={sub.block_count} fb={sub.fallback_count}", flush=True)

            if step % 50_000 == 0:
                nc, ns, ne = sub.stats()
                ac = sub.active_set()
                el = time.time() - t_start
                print(f"  s{seed} @{step} c={nc} sp={ns} ac={ac} go={go} "
                      f"blocked={sub.block_count} fb={sub.fallback_count} {el:.0f}s", flush=True)

            if time.time() > deadline:
                break

        nc, ns, ne = sub.stats()
        ac = sub.active_set()
        R.append(dict(seed=seed, l1=l1, l2=l2, cells=nc, splits=ns, go=go,
                      steps=step, active=ac, blocked=sub.block_count,
                      fallback=sub.fallback_count, level=level))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"\nResults (DIFF_THRESH={DIFF_THRESH}):")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        print(f"  s{r['seed']}: {tag:>3}  c={r['cells']:>5}  sp={r['splits']:>4}  "
              f"active={r['active']:>4}  go={r['go']:>4}  steps={r['steps']:>7}  "
              f"blocked={r['blocked']:>5}  fb={r['fallback']:>4}")

    l1n = sum(1 for r in R if r['l1'])
    l2n = sum(1 for r in R if r['l2'])
    if not R:
        print("No results.")
        return

    mc = max(r['cells'] for r in R)
    l1_steps = [r['l1'] for r in R if r['l1']]
    avg_l1 = np.mean(l1_steps) if l1_steps else None

    print(f"\nL1={l1n}/{len(R)}  L2={l2n}/{len(R)}  max_cells={mc}")
    print(f"Baseline (Step 554): L1=3/3 at step~15K, L2=0/3")
    if avg_l1:
        print(f"Avg L1 step: {avg_l1:.0f} (baseline: 15164)")

    if l2n > 0:
        print(f"\nFIND: L2={l2n}/{len(R)}. Frame-diff guidance reaches L2!")
    elif l1n >= 3:
        print(f"\nL1={l1n}/{len(R)}. Frame-diff guidance preserved L1 ability.")
        print("L2=0: Efficiency gain insufficient. Need object detection, not just wall avoidance.")
    elif l1n < 3:
        print(f"\nKILL: L1={l1n}/{len(R)} < 3. Blocked-action filter HURTS navigation.")
        print("Some 'blocked' (low diff) transitions are needed to reach the exit.")


if __name__ == "__main__":
    main()
