"""
Step 549 — Recode trajectory divergence: does refinement change WHERE the agent explores?

Two agents, same H matrix, same seed=0 on LS20:
  1. Recode: LSH k=16 + refinement (REFINE_EVERY=5000)
  2. Plain LSH: same Recode class but REFINE_EVERY=999999 (refinement disabled)

At checkpoints (50K, 100K, 200K, 300K steps), log:
  - Cell sets of each agent (live node IDs)
  - Cells unique to Recode
  - Cells unique to Plain LSH
  - Jaccard overlap = |R ∩ L| / |R ∪ L|

Predictions:
  - 50K: overlap > 90% (refinements few, same trajectory)
  - 200K: overlap 70-80% (Recode children diverge from LSH cells)
  - 500K: overlap < 60% (diverged trajectories)

Kill: overlap stays > 90% at all checkpoints → refinement cosmetic only.
Better: overlap < 70% by 200K → refinement redirects exploration.

5-min cap. 1 seed. LS20.
"""
import numpy as np
import time
import sys

N_A = 4
K = 16
DIM = 256
MIN_OBS = 8
H_SPLIT = 0.05


def enc(frame):
    """Avgpool16 + centered."""
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class Recode:

    def __init__(self, dim=DIM, k=K, seed=0, refine_every=5000):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.ns = 0
        self.dim = dim
        self.refine_every = refine_every

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
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k_key = (self._pn, self._pa, n)
            s, c = self.C.get(k_key, (np.zeros(self.dim, np.float64), 0))
            self.C[k_key] = (s + self._px.astype(np.float64), c + 1)
        self._px = x
        self._cn = n
        if self.t > 0 and self.t % self.refine_every == 0:
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

    def stats(self):
        return len(self.live), self.ns, len(self.G)


def jaccard(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 1.0


def t0():
    rng = np.random.RandomState(7)

    # Both agents share same H (same seed, full dim)
    r = Recode(seed=99, refine_every=5000)
    l = Recode(seed=99, refine_every=999999)
    assert np.allclose(r.H, l.H), "H matrices must match"

    # At start, same cell set
    frame = [rng.randint(0, 16, (64, 64))]
    n_r = r.observe(frame); r.act()
    n_l = l.observe(frame); l.act()
    assert n_r == n_l, f"Same frame should map to same node: {n_r} vs {n_l}"
    assert r.live == l.live, "Live sets should match initially"

    j = jaccard(r.live, l.live)
    assert j == 1.0, f"Jaccard should be 1.0 at start, got {j}"

    # Jaccard < 1 after sets diverge
    s1 = {1, 2, 3}
    s2 = {2, 3, 4}
    j2 = jaccard(s1, s2)
    assert abs(j2 - 0.5) < 0.01, f"Jaccard {j2} != 0.5"

    # refine_every=999999 means no refinement fired
    assert l.ns == 0, f"Plain LSH should have 0 splits, got {l.ns}"

    print("T0 PASS")


def run_pair(make_env, seed=0):
    """Run Recode and plain LSH in lockstep on same env sequence."""
    # Both use same H seed
    r = Recode(seed=seed * 1000, refine_every=5000)
    l = Recode(seed=seed * 1000, refine_every=999999)
    assert np.allclose(r.H, l.H), "H matrices must match"

    env = make_env()
    obs = env.reset(seed=seed)
    level = 0
    go = 0
    t_start = time.time()

    checkpoints = {50_000, 100_000, 200_000, 300_000}
    results = []

    for step in range(1, 500_001):
        if obs is None:
            obs = env.reset(seed=seed)
            r.on_reset()
            l.on_reset()
            continue

        # Both agents observe same frame
        r.observe(obs)
        l.observe(obs)

        # Actions are INDEPENDENT (each uses its own graph)
        action_r = r.act()
        action_l = l.act()

        # Use Recode's action to step the env (primary agent)
        # Note: both agents observe, but env follows Recode
        obs, reward, done, info = env.step(action_r)

        if done:
            go += 1
            obs = env.reset(seed=seed)
            r.on_reset()
            l.on_reset()

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            r.on_reset()
            l.on_reset()
            level = cl

        if step in checkpoints:
            j = jaccard(r.live, l.live)
            unique_r = len(r.live - l.live)
            unique_l = len(l.live - r.live)
            shared = len(r.live & l.live)
            nc_r, ns_r, _ = r.stats()
            nc_l, ns_l, _ = l.stats()
            el = time.time() - t_start
            print(f"@{step}: jaccard={j:.3f} shared={shared} "
                  f"unique_r={unique_r} unique_l={unique_l} | "
                  f"recode: c={nc_r} sp={ns_r} | lsh: c={nc_l} sp={ns_l} | "
                  f"{el:.0f}s", flush=True)
            results.append(dict(step=step, jaccard=j, unique_r=unique_r,
                                unique_l=unique_l, shared=shared,
                                cells_r=nc_r, splits_r=ns_r, cells_l=nc_l))

        if time.time() - t_start > 300:
            break

    return results


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        make_env = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}")
        return

    print("\nRunning Recode vs Plain LSH (same H, same env sequence):")
    results = run_pair(make_env, seed=0)

    print(f"\n{'='*60}")
    print(f"{'step':>8} {'jaccard':>8} {'shared':>8} {'uniq_R':>8} {'uniq_L':>8} {'sp_R':>6}")
    for r in results:
        print(f"{r['step']:>8} {r['jaccard']:>8.3f} {r['shared']:>8} "
              f"{r['unique_r']:>8} {r['unique_l']:>8} {r['splits_r']:>6}")

    if not results:
        print("No checkpoints reached.")
        return

    final = results[-1]
    j_final = final['jaccard']

    print(f"\nFinal jaccard @ step {final['step']}: {j_final:.3f}")
    if j_final > 0.90:
        print("KILL: overlap > 90% — refinement cosmetic, same frontier as plain LSH.")
    elif j_final < 0.70:
        print("DIVERGED: overlap < 70% — refinement genuinely redirected exploration.")
    else:
        print(f"PARTIAL: {j_final:.1%} overlap. Moderate trajectory divergence.")

    # Check 200K specifically
    r200 = next((r for r in results if r['step'] == 200_000), None)
    if r200:
        print(f"\n@200K jaccard={r200['jaccard']:.3f}: ", end="")
        if r200['jaccard'] < 0.70:
            print("BETTER THAN PREDICTED — refinement redirects exploration early.")
        elif r200['jaccard'] > 0.80:
            print("Slower divergence than predicted.")
        else:
            print("Within predicted range (70-80%).")


if __name__ == "__main__":
    main()
