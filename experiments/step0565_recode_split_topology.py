"""
Step 565: Q7 — Does metric refinement rearrange topology?

At each split, compare the observations retroactively assigned to child0 vs child1.
Inter-child cosine similarity:
  HIGH (>0.9) = split separated visually similar obs -> rearrangement (U20 violation)
  LOW (<0.5)  = split separated visually distinct obs -> refinement (U20 preserved)

Prediction: ~50/50 rearrange vs refine.
5-min cap. LS20 seed=0. 200K steps.
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
OBS_BUFFER = 20   # recent observations to keep per node

REARRANGE_THRESH = 0.9
REFINE_THRESH = 0.5


def enc(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def cosine_sim_batch(A, B):
    """Compute mean cosine similarity between all pairs in A x B."""
    if len(A) == 0 or len(B) == 0:
        return 0.0
    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)
    na = np.linalg.norm(A, axis=1, keepdims=True)
    nb = np.linalg.norm(B, axis=1, keepdims=True)
    na = np.maximum(na, 1e-10)
    nb = np.maximum(nb, 1e-10)
    An = A / na
    Bn = B / nb
    sims = An @ Bn.T  # (|A|, |B|)
    return float(sims.mean())


class RecodeTopology:
    """Recode + split-topology analysis."""

    def __init__(self, dim=DIM, k=K, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self._cn = None
        self.t = 0
        self.dim = dim
        # Track recent observations per node
        self.node_obs = {}   # node -> list of recent x vectors (capped at OBS_BUFFER)
        # Split analysis results
        self.split_results = []  # list of {parent, inter_cos, n_c0, n_c1, refine_time}
        self.ns = 0  # split count

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
        # Accumulate recent obs
        buf = self.node_obs.setdefault(n, [])
        buf.append(x)
        if len(buf) > OBS_BUFFER:
            buf.pop(0)
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
            direction = (diff / nm).astype(np.float32)
            self.ref[n] = direction
            self.live.discard(n)
            self.ns += 1

            # ---- Split topology analysis ----
            obs_at_parent = self.node_obs.get(n, [])
            if len(obs_at_parent) >= 4:
                # Retroactively assign each obs to child0 or child1
                child0_obs = []
                child1_obs = []
                for x_obs in obs_at_parent:
                    if float(direction @ x_obs) > 0:
                        child1_obs.append(x_obs)
                    else:
                        child0_obs.append(x_obs)
                if child0_obs and child1_obs:
                    inter_cos = cosine_sim_batch(child0_obs, child1_obs)
                    self.split_results.append({
                        'parent': n,
                        'inter_cos': inter_cos,
                        'n_c0': len(child0_obs),
                        'n_c1': len(child1_obs),
                        'refine_time': self.t
                    })


def t0():
    rng = np.random.RandomState(0)

    # Test cosine_sim_batch
    A = [np.ones(10, dtype=np.float32)]
    B = [np.ones(10, dtype=np.float32)]
    assert abs(cosine_sim_batch(A, B) - 1.0) < 1e-5

    A2 = [np.array([1, 0] * 5, dtype=np.float32)]
    B2 = [np.array([0, 1] * 5, dtype=np.float32)]
    assert abs(cosine_sim_batch(A2, B2)) < 1e-5

    # Test RecodeTopology
    sub = RecodeTopology(seed=0)
    for _ in range(5):
        frame = [rng.randint(0, 16, (64, 64))]
        sub.observe(frame)
        sub.act()
    assert sub.t == 5
    assert len(sub.node_obs) > 0

    print("T0 PASS")


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        env = arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}")
        return

    sub = RecodeTopology(seed=0)
    obs = env.reset(seed=0)
    t_start = time.time()
    level = 0
    go = 0
    l1_step = None

    print("Running 200K steps for split analysis...", flush=True)
    for step in range(1, 200_001):
        if obs is None:
            obs = env.reset(seed=0)
            sub.on_reset()
            continue
        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)
        if done:
            go += 1
            obs = env.reset(seed=0)
            sub.on_reset()
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level = cl
            sub.on_reset()
            if cl == 1 and l1_step is None:
                l1_step = step
                print(f"  L1 at step={step}", flush=True)
        if step % 50_000 == 0:
            el = time.time() - t_start
            print(f"  @{step} splits={sub.ns} analyzed={len(sub.split_results)} {el:.0f}s",
                  flush=True)
        if time.time() - t_start > 260:
            print(f"  Timeout at step={step}")
            break

    elapsed = time.time() - t_start
    print(f"\nDone: {elapsed:.0f}s, splits={sub.ns}, analyzed={len(sub.split_results)}")

    if not sub.split_results:
        print("No splits had enough obs to analyze.")
        return

    # ---- Analysis ----
    cos_vals = [r['inter_cos'] for r in sub.split_results]
    arr = np.array(cos_vals)

    rearrange = sum(1 for c in cos_vals if c > REARRANGE_THRESH)
    refine = sum(1 for c in cos_vals if c < REFINE_THRESH)
    middle = len(cos_vals) - rearrange - refine

    print(f"\n=== Split topology analysis ===")
    print(f"Total analyzed splits: {len(cos_vals)}")
    print(f"Inter-child cosine: mean={arr.mean():.4f} std={arr.std():.4f} "
          f"min={arr.min():.4f} max={arr.max():.4f}")
    print(f"\nRearrangement (cos > {REARRANGE_THRESH}): {rearrange} ({rearrange/len(cos_vals):.1%})")
    print(f"Refinement (cos < {REFINE_THRESH}): {refine} ({refine/len(cos_vals):.1%})")
    print(f"Ambiguous: {middle} ({middle/len(cos_vals):.1%})")

    # Distribution
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.01]
    print(f"\nCosine distribution:")
    for i in range(len(bins) - 1):
        cnt = sum(1 for c in cos_vals if bins[i] <= c < bins[i + 1])
        print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {cnt} ({cnt/len(cos_vals):.1%})")

    # Decay: does rearrangement rate change over time?
    n = len(sub.split_results)
    if n >= 4:
        print(f"\nRearrangement rate by time quartile:")
        sub.split_results.sort(key=lambda r: r['refine_time'])
        for q in range(4):
            lo = q * n // 4
            hi = (q + 1) * n // 4
            chunk = sub.split_results[lo:hi]
            re = sum(1 for r in chunk if r['inter_cos'] > REARRANGE_THRESH)
            t_lo = chunk[0]['refine_time']
            t_hi = chunk[-1]['refine_time']
            print(f"  Q{q+1} (steps {t_lo}-{t_hi}): {re}/{len(chunk)} rearrange "
                  f"({re/len(chunk):.1%})")

    print(f"\n=== Summary ===")
    print(f"Prediction: ~50% rearrange, ~50% refine")
    rr = rearrange / len(cos_vals)
    rf = refine / len(cos_vals)
    if rr > 0.3 and rf > 0.3:
        print(f"CONFIRMED: Mixed. {rr:.0%} rearrange, {rf:.0%} refine. "
              f"Some splits violate U20 Lipschitz property.")
    elif rr > 0.5:
        print(f"REARRANGEMENT-DOMINANT: {rr:.0%}. Most splits are U20 violations.")
    elif rf > 0.7:
        print(f"REFINEMENT-DOMINANT: {rf:.0%}. Most splits preserve U20.")
    else:
        print(f"Rearrange={rr:.0%} Refine={rf:.0%} Ambiguous={1-rr-rf:.0%}. Middle range.")


if __name__ == "__main__":
    main()
