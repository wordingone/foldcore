"""
Step 631 — Delta causality probe (diagnostic only).

Track delta_cell distributions per action. No change to action selection.
Report per action:
1. How many unique delta_cells across all nodes?
2. Action-invariant deltas (same delta regardless of node) = "universal effect"?
3. Node-specific deltas (different delta at different nodes) = "context-dependent"?

5 seeds × 60s on LS20. No kill criterion — pure diagnostic.
"""
import numpy as np
import time
import sys
from collections import Counter, defaultdict

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
PER_SEED_TIME = 60


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def enc_delta(curr_frame, prev_frame):
    a = np.array(curr_frame[0], dtype=np.float32) / 15.0
    b = np.array(prev_frame[0], dtype=np.float32) / 15.0
    d = (a - b).reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return d - d.mean()


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
        self.delta_by_na = defaultdict(Counter)  # (n, a) -> Counter of delta_cells
        self.total_transitions = 0

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

            if self._prev_frame is not None:
                dv = enc_delta(frame, self._prev_frame)
                dc = self._hash(dv)
                self._last_dc = dc
                self.delta_by_action[self._pa][dc] += 1
                self.delta_by_na[(self._pn, self._pa)][dc] += 1
                self.total_transitions += 1

        self._px = x
        self._cn = n
        self._prev_frame = frame

        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        # Pure argmin — no delta influence
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
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

    def delta_stats(self):
        """Compute per-action delta statistics."""
        total_all = self.total_transitions
        if total_all == 0:
            return {}

        # Global delta_cell frequencies (across all actions)
        global_ctr = Counter()
        for a in range(N_A):
            global_ctr.update(self.delta_by_action[a])

        stats = {}
        for a in range(N_A):
            ctr = self.delta_by_action[a]
            n_total = sum(ctr.values())
            if n_total == 0:
                stats[a] = dict(n_total=0, n_unique=0)
                continue

            n_unique = len(ctr)
            top_dc, top_count = ctr.most_common(1)[0]
            top_frac = top_count / n_total

            # Action-invariant: does top_dc appear as top choice across most (N, A) pairs?
            na_pairs = [(n, na) for (n, na) in self.delta_by_na if na == a]
            n_nodes_top = sum(
                1 for (n, na) in na_pairs
                if self.delta_by_na[(n, na)].most_common(1)[0][0] == top_dc
            )
            invariant_frac = n_nodes_top / len(na_pairs) if na_pairs else 0.0

            # Context-dependent: how many (N, A) pairs have >1 unique delta_cell?
            n_context_dep = sum(len(self.delta_by_na[(n, na)]) > 1 for (n, na) in na_pairs)

            # Global rank of top_dc (how dominant it is across ALL actions)
            global_frac = global_ctr[top_dc] / total_all

            stats[a] = dict(
                n_total=n_total,
                n_unique=n_unique,
                top_dc=hex(top_dc),
                top_frac=top_frac,
                invariant_frac=invariant_frac,
                n_na_pairs=len(na_pairs),
                n_context_dep=n_context_dep,
                global_frac=global_frac,
            )
        return stats


def t0():
    rng = np.random.RandomState(42)
    sub = Recode(seed=0)

    # Test _hash returns int
    v = rng.randn(DIM).astype(np.float32)
    dc = sub._hash(v)
    assert isinstance(dc, int), f"_hash should return int, got {type(dc)}"

    # Test enc_delta: same frame → zero
    f1 = [rng.randint(0, 16, (64, 64)).tolist()]
    d_same = enc_delta(f1, f1)
    assert np.allclose(d_same, 0.0), "same-frame delta should be zero"

    # Test enc_delta: different frames → non-trivial
    f2 = [rng.randint(0, 16, (64, 64)).tolist()]
    d_diff = enc_delta(f2, f1)
    assert len(d_diff) == DIM, f"delta should be {DIM}-dim, got {len(d_diff)}"

    # Test counter tracking via delta_stats
    sub2 = Recode(seed=0)
    dc_val = sub2._hash(d_diff)
    sub2.delta_by_action[1][dc_val] = 5
    sub2.delta_by_na[(99, 1)][dc_val] = 5
    sub2.total_transitions = 5
    stats = sub2.delta_stats()
    assert stats[1]['n_total'] == 5
    assert stats[1]['n_unique'] == 1
    assert stats[1]['invariant_frac'] == 1.0  # only 1 (N, A) pair, top_dc matches

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

        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset()

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            sub.on_reset()
            if cl == 1 and l1 is None:
                l1 = step
            if cl == 2 and l2 is None:
                l2 = step
            level = cl

        if time.time() - t_start > PER_SEED_TIME:
            break

    return dict(
        seed=seed, l1=l1, l2=l2, go=go,
        total_transitions=sub.total_transitions,
        delta_stats=sub.delta_stats(),
    )


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
        r = run(seed, mk)
        R.append(r)

        l1_str = f"@{r['l1']}" if r['l1'] else "NONE"
        print(f"  s{seed}: L1{l1_str} go={r['go']} transitions={r['total_transitions']}", flush=True)

        ds = r['delta_stats']
        for a in range(N_A):
            s = ds.get(a, {})
            if s.get('n_total', 0) == 0:
                print(f"  A{a}: no data", flush=True)
            else:
                print(
                    f"  A{a}: unique={s['n_unique']:4d} top_frac={s['top_frac']:.3f} "
                    f"invariant={s['invariant_frac']:.3f} "
                    f"ctx_dep={s['n_context_dep']}/{s['n_na_pairs']} "
                    f"global_frac={s['global_frac']:.3f}",
                    flush=True
                )

    print(f"\n{'='*60}")
    l1n = sum(1 for r in R if r['l1'])
    print(f"L1={l1n}/5  (diagnostic — no kill criterion)")

    print(f"\nCross-seed per-action summary:")
    for a in range(N_A):
        uniq = [r['delta_stats'].get(a, {}).get('n_unique', 0) for r in R]
        inv = [r['delta_stats'].get(a, {}).get('invariant_frac', 0.0) for r in R]
        ctx = [r['delta_stats'].get(a, {}).get('n_context_dep', 0) for r in R]
        na = [r['delta_stats'].get(a, {}).get('n_na_pairs', 0) for r in R]
        top_f = [r['delta_stats'].get(a, {}).get('top_frac', 0.0) for r in R]
        print(
            f"  A{a}: avg_unique={np.mean(uniq):.0f} avg_top_frac={np.mean(top_f):.3f} "
            f"avg_invariant={np.mean(inv):.3f} "
            f"avg_ctx_dep={np.mean(ctx):.0f}/{np.mean(na):.0f}"
        )

    # Summary signals
    universal = any(
        r['delta_stats'].get(a, {}).get('invariant_frac', 0.0) > 0.8
        for r in R for a in range(N_A)
    )
    context_dep = any(
        r['delta_stats'].get(a, {}).get('n_context_dep', 0) > 0
        for r in R for a in range(N_A)
    )
    # Dominant delta: any action where >50% of transitions share one delta_cell
    dominant = any(
        r['delta_stats'].get(a, {}).get('top_frac', 0.0) > 0.5
        for r in R for a in range(N_A)
    )

    print(f"\nUniversal effects (invariant > 80%): {'YES' if universal else 'no'}")
    print(f"Context-dependent effects present: {'YES' if context_dep else 'no'}")
    print(f"Dominant delta (>50% of transitions): {'YES' if dominant else 'no'}")

    if universal:
        print("SIGNAL: action-invariant delta found — causal structure usable for step 630.")
    elif dominant:
        print("SIGNAL: dominant delta per action — structure present, partially invariant.")
    elif context_dep:
        print("MARGINAL: only context-dependent variation — structure exists but node-specific.")
    else:
        print("INERT: delta_cells appear random — no exploitable structure for step 630.")


if __name__ == "__main__":
    main()
