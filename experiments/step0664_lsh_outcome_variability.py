"""
Step 664 — Outcome variability per cell (diagnostic).

Standard LSH k=12. For each cell, track entropy of successor distribution.
At L1: compare exit cell variability vs average cell variability.
Is exit cell in top 10% by outcome variance?

If exit cell has HIGHER variability: hidden state is detectable from outcomes.
If same/lower: POMDP is opaque from transitions.
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
PER_SEED_TIME = 25
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
        self.dim = dim
        self._cn = None

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
        best_a, best_s = 0, float('inf')
        for a in range(N_A):
            s = sum(self.G.get((self._cn, a), {}).values())
            if s < best_s:
                best_s = s
                best_a = a
        self._pn = self._cn
        self._pa = best_a
        return best_a

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
            did += 1
            if did >= 3:
                break


def cell_outcome_entropy(G, cell):
    """Average entropy of successor distribution across all actions."""
    ents = []
    for a in range(N_A):
        d = G.get((cell, a), {})
        if sum(d.values()) < 2:
            continue
        v = np.array(list(d.values()), np.float64)
        p = v / v.sum()
        ents.append(float(-np.sum(p * np.log2(np.maximum(p, 1e-15)))))
    return np.mean(ents) if ents else 0.0


def run(seed, make):
    env = make()
    sub = Recode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = None
    exit_cell = None
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
                exit_cell = sub._cn
            level = cl
            sub.on_reset()
        if done:
            obs = env.reset(seed=seed)
            sub.on_reset()
        if time.time() - t_start > PER_SEED_TIME:
            break

    if l1 is None or exit_cell is None:
        print(f"  s{seed:2d}: L1=None", flush=True)
        return dict(seed=seed, l1=None)

    # Compute entropy for all live cells
    all_ents = {n: cell_outcome_entropy(sub.G, n) for n in sub.live}
    exit_ent = all_ents.get(exit_cell, 0.0)
    avg_ent = float(np.mean(list(all_ents.values()))) if all_ents else 0.0

    # Rank exit cell (0 = highest entropy)
    sorted_ents = sorted(all_ents.values(), reverse=True)
    if exit_cell in all_ents and len(sorted_ents) > 0:
        rank = sorted_ents.index(all_ents[exit_cell])
        pct = rank / len(sorted_ents) * 100
    else:
        pct = None

    # Count cells with higher entropy
    n_higher = sum(1 for e in all_ents.values() if e > exit_ent)
    top10 = pct is not None and pct <= 10.0

    pct_str = f"{pct:.0f}%" if pct is not None else "N/A"
    print(f"  s{seed:2d}: L1={l1} exit_ent={exit_ent:.3f} avg={avg_ent:.3f} "
          f"pct={pct_str} top10={top10} n_cells={len(all_ents)}", flush=True)
    return dict(seed=seed, l1=l1, exit_ent=exit_ent, avg_ent=avg_ent,
                pct=pct, top10=top10)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Outcome variability per cell: {N_SEEDS} seeds, {PER_SEED_TIME}s cap")

    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1_results = [r for r in results if r['l1']]
    if not l1_results:
        print("No L1 reached — cannot analyze exit cell variability")
        return

    exit_ents = [r['exit_ent'] for r in l1_results]
    avg_ents = [r['avg_ent'] for r in l1_results]
    top10_count = sum(1 for r in l1_results if r.get('top10'))

    print(f"L1={len(l1_results)}/{N_SEEDS}")
    print(f"Exit cell entropy: avg={np.mean(exit_ents):.3f}")
    print(f"All cells entropy: avg={np.mean(avg_ents):.3f}")
    print(f"Ratio exit/avg: {np.mean(exit_ents)/max(np.mean(avg_ents),1e-8):.2f}x")
    print(f"Exit in top 10%: {top10_count}/{len(l1_results)}")

    ratio = np.mean(exit_ents) / max(np.mean(avg_ents), 1e-8)
    if ratio > 1.5 and top10_count > len(l1_results) // 2:
        print("\nFINDING: EXIT CELL IS HIGH-VARIABILITY — hidden state detectable from outcomes")
    elif ratio < 0.8:
        print("\nFINDING: EXIT CELL IS LOW-VARIABILITY — POMDP is opaque, hidden state invisible")
    else:
        print(f"\nFINDING: EXIT CELL IS AVERAGE VARIABILITY (ratio={ratio:.2f}x) — weak POMDP signal")


if __name__ == "__main__":
    main()
