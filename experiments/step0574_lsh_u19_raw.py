"""
Step 574 -- U19 validation: LSH k=12 at 64x64 raw on LS20.

U19 (provisional): "Dynamics != features" — codebook-only.
Test: does pure LSH k=12 on raw LS20 frames achieve any levels?

Expected: L1=0. Raw frames change every step (animated sprites, HUD flash).
LSH cells = noise. Argmin navigation = diffuse random walk. No levels.

If U19 is confirmed: dynamics alone don't help, task-relevant features needed.
If challenged: LSH k=12 achieves L1 → U19 may be codebook-specific.

Kill: L1=0 across all seeds → U19 confirmed for LSH family.
Signal: L1>0 in any seed → reconsider U19.

Enc: flatten 64x64 → float [0,1] → center → 12-bit LSH hash.
Nav: argmin over per-cell action visit counts (no BFS, no mode map).
Cap: 3 seeds × 60s each.
"""
import numpy as np
import time
import sys

K = 12
DIM = 4096
N_A = 4


class SubLSH:
    def __init__(self, k=K, dim=DIM, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.G = {}
        self._pn = self._pa = self._cn = None
        self.t = 0
        self.n_cells = set()

    def encode(self, obs):
        """obs[0] is 64x64 uint8 [0-15] -> k-bit LSH hash (int)."""
        x = np.array(obs[0], dtype=np.float32).flatten() / 15.0
        x -= x.mean()
        bits = (self.H @ x > 0).astype(np.uint8)
        return int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)

    def observe(self, obs):
        n = self.encode(obs)
        self.n_cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n
        self.t += 1

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

    def on_reset(self):
        self._pn = None

    def stats(self):
        return len(self.n_cells), len(self.G)


def t0():
    rng = np.random.RandomState(99)
    sub = SubLSH(k=12, dim=4096, seed=0)
    # obs is a tuple: obs[0] = 64x64 uint8 array
    obs1 = [rng.randint(0, 16, (64, 64), dtype=np.uint8)]
    sub.observe(obs1)
    a = sub.act()
    assert 0 <= a < 4
    obs2 = [rng.randint(0, 16, (64, 64), dtype=np.uint8)]
    sub.observe(obs2)
    sub.act()
    cells, edges = sub.stats()
    assert cells >= 1
    print(f"T0 PASS (cells={cells})")


def run_seed(env_factory, seed, time_cap=60):
    env = env_factory()
    sub = SubLSH(k=K, dim=DIM, seed=seed * 1000)
    obs = env.reset(seed=seed)
    sub.on_reset()

    l1 = l2 = l3 = go = 0
    prev_cl = 0
    t_start = time.time()
    step = 0

    while time.time() - t_start < time_cap:
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = 0
            go += 1
            continue

        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)
        step += 1

        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = 0
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl >= 1 and prev_cl < 1:
            l1 += 1
            print(f"  s{seed} L1@{step} go={go}", flush=True)
        if cl >= 2 and prev_cl < 2:
            l2 += 1
            print(f"  s{seed} L2@{step}", flush=True)
        if cl >= 3 and prev_cl < 3:
            l3 += 1
            print(f"  s{seed} L3@{step}", flush=True)
        prev_cl = cl

    cells, edges = sub.stats()
    elapsed = time.time() - t_start
    print(f"  s{seed}: L1={l1} L2={l2} L3={l3} steps={step} go={go} "
          f"cells={cells} edges={edges} {elapsed:.0f}s", flush=True)
    return dict(seed=seed, l1=l1, l2=l2, l3=l3, steps=step, cells=cells)


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3 import failed: {e}")
        return

    results = []
    t_total = time.time()

    for seed in range(3):
        if time.time() - t_total > 270:
            print("TOTAL TIME CAP HIT")
            break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(mk, seed, time_cap=80)
        results.append(r)

    print(f"\n{'='*50}")
    for r in results:
        print(f"  s{r['seed']}: L1={r['l1']} L2={r['l2']} L3={r['l3']} cells={r['cells']}")

    any_l1 = any(r['l1'] > 0 for r in results)
    avg_cells = float(np.mean([r['cells'] for r in results])) if results else 0

    print(f"\navg cells: {avg_cells:.0f}")
    if not any_l1:
        print("U19 CONFIRMED: LSH k=12 raw achieves L1=0. Dynamics != features.")
    else:
        total_l1 = sum(r['l1'] for r in results)
        print(f"U19 CHALLENGED: LSH k=12 raw achieves L1>0 (total={total_l1}). Investigate.")


if __name__ == "__main__":
    main()
