"""
Step 579 -- U8 validation: soft blending in LSH on LS20.

U8 (provisional): "Hard selection over soft blending."
Evidence so far: codebook soft blending -> centroid convergence. Both LSH and
graph use hard argmin. Neural networks use soft operations successfully.
May be navigation-specific, not universal.

Test: LSH k=12 on LS20, softmax temperature T=0.5 on action visit counts.
Softmax(-count/T): gives highest prob to least-visited action.
T=0.5 = moderate softening (not full argmin, not uniform random).

Baseline: argmin LSH k=12 = 6/10 L1 at 50K (step 459), 9/10 at 120K (step 485).
Kill: 0/5 at 50K -> U8 CONFIRMED for LSH family.
Signal: >=3/5 -> U8 may be codebook-specific.

Also test T=0.1 (near-argmin) and T=5.0 (near-uniform) for comparison.
5 seeds x 50K steps, 5-min total cap.
"""
import time
import numpy as np
import sys

K = 12
DIM = 256
N_A = 4
MAX_STEPS = 50_000
TIME_CAP = 55     # seconds per seed


def enc_ls20(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class SubLSHSoftmax:
    """LSH k=12 with softmax(-count/T) action selection."""

    def __init__(self, k=K, dim=DIM, seed=0, temp=0.5):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.G = {}
        self._pn = self._pa = self._cn = None
        self.cells = set()
        self.temp = temp

    def observe(self, frame):
        x = enc_ls20(frame)
        n = int(np.packbits((self.H @ x > 0).astype(np.uint8),
                             bitorder='big').tobytes().hex(), 16)
        self.cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def act(self):
        counts = np.array([sum(self.G.get((self._cn, a), {}).values())
                           for a in range(N_A)], dtype=np.float32)
        scores = -counts / self.temp
        scores -= scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()
        a = int(np.random.choice(N_A, p=probs))
        self._pn = self._cn; self._pa = a; return a

    def on_reset(self):
        self._pn = None


# ── tests ─────────────────────────────────────────────────────────────────────

def t0():
    sub = SubLSHSoftmax(seed=0, temp=0.5)
    frame = [np.random.RandomState(1).randint(0, 16, (64, 64), dtype=np.uint8)]
    sub.observe(frame)
    a = sub.act()
    assert 0 <= a < 4
    # Verify softmax works: inject counts 0,1,2,3 -> lowest count should dominate
    sub2 = SubLSHSoftmax(seed=99, temp=0.1)
    frame2 = [np.random.RandomState(2).randint(0, 16, (64, 64), dtype=np.uint8)]
    sub2.observe(frame2)
    n = sub2._cn
    # Manually set counts
    for a2 in range(4):
        sub2.G[(n, a2)] = {99: a2}  # action a2 has been visited a2 times
    action_counts = np.zeros(4)
    for _ in range(100):
        sub2._cn = n
        action_counts[sub2.act()] += 1
    # Action 0 (count=0) should dominate with T=0.1
    assert action_counts[0] > 50, f"Expected action 0 to dominate, got {action_counts}"
    print(f"T0 PASS (action_counts={action_counts})")


# ── experiment ────────────────────────────────────────────────────────────────

def run_seed(mk, seed, temp, time_cap=TIME_CAP):
    env = mk()
    sub = SubLSHSoftmax(k=K, dim=DIM, seed=seed * 1000, temp=temp)
    obs = env.reset(seed=seed)
    l1 = l2 = go = 0
    prev_cl = 0; t_start = time.time()
    step = 0

    for step in range(1, MAX_STEPS + 1):
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset(); prev_cl = 0; go += 1; continue

        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)

        if done:
            go += 1; obs = env.reset(seed=seed); sub.on_reset(); prev_cl = 0; continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl >= 1 and prev_cl < 1:
            l1 += 1
            print(f"  T={temp} s{seed} L1@{step} go={go}", flush=True)
        if cl >= 2 and prev_cl < 2:
            l2 += 1
        prev_cl = cl
        if time.time() - t_start > time_cap:
            break

    elapsed = time.time() - t_start
    cells = len(sub.cells)
    print(f"  T={temp} s{seed}: L1={l1} L2={l2} go={go} cells={cells} "
          f"steps={step} {elapsed:.0f}s", flush=True)
    return dict(seed=seed, l1=l1, l2=l2, steps=step, cells=cells, temp=temp)


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    # Test T=0.5 (primary) and T=0.1 (near-argmin) for comparison
    temps = [0.5, 0.1]
    all_results = {}
    t_total = time.time()

    for temp in temps:
        if time.time() - t_total > 270:
            print(f"TOTAL TIME CAP -- skipping T={temp}"); break
        print(f"\n--- T={temp} ---", flush=True)
        results = []
        for seed in range(5):
            if time.time() - t_total > 270:
                print("TOTAL TIME CAP HIT"); break
            r = run_seed(mk, seed, temp, time_cap=TIME_CAP)
            results.append(r)
        all_results[temp] = results

    print(f"\n{'='*50}")
    print("STEP 579: U8 validation -- softmax temperature")
    for temp, results in all_results.items():
        wins = sum(1 for r in results if r['l1'] > 0)
        avg_cells = float(np.mean([r['cells'] for r in results])) if results else 0
        print(f"  T={temp}: {wins}/{len(results)} seeds L1  avg_cells={avg_cells:.0f}")
    print(f"Baseline argmin: 6/10 L1 at 50K (step 459)")

    primary = all_results.get(0.5, [])
    wins_primary = sum(1 for r in primary if r['l1'] > 0)
    if wins_primary == 0:
        print("U8 CONFIRMED: Soft selection (T=0.5) fails. Hard argmin required for LSH.")
    elif wins_primary >= 3:
        print(f"U8 CHALLENGED: T=0.5 navigates {wins_primary}/{len(primary)}. Soft works.")
    else:
        print(f"PARTIAL: T=0.5 gets {wins_primary}/{len(primary)}. Degraded but not dead.")


if __name__ == "__main__":
    main()
