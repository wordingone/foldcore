"""
Step 694 — Regression diagnosis: plain k=12 on seeds 0 and 4.

674 showed regressions: s0=18401 (vs baseline 1362, 13.5x slower),
s4=24968 (vs baseline 846, 29.5x slower).

Run plain k=12 argmin (NO fine hash, no refinement) on seeds 0 and 4
at 120K steps to verify the baseline numbers from Step 485.

Kill: if baseline matches 674 (regression is noise, not mechanism).
"""
import numpy as np
import time
import sys

K = 12
DIM = 256
N_A = 4
MAX_STEPS = 120_001
SEEDS = [0, 4]

# Step 485 baseline for these seeds
BASELINE_485 = {0: 1362, 4: 846}


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class PlainArgmin:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.H = rng.randn(K, DIM).astype(np.float32)
        self.G = {}
        self._pn = self._pa = self._px = None
        self._cn = None

    def _hash(self, x):
        return int(np.packbits((self.H @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def observe(self, frame):
        x = enc_frame(frame)
        n = self._hash(x)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._px = x; self._cn = n
        return n

    def act(self):
        best_a, best_s = 0, float('inf')
        for a in range(N_A):
            s = sum(self.G.get((self._cn, a), {}).values())
            if s < best_s: best_s = s; best_a = a
        self._pn = self._cn; self._pa = best_a; return best_a

    def on_reset(self): self._pn = None


def run(seed, make):
    env = make()
    sub = PlainArgmin(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0; l1 = None
    t_start = time.time()

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset(); continue
        sub.observe(obs); action = sub.act()
        obs, reward, done, info = env.step(action)
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None: l1 = step
            level = cl; sub.on_reset()
        if done:
            obs = env.reset(seed=seed); sub.on_reset()

    elapsed = time.time() - t_start
    bsl = BASELINE_485.get(seed)
    match = "MATCHES" if l1 and bsl and abs(l1 - bsl) < 100 else "DIFFERS"
    print(f"  s{seed}: L1={l1} baseline={bsl} [{match}] t={elapsed:.1f}s", flush=True)
    return dict(seed=seed, l1=l1, baseline=bsl)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 694: Plain k=12 argmin on seeds {SEEDS}, 120K steps (regression verification)")

    results = []
    for seed in SEEDS:
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    for r in results:
        s0_674 = {0: 18401, 4: 24968}
        ratio_674 = s0_674[r['seed']] / r['l1'] if r['l1'] else None
        ratio_str = f"674/{r['seed']}:{s0_674[r['seed']]} plain:{r['l1']} = {ratio_674:.1f}x slower in 674" if ratio_674 else ""
        print(f"  s{r['seed']}: plain_L1={r['l1']} {ratio_str}")
    print()
    if all(r['l1'] and r['baseline'] and abs(r['l1'] - r['baseline']) < 200 for r in results):
        print("CONFIRMED: Baseline verified — 674 regression is real mechanism effect")
    else:
        print("DISCREPANCY: Baseline doesn't match Step 485 — check for environment changes")


if __name__ == "__main__":
    main()
