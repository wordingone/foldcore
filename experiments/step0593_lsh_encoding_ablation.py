"""
Step 593 -- Encoding ablation: no centering (U16 confirmation).

U16 (validated universal): centering is load-bearing.
Step 412 (3 seeds): 0/3 without centering.
Step 544 (5 seeds): 0/5 without centering, only 62 cells.

This is the definitive 5-seed confirmation with SoftPenalty (581d) as the vehicle.

Two conditions, 5 seeds, 10K steps:
  A) SoftPenalty K=12, standard enc (WITH centering) -- x -= x.mean()
  B) SoftPenalty K=12, uncentered (WITHOUT centering) -- raw pooling output

Key point: if centered gets 3/5 and uncentered gets 0/5, centering is a strong paper point.
Simple preprocessing = critical component. Cite as evidence for U16.
"""
import numpy as np
import time
import sys

N_A = 4
PENALTY = 100
DIM = 256
MAX_STEPS = 10_000
TIME_CAP = 60
N_SEEDS = 5


def enc_centered(frame):
    """Standard encoding with centering (x -= x.mean())."""
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def enc_uncentered(frame):
    """Uncentered encoding -- raw pooling output, no mean subtraction."""
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x   # NO centering


def lsh_hash(x, H):
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(len(bits))))


class SoftPenalty:
    def __init__(self, seed=0, enc_fn=enc_centered):
        self.H = np.random.RandomState(seed).randn(12, DIM).astype(np.float32)
        self.G = {}
        self.death_edges = set()
        self._pn = self._pa = self._cn = None
        self.cells = set()
        self.total_deaths = 0
        self.enc = enc_fn

    def observe(self, frame):
        x = self.enc(frame)
        n = lsh_hash(x, self.H)
        self.cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def act(self):
        counts = np.array(
            [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)],
            dtype=np.float64
        )
        for a in range(N_A):
            if (self._cn, a) in self.death_edges:
                counts[a] += PENALTY
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

    def on_death(self):
        if self._pn is not None:
            self.death_edges.add((self._pn, self._pa))
            self.total_deaths += 1

    def on_reset(self):
        self._pn = None


def run_seed(mk, seed, enc_fn):
    env = mk()
    sub = SoftPenalty(seed=seed * 100 + 7, enc_fn=enc_fn)
    obs = env.reset(seed=seed)
    sub.on_reset()

    l1 = go = step = 0
    prev_cl = 0
    fresh = True
    t0 = time.time()

    while step < MAX_STEPS and time.time() - t0 < TIME_CAP:
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue
        sub.observe(obs)
        action = sub.act()
        obs, _, done, info = env.step(action)
        step += 1
        if done:
            sub.on_death()
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if fresh:
            prev_cl = cl; fresh = False
        elif cl >= 1 and prev_cl < 1:
            l1 += 1
            if l1 <= 2:
                print(f"    s{seed} L1@{step}", flush=True)
        prev_cl = cl

    print(f"  s{seed}: L1={l1} go={go} step={step} cells={len(sub.cells)} "
          f"deaths={sub.total_deaths} {time.time()-t0:.0f}s", flush=True)
    return dict(seed=seed, l1=l1, go=go, steps=step,
                cells=len(sub.cells), deaths=sub.total_deaths)


def run_condition(mk, label, enc_fn):
    print(f"\n--- {label} ---", flush=True)
    results = []
    for seed in range(N_SEEDS):
        try:
            results.append(run_seed(mk, seed, enc_fn))
        except Exception as e:
            print(f"  s{seed} FAIL: {e}", flush=True)
    wins = sum(1 for r in results if r['l1'] > 0)
    l1 = sum(r['l1'] for r in results)
    avg_cells = np.mean([r['cells'] for r in results]) if results else 0
    print(f"  {label}: {wins}/{N_SEEDS} L1={l1} avg_cells={avg_cells:.0f}", flush=True)
    return wins, l1, avg_cells


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 593: Encoding ablation -- centering load-bearing test on LS20", flush=True)
    print(f"  {N_SEEDS} seeds x {MAX_STEPS} steps | SoftPenalty K=12", flush=True)

    t_total = time.time()
    c_wins, c_l1, c_cells = run_condition(mk, "Centered (standard)", enc_centered)
    u_wins, u_l1, u_cells = run_condition(mk, "Uncentered (ablation)", enc_uncentered)

    print(f"\n{'='*60}", flush=True)
    print(f"Step 593: Encoding ablation (U16 confirmation)", flush=True)
    print(f"  Centered:   {c_wins}/{N_SEEDS} L1={c_l1} avg_cells={c_cells:.0f}", flush=True)
    print(f"  Uncentered: {u_wins}/{N_SEEDS} L1={u_l1} avg_cells={u_cells:.0f}", flush=True)

    if u_wins == 0 and c_wins > 0:
        print(f"\n  U16 CONFIRMED: Centering is load-bearing.", flush=True)
        print(f"  Without x-=x.mean(), substrate cannot navigate.", flush=True)
        print(f"  Strong paper point: simple preprocessing = critical component.", flush=True)
    elif u_wins > 0:
        print(f"\n  U16 CHALLENGE: Uncentered gets {u_wins}/5. Centering may not be essential at 10K.", flush=True)
    else:
        print(f"\n  BOTH 0/5: Budget too small for this ablation. Repeat at 50K?", flush=True)

    print(f"\n  Total elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
