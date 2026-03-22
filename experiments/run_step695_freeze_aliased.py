"""
Step 695 — 674 variant: freeze aliased set at 150.

674 gets regressions when aliased > 180 (s0: 224 aliased, s4: 337 aliased at 25s).
Hypothesis: binary criterion works at < 130 aliased, degrades at > 180.
Fix: freeze the aliased set at 150 — stop adding new aliased cells,
keep using existing ones.

10 seeds (0-9), 25s cap.
Kill: L1 < 9/10 (must match or beat 674's 9/10).
"""
import numpy as np
import time
import sys

K_NAV = 12
K_FINE = 20
DIM = 256
N_A = 4
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MAX_STEPS = 500_001
PER_SEED_TIME = 25
N_SEEDS = 10
MIN_VISITS_ALIAS = 3
ALIAS_CAP = 150  # freeze aliased set at this size


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class TransitionTriggeredFrozen:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
        self.H_fine = rng.randn(K_FINE, DIM).astype(np.float32)
        self.ref = {}; self.G = {}; self.C = {}; self.live = set()
        self.G_fine = {}; self.aliased = set()
        self._pn = self._pa = self._px = None; self._pfn = None
        self.t = 0; self.dim = DIM; self._cn = None; self._fn = None

    def _hash_nav(self, x):
        return int(np.packbits((self.H_nav @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def _hash_fine(self, x):
        return int(np.packbits((self.H_fine @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash_nav(x)
        while n in self.ref: n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        x = enc_frame(frame); n = self._node(x); fn = self._hash_fine(x)
        self.live.add(n); self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)
            succ = self.G.get((self._pn, self._pa), {})
            # Only add to aliased if cap not reached
            if len(self.aliased) < ALIAS_CAP:
                if sum(succ.values()) >= MIN_VISITS_ALIAS and len(succ) >= 2:
                    self.aliased.add(self._pn)
            if self._pn in self.aliased and self._pfn is not None:
                d2 = self.G_fine.setdefault((self._pfn, self._pa), {})
                d2[fn] = d2.get(fn, 0) + 1
        self._px = x; self._cn = n; self._fn = fn
        if self.t % REFINE_EVERY == 0: self._refine()
        return n

    def act(self):
        if self._cn in self.aliased and self._fn is not None:
            best_a, best_s = 0, float('inf')
            for a in range(N_A):
                s = sum(self.G_fine.get((self._fn, a), {}).values())
                if s < best_s: best_s = s; best_a = a
            self._pn = self._cn; self._pfn = self._fn; self._pa = best_a; return best_a
        best_a, best_s = 0, float('inf')
        for a in range(N_A):
            s = sum(self.G.get((self._cn, a), {}).values())
            if s < best_s: best_s = s; best_a = a
        self._pn = self._cn; self._pfn = self._fn; self._pa = best_a; return best_a

    def on_reset(self): self._pn = None; self._pfn = None

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d or sum(d.values()) < 4: return 0.0
        v = np.array(list(d.values()), np.float64); p = v/v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self):
        did = 0
        for (n, a), d in list(self.G.items()):
            if n not in self.live or n in self.ref: continue
            if len(d) < 2 or sum(d.values()) < MIN_OBS: continue
            if self._h(n, a) < H_SPLIT: continue
            top = sorted(d, key=d.get, reverse=True)[:2]
            r0 = self.C.get((n, a, top[0])); r1 = self.C.get((n, a, top[1]))
            if r0 is None or r1 is None or r0[1] < 3 or r1[1] < 3: continue
            diff = (r0[0]/r0[1]) - (r1[0]/r1[1]); nm = np.linalg.norm(diff)
            if nm < 1e-8: continue
            self.ref[n] = (diff/nm).astype(np.float32); self.live.discard(n); did += 1
            if did >= 3: break


# Step 674 baseline L1 at 25s (from Step 690)
BASELINE_674 = {0: 18401, 1: 2847, 2: 20158, 3: 2402, 4: 24968,
                5: 6272, 6: 20397, 7: None, 8: 126, 9: 11407}


def run(seed, make):
    env = make()
    sub = TransitionTriggeredFrozen(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0; l1 = l2 = None
    t_start = time.time()

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset(); continue
        sub.observe(obs); action = sub.act()
        obs, reward, done, info = env.step(action)
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None: l1 = step
            if cl == 2 and l2 is None: l2 = step
            level = cl; sub.on_reset()
        if done:
            obs = env.reset(seed=seed); sub.on_reset()
        if time.time() - t_start > PER_SEED_TIME:
            break

    aliased_final = len(sub.aliased)
    bsl = BASELINE_674.get(seed)
    if l1 and bsl:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    elif l1:
        spd = "no_baseline"
    else:
        spd = "NO_L1"
    print(f"  s{seed:2d}: L1={l1}({spd}) L2={l2} aliased={aliased_final}", flush=True)
    return dict(seed=seed, l1=l1, l2=l2)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 695: 674+freeze(cap={ALIAS_CAP}) on LS20, {N_SEEDS} seeds, {PER_SEED_TIME}s cap")

    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1_n = sum(1 for r in results if r['l1'])
    l2_n = sum(1 for r in results if r['l2'])
    print(f"L1={l1_n}/{N_SEEDS}  L2={l2_n}/{N_SEEDS}")
    if l1_n >= 10: print("FINDING: Freeze fixes regressions — 10/10 beats 674's 9/10")
    elif l1_n >= 9: print("FINDING: Freeze matches 674 (9/10) — no regression fix but no harm")
    else: print(f"KILL: L1={l1_n}/{N_SEEDS} — freeze hurts")


if __name__ == "__main__":
    main()
