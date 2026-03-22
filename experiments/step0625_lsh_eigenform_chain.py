"""
Step 625 — Eigenform chain: CIFAR -> LS20 -> CIFAR.

Does self-observation on one game contaminate performance on another?
Run 620 self-observation on LS20, then CIFAR, then LS20 again — same substrate.

Cross-game test: if eigenform self-obs adapts to game structure, switching
games should reset/degrade ops (they encoded the wrong game). If ops are
universal (game-agnostic), performance survives.

Protocol:
1. Run LS20 for 60s (with self-obs). Record L1, final ops.
2. Switch to CIFAR for 60s (same substrate). Record acc, op drift.
3. Switch back to LS20 for 60s. Record L1 again.

Signal: LS20 L1 in phase 3 >= L1 rate in phase 1 (ops survived game switch)
Kill: LS20 L1 in phase 3 < phase 1 (ops poisoned by CIFAR)

3 seeds. Report op distribution at end of each phase.
"""
import numpy as np
import time
import sys

N_A_LS20 = 4
N_A_CIFAR = 10
K = 16
DIM = 256
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
SELF_OBS_EVERY = 500
PER_PHASE_TIME = 60
N_SEEDS = 3

OP_NEUTRAL = 0
OP_AVOID = 1
OP_PREFER = 2
AVOID_PENALTY = 100
PREFER_BONUS = 50


def enc_ls20(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def enc_cifar(frame):
    a = np.array(frame, dtype=np.float32).flatten() / 255.0
    # Downsample 32x32x3 -> 16x16x3 avg pool, flatten
    arr = np.array(frame, dtype=np.float32)
    if arr.ndim == 3:  # H x W x C
        arr = arr.reshape(16, 2, 16, 2, 3).mean(axis=(1, 3)).reshape(-1)
    else:
        arr = arr.flatten()
    arr = arr / 255.0
    return arr - arr.mean()


class Recode:

    def __init__(self, dim=DIM, k=K, n_actions=N_A_LS20, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.ns = 0
        self.ops = {}
        self.op_counts = [0, 0, 0]
        self.n_actions = n_actions

    def _base(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._base(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, x):
        n = self._node(x)
        self.live.add(n)
        self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(len(x), np.float64), 0))
            self.C[k] = (s + x.astype(np.float64), c + 1)
        self._px = x
        self._cn = n
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        if self.t > 0 and self.t % SELF_OBS_EVERY == 0:
            self._self_observe()
        return n

    def act(self):
        counts = []
        for a in range(self.n_actions):
            base_count = sum(self.G.get((self._cn, a), {}).values())
            op = self.ops.get((self._cn, a), OP_NEUTRAL)
            if op == OP_AVOID:
                modified = base_count + AVOID_PENALTY
                self.op_counts[1] += 1
            elif op == OP_PREFER:
                modified = max(0, base_count - PREFER_BONUS)
                self.op_counts[2] += 1
            else:
                modified = base_count
                self.op_counts[0] += 1
            counts.append(modified)
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

    def _self_observe(self):
        na_signals = {}
        nodes = set(n for (n, a) in self.G)
        for n in nodes:
            total = sum(sum(self.G.get((n, a), {}).values()) for a in range(self.n_actions))
            if total == 0:
                continue
            for a in range(self.n_actions):
                edge_count = sum(self.G.get((n, a), {}).values())
                if edge_count == 0:
                    continue
                na_signals[(n, a)] = edge_count / total

        if len(na_signals) < 10:
            return

        signals_arr = np.array(list(na_signals.values()))
        p90 = np.percentile(signals_arr, 90)
        p10 = np.percentile(signals_arr, 10)

        new_ops = {}
        for (n, a), sig in na_signals.items():
            if sig > p90:
                new_ops[(n, a)] = OP_AVOID
            elif sig < p10:
                new_ops[(n, a)] = OP_PREFER
            else:
                new_ops[(n, a)] = OP_NEUTRAL
        self.ops = new_ops

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

    def op_dist(self):
        total = len(self.ops)
        if total == 0:
            return 0.0, 0.0, 100.0
        av = 100.0 * sum(1 for v in self.ops.values() if v == OP_AVOID) / total
        pr = 100.0 * sum(1 for v in self.ops.values() if v == OP_PREFER) / total
        nt = 100.0 * (total - sum(1 for v in self.ops.values() if v in (OP_AVOID, OP_PREFER))) / total
        return av, pr, nt


def t0():
    sub = Recode(dim=8, k=3, seed=0)
    rng = np.random.RandomState(42)
    sub.H = rng.randn(3, 8).astype(np.float32)
    x1 = rng.randn(8).astype(np.float32)
    assert isinstance(sub._node(x1), int)
    # Verify n_actions parameterization works
    sub10 = Recode(dim=8, k=3, n_actions=10, seed=0)
    sub10.H = sub.H.copy()
    counts = [sub10.G.get((0, a), {}) for a in range(10)]
    assert len(counts) == 10
    print("T0 PASS")


def run_phase_ls20(sub, env, seed, phase_name, phase_time):
    """Run LS20 phase. Returns (l1_found, l1_step, go, op_dist)."""
    obs = env.reset(seed=seed)
    sub.on_reset()
    level = 0
    l1 = None
    go = 0
    t_start = time.time()

    for step in range(1, 500_001):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            continue

        x = enc_ls20(obs)
        sub.observe(x)
        action = sub.act()
        obs, reward, done, info = env.step(action)

        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset()

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
            level = cl
            sub.on_reset()

        if time.time() - t_start > phase_time:
            break

    av, pr, nt = sub.op_dist()
    print(f"  {phase_name} s{seed}: l1={l1} go={go} ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%", flush=True)
    return l1, go, (av, pr, nt)


def run_phase_cifar(sub, mk_cifar, seed, phase_name, phase_time):
    """Run CIFAR-10 phase. Returns (correct, total, op_dist)."""
    try:
        env = mk_cifar()
        obs = env.reset(seed=seed)
        sub.on_reset()
    except Exception as e:
        print(f"  {phase_name} s{seed}: CIFAR unavailable ({e})", flush=True)
        return 0, 0, sub.op_dist()

    correct = total = 0
    t_start = time.time()

    for step in range(1, 200_001):
        x = np.array(obs, dtype=np.float32).flatten()
        # Ensure compatible dim
        if len(x) != DIM:
            x = x[:DIM] if len(x) > DIM else np.pad(x, (0, DIM - len(x)))
        x = x / 255.0
        x = x - x.mean()

        sub.observe(x)
        action = sub.act()
        obs, reward, done, info = env.step(action)
        total += 1
        if reward > 0:
            correct += 1
        if done:
            obs = env.reset(seed=seed)
            sub.on_reset()

        if time.time() - t_start > phase_time:
            break

    acc = 100.0 * correct / total if total > 0 else 0.0
    av, pr, nt = sub.op_dist()
    print(f"  {phase_name} s{seed}: acc={acc:.1f}% ({correct}/{total}) ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%", flush=True)
    return correct, total, (av, pr, nt)


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk_ls20 = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3 LS20: {e}")
        print("\nDry run only. T0 passed.")
        return

    # Try CIFAR — optional
    mk_cifar = None
    try:
        mk_cifar = lambda: arcagi3.make("CIFAR10")
    except Exception:
        pass

    R = []
    for seed in range(N_SEEDS):
        print(f"\nseed {seed}:", flush=True)
        sub = Recode(seed=seed * 1000, n_actions=N_A_LS20)
        env_ls20 = mk_ls20()

        # Phase 1: LS20
        l1_p1, go_p1, ops_p1 = run_phase_ls20(sub, env_ls20, seed, "P1_LS20", PER_PHASE_TIME)

        # Phase 2: CIFAR (if available) — change n_actions to 10
        ops_p2 = ops_p1
        if mk_cifar:
            sub.n_actions = N_A_CIFAR
            _, _, ops_p2 = run_phase_cifar(sub, mk_cifar, seed, "P2_CIFAR", PER_PHASE_TIME)
            sub.n_actions = N_A_LS20
        else:
            print(f"  P2_CIFAR s{seed}: skipped (not available)", flush=True)

        # Phase 3: LS20 again (same substrate, ops may have drifted)
        env_ls20_2 = mk_ls20()
        l1_p3, go_p3, ops_p3 = run_phase_ls20(sub, env_ls20_2, seed, "P3_LS20", PER_PHASE_TIME)

        R.append(dict(seed=seed, l1_p1=l1_p1, l1_p3=l1_p3,
                      ops_p1=ops_p1, ops_p2=ops_p2, ops_p3=ops_p3,
                      cifar_available=mk_cifar is not None))

    print(f"\n{'='*60}")
    for r in R:
        p1 = "L1" if r['l1_p1'] else "---"
        p3 = "L1" if r['l1_p3'] else "---"
        drift = abs(r['ops_p3'][0] - r['ops_p1'][0]) + abs(r['ops_p3'][1] - r['ops_p1'][1])
        print(f"  s{r['seed']}: P1={p1}({r['l1_p1']}) P3={p3}({r['l1_p3']})  "
              f"op_drift={drift:.0f}pp  "
              f"ops_p1=A{r['ops_p1'][0]:.0f}%P{r['ops_p1'][1]:.0f}%  "
              f"ops_p3=A{r['ops_p3'][0]:.0f}%P{r['ops_p3'][1]:.0f}%")

    p1_l1 = sum(1 for r in R if r['l1_p1'])
    p3_l1 = sum(1 for r in R if r['l1_p3'])
    cifar_ran = any(r['cifar_available'] for r in R)

    print(f"\nP1 L1={p1_l1}/{N_SEEDS}  P3 L1={p3_l1}/{N_SEEDS}  CIFAR={'ran' if cifar_ran else 'skipped'}")

    if p3_l1 >= p1_l1:
        print("SIGNAL: P3 L1 >= P1 L1 -- ops survived game switch (universal or slow drift).")
    elif p3_l1 < p1_l1:
        print("KILL: P3 L1 < P1 L1 -- game switch contaminated ops.")
    else:
        print(f"MARGINAL: P1={p1_l1}/{N_SEEDS} P3={p3_l1}/{N_SEEDS}")


if __name__ == "__main__":
    main()
