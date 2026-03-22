"""
Step 629 — Eigenform L2 attempt: L1 success tagging as L2 mechanism.

Base: LSH k=16, centered_enc, argmin, eigenform 620 self-observation. LS20 new game.

Addition: L1 SUCCESS TAGGING.
- When L1 is first reached, tag all nodes visited in that winning episode as "L1-productive"
- Self-observation now combines two sources:
  1. Edge count distribution (as in 620) -> AVOID over-visited, PREFER under-visited
  2. L1-tag override -> PREFER any node in l1_productive (overrides source 1)
- After first L1 success, navigation biases toward L1-productive subgraph

Hypothesis: faster L1 replay leaves more exploration budget to discover L2.

5 seeds x 300s. Signal: L2 >= 1/5. Kill: L2 = 0/5 AND no L1 speedup vs 620 baseline.
Also report: L1 step (629) vs 620 baseline for tagging speedup.
"""
import numpy as np
import time
import sys

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
SELF_OBS_EVERY = 500
PER_SEED_TIME = 300

OP_NEUTRAL = 0
OP_AVOID = 1
OP_PREFER = 2
AVOID_PENALTY = 100
PREFER_BONUS = 50


def enc(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class Recode:

    def __init__(self, dim=DIM, k=K, seed=0):
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
        self.op_history = []
        # L1 tagging state
        self.l1_productive = set()
        self.current_episode_nodes = []
        self.l1_ever = False

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
        self.current_episode_nodes.append(n)
        self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)
        self._px = x
        self._cn = n
        self.dim = len(x)
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        if self.t > 0 and self.t % SELF_OBS_EVERY == 0:
            self._self_observe()
        return n

    def tag_l1_episode(self):
        """Tag current episode nodes as L1-productive. Call BEFORE on_reset()."""
        self.l1_productive.update(self.current_episode_nodes)
        self.l1_ever = True

    def act(self):
        counts = []
        for a in range(N_A):
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
            total = sum(sum(self.G.get((n, a), {}).values()) for a in range(N_A))
            if total == 0:
                continue
            for a in range(N_A):
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
        avoid_n = prefer_n = neutral_n = 0
        for (n, a), sig in na_signals.items():
            if sig > p90:
                new_ops[(n, a)] = OP_AVOID
                avoid_n += 1
            elif sig < p10:
                new_ops[(n, a)] = OP_PREFER
                prefer_n += 1
            else:
                new_ops[(n, a)] = OP_NEUTRAL
                neutral_n += 1

        # Source 2: L1-tag override — PREFER nodes on the L1-productive path
        if self.l1_ever:
            for (n, a) in list(new_ops.keys()):
                if n in self.l1_productive and new_ops[(n, a)] != OP_PREFER:
                    new_ops[(n, a)] = OP_PREFER
            # Recount after overrides
            avoid_n = sum(1 for v in new_ops.values() if v == OP_AVOID)
            prefer_n = sum(1 for v in new_ops.values() if v == OP_PREFER)
            neutral_n = sum(1 for v in new_ops.values() if v == OP_NEUTRAL)

        self.ops = new_ops
        total_na = avoid_n + prefer_n + neutral_n
        if total_na > 0:
            self.op_history.append((
                self.t,
                100.0 * avoid_n / total_na,
                100.0 * prefer_n / total_na,
                100.0 * neutral_n / total_na,
            ))

    def on_reset(self):
        self._pn = None
        self.current_episode_nodes = []

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

    def op_stats(self):
        total = sum(self.op_counts)
        if total == 0:
            return 0.0, 0.0, 100.0
        return (
            100.0 * self.op_counts[1] / total,
            100.0 * self.op_counts[2] / total,
            100.0 * self.op_counts[0] / total,
        )


def t0():
    rng = np.random.RandomState(42)
    sub = Recode(dim=8, k=3, seed=0)
    sub.H = rng.randn(3, 8).astype(np.float32)
    sub.dim = 8
    x1 = rng.randn(8).astype(np.float32)
    assert isinstance(sub._node(x1), int)

    # Test: episode tracking — nodes accumulate, clear on reset
    sub.current_episode_nodes = [1, 2, 3]
    sub.on_reset()
    assert sub.current_episode_nodes == [], "on_reset should clear episode nodes"

    # Test: tag_l1_episode adds to l1_productive and sets l1_ever
    sub2 = Recode(dim=8, k=3, seed=0)
    sub2.H = sub.H.copy()
    sub2.dim = 8
    sub2.current_episode_nodes = [10, 11, 12]
    assert not sub2.l1_ever
    sub2.tag_l1_episode()
    assert sub2.l1_ever
    assert {10, 11, 12}.issubset(sub2.l1_productive)

    # Test: L1-tag override in _self_observe
    # Build graph with enough signals (outlier at node 0, uniform at 1-19)
    sub3 = Recode(dim=8, k=3, seed=0)
    sub3.H = sub.H.copy()
    sub3.dim = 8
    sub3.G[(0, 0)] = {1: 999}
    sub3.G[(0, 1)] = {1: 1}
    for i in range(1, 20):
        sub3.G[(i, 0)] = {i+1: 50}
        sub3.G[(i, 1)] = {i+1: 50}
    sub3.live = set(range(25))
    sub3.t = 1

    # Without tagging: node 5 should be NEUTRAL (signal=0.5, in middle)
    sub3._self_observe()
    assert len(sub3.ops) > 0, "ops not assigned"
    op_5_0 = sub3.ops.get((5, 0), OP_NEUTRAL)
    assert op_5_0 == OP_NEUTRAL, f"node 5 should be NEUTRAL before tagging, got {op_5_0}"

    # Tag node 5 as L1-productive, re-observe
    sub3.l1_productive = {5}
    sub3.l1_ever = True
    sub3._self_observe()
    tagged_ops = [sub3.ops.get((5, a), OP_NEUTRAL) for a in range(N_A)]
    assert any(op == OP_PREFER for op in tagged_ops), \
        f"tagged node 5 should have PREFER: {tagged_ops}"

    print("T0 PASS")


# 620 baseline L1 steps (for comparison in output)
BASELINE_620_L1 = {
    # seed: step — fill from step620 log if available, else None
    0: None, 1: None, 2: None, 3: None, 4: None,
}


def run(seed, make):
    env = make()
    sub = Recode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = l2 = None
    go = 0
    t_start = time.time()

    for step in range(1, 2_000_001):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            continue

        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)

        cl = info.get('level', 0) if isinstance(info, dict) else 0

        # Check level BEFORE done so episode nodes are intact for tagging
        if cl > level:
            nc, ns, ne = sub.stats()
            av, pr, nt = sub.op_stats()
            if cl == 1 and l1 is None:
                l1 = step
                sub.tag_l1_episode()  # tag BEFORE on_reset
                print(f"  s{seed} L1@{step} c={nc} sp={ns} e={ne} go={go} "
                      f"tagged={len(sub.l1_productive)} ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%",
                      flush=True)
            if cl == 2 and l2 is None:
                l2 = step
                print(f"  s{seed} L2@{step} c={nc} sp={ns} e={ne} go={go} "
                      f"l1_prod={len(sub.l1_productive)} ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%",
                      flush=True)
            level = cl
            sub.on_reset()

        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset()

        if time.time() - t_start > PER_SEED_TIME:
            break

    nc, ns, ne = sub.stats()
    av, pr, nt = sub.op_stats()

    return dict(seed=seed, l1=l1, l2=l2, cells=nc, splits=ns, edges=ne, go=go,
                avoid_pct=av, prefer_pct=pr, neutral_pct=nt,
                l1_productive=len(sub.l1_productive))


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        print("\nDry run only (no ARC environment). T0 passed.")
        return

    R = []
    for seed in range(5):
        print(f"\nseed {seed}:", flush=True)
        R.append(run(seed, mk))

    print(f"\n{'='*60}")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        av, pr, nt = r['avoid_pct'], r['prefer_pct'], r['neutral_pct']
        b620 = BASELINE_620_L1.get(r['seed'])
        speedup = ""
        if r['l1'] and b620:
            speedup = f"  speedup={b620-r['l1']:+d}"
        print(f"  s{r['seed']}: {tag:>3}  c={r['cells']:>5}  "
              f"sp={r['splits']:>3}  e={r['edges']:>5}  go={r['go']}  "
              f"l1_prod={r['l1_productive']}  ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%{speedup}")

    l2n = sum(1 for r in R if r['l2'])
    l1n = sum(1 for r in R if r['l1'])
    l1_steps = [r['l1'] for r in R if r['l1']]

    print(f"\nL1={l1n}/5  L2={l2n}/5")
    print(f"L1 steps (629): {l1_steps}")
    print(f"620 baseline:   {[BASELINE_620_L1[s] for s in range(5)]}")

    if l2n >= 1:
        print("SIGNAL: L2 >= 1/5 -- eigenform tagging enables L2 discovery.")
    elif l1n < 3:
        print("KILL: L1 degraded -- tagging mechanism hurts L1 performance.")
    else:
        # Check if L1 steps are faster than baseline
        faster = sum(1 for r in R if r['l1'] and BASELINE_620_L1.get(r['seed'])
                     and r['l1'] < BASELINE_620_L1[r['seed']])
        if faster >= 3:
            print(f"PARTIAL: L2=0/5 but L1 faster in {faster}/5 seeds -- tagging has navigation value. Run longer?")
        else:
            print(f"KILL: L2=0/5, no L1 speedup confirmed -- eigenform tagging inert at this level.")


if __name__ == "__main__":
    main()
