"""
Step 570 — Self-observation via graph planning: LS20 A/B test.

A: Sub(self_obs=False) — pure argmin baseline
B: Sub(self_obs=True)  — BFS routing after exploration stall (no new nodes 500 steps)

10 seeds x 10K steps each condition. 5-min cap total.
Kill: if self_obs <= argmin on L1 speed/wins.
"""
import numpy as np
from collections import defaultdict, deque
import time
import sys


class Sub:

    def __init__(self, k=12, na=4, self_obs=True, pool=16):
        self.k, self.na, self.self_obs = k, na, self_obs
        self.pool = pool
        self.P = None
        self.E = defaultdict(lambda: np.zeros(na, dtype=np.int64))
        self.T = defaultdict(lambda: [defaultdict(int) for _ in range(na)])
        self.mu = None
        self.ct = 0
        self.prev = None
        self.route = deque()
        self.known = set()
        self.last_new = 0
        self.t = 0
        self.plans = 0
        self.plan_steps = 0
        self.route_aborts = 0

    def _pool(self, frame):
        if frame.ndim == 3:
            frame = frame.mean(axis=2)
        h, w = frame.shape
        bh, bw = h // self.pool, w // self.pool
        pooled = frame[:bh*self.pool, :bw*self.pool].reshape(
            self.pool, bh, self.pool, bw).mean(axis=(1, 3))
        return pooled.ravel().astype(np.float64)

    def _enc(self, frame):
        v = self._pool(frame)
        if self.P is None:
            dim = len(v)
            rng = np.random.RandomState(42)
            self.P = rng.randn(self.k, dim).astype(np.float32)
            self.mu = np.zeros(dim, dtype=np.float64)
        self.ct += 1
        self.mu += (v - self.mu) / self.ct
        c = v - self.mu
        n = np.linalg.norm(c)
        return (c / max(n, 1e-8)).astype(np.float32)

    def _h(self, x):
        return sum(int(b) << i for i, b in enumerate(self.P @ x > 0))

    def _seek(self, origin):
        q = deque([(origin, [])])
        vis = {origin}
        tgt_path, tgt_node, tgt_score = None, None, float('inf')
        while q:
            nd, path = q.popleft()
            if len(path) > 60:
                continue
            if nd != origin:
                sc = int(self.E[nd].sum())
                if sc < tgt_score:
                    tgt_score, tgt_path, tgt_node = sc, list(path), nd
            for a in range(self.na):
                tr = self.T[nd][a]
                if tr:
                    nxt = max(tr, key=tr.get)
                    tot = sum(tr.values())
                    if nxt not in vis and tr[nxt] / tot > 0.5:
                        vis.add(nxt)
                        q.append((nxt, path + [a]))
        if tgt_path is None:
            return deque()
        path = list(tgt_path)
        node = tgt_node
        for _ in range(40):
            ba, bn, bs = None, None, float('inf')
            for a in range(self.na):
                tr = self.T[node][a]
                if tr:
                    nxt = max(tr, key=tr.get)
                    tot = sum(tr.values())
                    if nxt not in vis and tr[nxt] / tot > 0.5:
                        sc = int(self.E[nxt].sum())
                        if sc < bs:
                            ba, bn, bs = a, nxt, sc
            if ba is None:
                break
            path.append(ba)
            vis.add(bn)
            node = bn
        return deque(path)

    def act(self, obs):
        x = self._enc(obs)
        node = self._h(x)

        if node not in self.known:
            self.known.add(node)
            self.last_new = self.t

        if self.prev:
            pn, pa, exp = self.prev
            self.E[pn][pa] += 1
            self.T[pn][pa][node] += 1
            if exp is not None and node != exp:
                self.route.clear()
                self.route_aborts += 1
                if self.self_obs and self.t - self.last_new > 500:
                    self.route = self._seek(node)
                    if self.route:
                        self.plans += 1

        if (self.self_obs and not self.route
                and self.t - self.last_new > 500 and self.t % 100 == 0):
            self.route = self._seek(node)
            if self.route:
                self.plans += 1

        exp = None
        if self.route:
            a = self.route.popleft()
            tr = self.T[node][a]
            if tr:
                exp = max(tr, key=tr.get)
            self.plan_steps += 1
        else:
            a = int(np.argmin(self.E[node]))

        self.prev = (node, a, exp)
        self.t += 1
        return a

    def on_reset(self):
        self.prev = None
        self.route.clear()


def t0():
    rng = np.random.RandomState(0)
    for obs_flag in [False, True]:
        sub = Sub(self_obs=obs_flag)
        for _ in range(20):
            obs = rng.randint(0, 16, (64, 64)).astype(np.float32)
            a = sub.act(obs)
            assert 0 <= a < 4, f"action {a} out of range"
        sub.on_reset()
        assert sub.prev is None
        assert len(sub.route) == 0
    # BFS only plans when self_obs=True
    sub_a = Sub(self_obs=False)
    sub_b = Sub(self_obs=True)
    for _ in range(600):
        obs = rng.randint(0, 16, (64, 64)).astype(np.float32)
        sub_a.act(obs); sub_b.act(obs)
    assert sub_a.plans == 0, f"argmin should have 0 plans: {sub_a.plans}"
    print("T0 PASS")


def run_condition(env_factory, seed, steps, self_obs):
    env = env_factory()
    sub = Sub(self_obs=self_obs)
    obs = env.reset(seed=seed)
    wins = 0
    first_win = None
    win_steps = []
    last_win_step = 0
    level = 0

    for step in range(1, steps + 1):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            continue
        frame = np.array(obs[0], dtype=np.float32)
        action = sub.act(frame)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset(seed=seed)
            sub.on_reset()
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level = cl
            sub.on_reset()
            if cl >= 1:
                wins += 1
                if first_win is None:
                    first_win = step
                win_steps.append(step - last_win_step)
                last_win_step = step

    median_gap = float(np.median(win_steps)) if win_steps else None
    return {
        'wins': wins,
        'first_win': first_win,
        'nodes': len(sub.known),
        'plans': sub.plans,
        'plan_steps': sub.plan_steps,
        'route_aborts': sub.route_aborts,
        'median_gap': median_gap,
        'level': level,
    }


def main():
    t0()
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    n_seeds = 10
    steps_per = 10_000
    global_cap = 280
    t_start = time.time()

    results_a, results_b = [], []

    for seed in range(n_seeds):
        elapsed = time.time() - t_start
        if elapsed > global_cap - 10:
            print(f"\nCap hit at seed {seed}", flush=True); break

        print(f"\nseed {seed}:", flush=True)

        # Condition A: pure argmin
        r_a = run_condition(mk, seed, steps_per, self_obs=False)
        # Condition B: BFS self-observation
        r_b = run_condition(mk, seed, steps_per, self_obs=True)

        results_a.append(r_a)
        results_b.append(r_b)

        el = time.time() - t_start
        fw_a = r_a['first_win'] or '-'
        fw_b = r_b['first_win'] or '-'
        print(f"  A(argmin): wins={r_a['wins']} first={fw_a} nodes={r_a['nodes']} {el:.0f}s")
        print(f"  B(selfobs): wins={r_b['wins']} first={fw_b} nodes={r_b['nodes']} "
              f"plans={r_b['plans']} ps={r_b['plan_steps']} aborts={r_b['route_aborts']}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}\nElapsed: {elapsed:.0f}s")

    if not results_a:
        print("No results."); return

    # Per-seed table
    print(f"\n{'Seed':>4} | {'A_wins':>6} {'A_1st':>6} {'A_nodes':>7} | "
          f"{'B_wins':>6} {'B_1st':>6} {'B_nodes':>7} {'B_plans':>7} {'B_ps':>6} {'B_abts':>6}")
    print("-" * 72)
    for i, (ra, rb) in enumerate(zip(results_a, results_b)):
        fw_a = str(ra['first_win']) if ra['first_win'] else '-'
        fw_b = str(rb['first_win']) if rb['first_win'] else '-'
        print(f"  {i:>2} | {ra['wins']:>6} {fw_a:>6} {ra['nodes']:>7} | "
              f"{rb['wins']:>6} {fw_b:>6} {rb['nodes']:>7} {rb['plans']:>7} "
              f"{rb['plan_steps']:>6} {rb['route_aborts']:>6}")

    # Summary
    def mean_of(lst, key): return np.mean([r[key] for r in lst])
    def std_of(lst, key): return np.std([r[key] for r in lst])

    a_wins = mean_of(results_a, 'wins'); b_wins = mean_of(results_b, 'wins')
    a_fw = [r['first_win'] for r in results_a if r['first_win']]
    b_fw = [r['first_win'] for r in results_b if r['first_win']]
    a_nodes = mean_of(results_a, 'nodes'); b_nodes = mean_of(results_b, 'nodes')

    print(f"\n{'='*60}")
    print(f"Summary ({len(results_a)} seeds, {steps_per} steps each):")
    print(f"  A(argmin): wins={a_wins:.1f} nodes={a_nodes:.0f} "
          f"first_win={'%.0f'%np.mean(a_fw) if a_fw else 'none'} ({len(a_fw)}/{len(results_a)} got win)")
    print(f"  B(selfobs): wins={b_wins:.1f} nodes={b_nodes:.0f} "
          f"first_win={'%.0f'%np.mean(b_fw) if b_fw else 'none'} ({len(b_fw)}/{len(results_b)} got win) "
          f"plans={mean_of(results_b,'plans'):.1f} ps={mean_of(results_b,'plan_steps'):.0f} "
          f"aborts={mean_of(results_b,'route_aborts'):.0f}")
    print(f"  Delta wins: {b_wins - a_wins:+.1f}  Delta nodes: {b_nodes - a_nodes:+.0f}")

    if b_wins > a_wins:
        print(f"\nFIND: B > A on wins ({b_wins:.1f} vs {a_wins:.1f}). BFS self-obs helps.")
    elif b_wins < a_wins:
        print(f"\nKILL: A > B ({a_wins:.1f} vs {b_wins:.1f}). BFS routing hurts.")
    else:
        if b_fw and a_fw and np.mean(b_fw) < np.mean(a_fw):
            print(f"\nTIE wins, B faster ({np.mean(b_fw):.0f} vs {np.mean(a_fw):.0f} steps to first win).")
        else:
            print(f"\nKILL: No improvement. BFS self-obs does not help.")


if __name__ == "__main__":
    main()
