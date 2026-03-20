import numpy as np
from collections import defaultdict, deque
import time


class Sub:

    def __init__(self, dim, k, na, P, self_obs):
        self.k, self.na, self.self_obs = k, na, self_obs
        self.P = P
        self.E = defaultdict(lambda: np.zeros(na, dtype=np.int64))
        self.T = defaultdict(lambda: [defaultdict(int) for _ in range(na)])
        self.mu = np.zeros(dim, dtype=np.float64)
        self.ct = 0
        self.prev = None
        self.route = deque()
        self.known = set()
        self.last_new = 0
        self.t = 0
        self.plans = 0
        self.plan_steps = 0
        self.route_aborts = 0

    def _enc(self, x):
        v = x.ravel().astype(np.float64)
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


class Env:

    def __init__(self, n, na, corr, noise, seed):
        r = np.random.RandomState(seed)
        self.n, self.na, self.noise = n, na, noise
        self.corr = corr
        self.total = n + corr + 1
        self.gate = (n * 3 // 4, na - 1)
        self.dim = 16

        self.emb = np.zeros((self.total, self.dim), dtype=np.float32)
        self.emb[0] = r.randn(self.dim).astype(np.float32) * 0.5
        for i in range(1, self.total):
            self.emb[i] = self.emb[i - 1] + r.randn(self.dim).astype(np.float32) * 0.12

        self.det = r.randint(0, n, size=(n, na)).astype(np.int32)
        for s in range(n):
            self.det[s, 0] = (s + 1) % n

        self.rng = np.random.RandomState(seed + 7777)
        self.s = 0

    def obs(self):
        return self.emb[self.s] + self.rng.randn(self.dim).astype(np.float32) * 0.005

    def reset(self):
        self.s = 0
        return self.obs()

    def step(self, a):
        s = self.s
        if s >= self.n + self.corr:
            return self.obs(), True
        if s >= self.n:
            if a == 0:
                self.s = s + 1
            else:
                self.s = self.rng.randint(self.n)
            return self.obs(), self.s >= self.n + self.corr
        if s == self.gate[0] and a == self.gate[1]:
            self.s = self.n
            return self.obs(), False
        target = int(self.det[s, a])
        if self.rng.rand() < self.noise:
            target = (target + self.rng.randint(-3, 4)) % self.n
        self.s = target
        return self.obs(), False


def trial(seed, n, na, k, noise, steps, self_obs):
    np.random.seed(seed * 31337)
    P = np.random.randn(k, 16).astype(np.float32)
    env = Env(n, na, 5, noise, seed)
    sub = Sub(16, k, na, P, self_obs)

    obs = env.reset()
    wins, first_win = 0, -1
    states = set()
    win_steps = []

    for t in range(steps):
        a = sub.act(obs)
        obs, done = env.step(a)
        states.add(env.s)
        if done:
            wins += 1
            if first_win < 0:
                first_win = t
            win_steps.append(t)
            obs = env.reset()
            sub.on_reset()

    gap = -1.0
    if len(win_steps) > 1:
        gaps = [win_steps[i + 1] - win_steps[i] for i in range(len(win_steps) - 1)]
        gap = float(np.median(gaps))

    return (wins, first_win, len(sub.known), len(states), env.total,
            sub.plans, sub.plan_steps, sub.route_aborts, gap)


if __name__ == '__main__':
    N = 100
    NA = 4
    K = 10
    NOISE = 0.1
    STEPS = 50000
    SEEDS = 10

    print(f"n={N} na={NA} k={K} noise={NOISE} steps={STEPS}")
    print()
    print(f"{'s':>2s} {'method':>8s} {'wins':>5s} {'1st':>7s} "
          f"{'nodes':>5s} {'cover':>9s} {'plans':>5s} "
          f"{'pstep':>5s} {'abort':>5s} {'gap':>7s}")
    print("-" * 72)

    aw, ow = [], []
    ag, og = [], []
    t0 = time.time()

    for s in range(SEEDS):
        for flag in [False, True]:
            w, f, nd, st, tot, pl, ps, ab, gp = trial(
                s, N, NA, K, NOISE, STEPS, flag)
            tag = "observe" if flag else "argmin"
            fs = f"{f:7d}" if f >= 0 else "  never"
            gs = f"{gp:7.0f}" if gp >= 0 else "      -"
            print(f"{s:2d} {tag:>8s} {w:5d} {fs} "
                  f"{nd:5d} {st:3d}/{tot:3d} {pl:5d} "
                  f"{ps:5d} {ab:5d} {gs}")
            (ow if flag else aw).append(w)
            (og if flag else ag).append(gp)
        print()

    elapsed = time.time() - t0
    print(f"argmin   wins: mean={np.mean(aw):6.1f} std={np.std(aw):5.1f} "
          f"med_gap={np.median([g for g in ag if g > 0]):7.0f}")
    print(f"observe  wins: mean={np.mean(ow):6.1f} std={np.std(ow):5.1f} "
          f"med_gap={np.median([g for g in og if g > 0]):7.0f}")
    print(f"delta: {np.mean(ow) - np.mean(aw):+.1f} wins ({elapsed:.1f}s)")
