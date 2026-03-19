"""
Step 532 — Level 2 stochastic attack, k=16.

Step 531 killed k=12: k=16 reaches 1094 cells vs k=12's 425 at 200K.
Finer partition may reveal more stochastic edges and better navigate L2.
Based on Jun's Step 528 script, K changed from 12 to 16.

Predictions:
- More stochastic edges (finer partition = narrower state boundaries).
- max_cells > 1094 (500K steps >> 200K).
- Still 0/3 L2 (structural disconnect, not partition artifact).
Kill: L2 reached.
"""
import time
import numpy as np
from collections import deque

K = 16
N_ACTIONS = 4
MAX_STEPS = 500_000
TIME_CAP = 300
N_SEEDS = 3
EXPLOIT_REPS = 500
WANDER = 200
TOP_K = 20


def encode(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class LSH:
    def __init__(self, dim=256, k=K, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)

    def __call__(self, x):
        return int(np.packbits((self.H @ x > 0).astype(np.uint8),
                               bitorder='big').tobytes().hex(), 16)


class WorldModel:
    def __init__(self, lsh):
        self.lsh = lsh
        self.edges = {}
        self.prev_node = None
        self.prev_action = None
        self.cells = set()
        self.max_cells = 0

    def observe(self, frame):
        node = self.lsh(encode(frame))
        self.cells.add(node)
        self.max_cells = max(self.max_cells, len(self.cells))
        if self.prev_node is not None:
            d = self.edges.setdefault((self.prev_node, self.prev_action), {})
            d[node] = d.get(node, 0) + 1
        self.prev_node = node
        return node

    def act(self, a):
        self.prev_action = a

    def on_reset(self):
        self.prev_node = None

    def argmin(self, node):
        counts = [sum(self.edges.get((node, a), {}).values())
                  for a in range(N_ACTIONS)]
        return int(np.argmin(counts))

    def stochastic_edges(self):
        found = []
        for (node, action), dist in self.edges.items():
            if len(dist) < 2:
                continue
            total = sum(dist.values())
            if total < 20:
                continue
            p = np.array(list(dist.values()), dtype=np.float64) / total
            entropy = float(-np.sum(p * np.log2(p + 1e-15)))
            found.append((entropy, node, action, dict(dist)))
        found.sort(reverse=True)
        return found[:TOP_K]

    def route(self, start, goal):
        if start == goal:
            return []
        likely = {}
        for (n, a), dist in self.edges.items():
            likely.setdefault(n, {})[a] = max(dist, key=dist.get)
        visited = {start}
        queue = deque([(start, [])])
        while queue:
            cur, path = queue.popleft()
            for a in range(N_ACTIONS):
                nxt = likely.get(cur, {}).get(a)
                if nxt is None or nxt in visited:
                    continue
                if nxt == goal:
                    return path + [a]
                visited.add(nxt)
                queue.append((nxt, path + [a]))
        return None


def t1():
    lsh = LSH(seed=0)
    wm = WorldModel(lsh)

    rng = np.random.RandomState(7)
    nodes = list(range(20))
    for _ in range(500):
        wm.prev_node = rng.choice(nodes[:10])
        wm.prev_action = rng.randint(N_ACTIONS)
        wm.edges.setdefault((wm.prev_node, wm.prev_action), {})
        d = wm.edges[(wm.prev_node, wm.prev_action)]
        nxt = rng.choice(nodes[:10])
        d[nxt] = d.get(nxt, 0) + 1

    wm.edges[(5, 2)] = {10: 997, 15: 3}

    targets = wm.stochastic_edges()
    assert any(n == 5 and a == 2 for _, n, a, _ in targets), "missed planted edge"

    wm.edges.clear()
    for i in range(5):
        wm.edges[(i, 0)] = {i + 1: 100}
    path = wm.route(0, 5)
    assert path == [0, 0, 0, 0, 0], f"bad route: {path}"
    assert wm.route(0, 99) is None

    print("T1 PASS")


def run_seed(seed, env_cls):
    env = env_cls()
    wm = WorldModel(LSH(seed=seed * 1000))
    obs = env.reset(seed=seed)
    level = 0
    l1_step = None
    l2_step = None

    phase = 'argmin'
    targets = []
    ti = 0
    nav = []
    exploit_left = 0
    wander_left = 0
    cells_at_l1 = 0
    t0 = time.time()

    for step in range(1, MAX_STEPS + 1):
        node = wm.observe(obs)

        if phase == 'argmin':
            action = wm.argmin(node)

        elif phase == 'pursue':
            if ti >= len(targets):
                phase = 'argmin'
                action = wm.argmin(node)
            elif node == targets[ti][1]:
                phase = 'exploit'
                exploit_left = EXPLOIT_REPS
                action = targets[ti][2]
            elif nav:
                action = nav.pop(0)
            else:
                plan = wm.route(node, targets[ti][1])
                if plan:
                    nav = plan
                    action = nav.pop(0)
                else:
                    ti += 1
                    action = wm.argmin(node)

        elif phase == 'exploit':
            action = targets[ti][2]
            exploit_left -= 1
            if exploit_left <= 0:
                ti += 1
                phase = 'wander'
                wander_left = WANDER

        elif phase == 'wander':
            action = wm.argmin(node)
            wander_left -= 1
            if wander_left <= 0:
                phase = 'pursue'
                nav = []

        wm.act(action)
        obs, reward, done, info = env.step(action)

        if done:
            obs = env.reset(seed=seed)
            wm.on_reset()
            if phase in ('pursue', 'exploit'):
                phase = 'pursue'
                nav = []

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1_step is None:
                l1_step = step
                cells_at_l1 = len(wm.cells)
                targets = wm.stochastic_edges()
                ti = 0
                phase = 'pursue'
                nav = []
                n_stoch = len(targets)
                if n_stoch == 0:
                    phase = 'argmin'
                print(f"  seed {seed}: L1@{step}, {cells_at_l1} cells, "
                      f"{n_stoch} stochastic edges")
                if n_stoch > 0:
                    for i, (ent, n, a, dist) in enumerate(targets[:5]):
                        print(f"    [{i}] node={n} act={a} H={ent:.3f} dist={dist}")
            if cl == 2:
                l2_step = step
                print(f"  seed {seed}: L2@{step}, {len(wm.cells)} cells")
            level = cl

        if step % 100_000 == 0:
            print(f"  seed {seed}: {step}, cells={len(wm.cells)}, "
                  f"phase={phase}, ti={ti}/{len(targets)}, "
                  f"{time.time()-t0:.0f}s")

        if time.time() - t0 > TIME_CAP:
            break

    return dict(seed=seed, l1=l1_step, l2=l2_step,
                cells_l1=cells_at_l1, max_cells=wm.max_cells,
                stochastic=len(targets), tried=min(ti, len(targets)))


def main():
    t1()

    try:
        import arcagi3
        make = lambda: arcagi3.make("LS20")
    except ImportError:
        print("arcagi3 unavailable")
        return

    results = []
    for seed in range(N_SEEDS):
        print(f"\nseed {seed}:")
        results.append(run_seed(seed, make))

    print(f"\n{'='*50}")
    for r in results:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "—")
        print(f"  seed {r['seed']}: {tag}  max_cells={r['max_cells']}  "
              f"stochastic={r['stochastic']}  tried={r['tried']}")

    mc = max(r['max_cells'] for r in results)
    n_stoch = max(r['stochastic'] for r in results)
    l2 = sum(1 for r in results if r['l2'])

    print(f"\nL2: {l2}/{N_SEEDS}   max_cells: {mc}   stochastic_edges: {n_stoch}")

    if l2 > 0:
        print("KILL: L2 reached with k=16!")
    elif mc > 1094:
        print(f"GROWTH: {mc} > 1094. k=16 expands reachable set further.")
    elif n_stoch > 20:
        print(f"DIAGNOSTIC: {n_stoch} stochastic edges (k=12 found ~20). "
              f"Finer partition reveals more boundaries.")
    else:
        print(f"CONFIRMED: L2 structurally disconnected. k=16 does not help.")


if __name__ == "__main__":
    main()
