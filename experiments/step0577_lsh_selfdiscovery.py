"""
Step 577 -- Self-discovering substrate (R3 attack).

Goal: A substrate that discovers its own processing pipeline from generic
primitives, evaluated by task performance. No prescribed detection, no
prescribed navigation targets, no game-specific knowledge.

Seed: LSH k=12 + argmin on LS20. Baseline navigates L1 at ~2K steps/episode.
Everything beyond this must be SELF-DISCOVERED.

Primitives (generic, not game-specific):
  accumulate  -- running sum per pixel-position-color (builds color histogram)
  argmax      -- most frequent color per pixel (mode map)
  threshold_C -- binary mask: mode_map == C, for C in 1..15
  label_cc    -- find isolated connected components (size 2-60)
  nearest     -- nearest CC center to agent position
  navigate    -- greedy 4-direction action toward (cy, cx)

Search:
  Population: N=20 random programs (3-7 primitives each)
  Each episode: run one program (round-robin), fall back to argmin if no action
  Score: BASELINE_EP / steps_to_L1 (>1 = improvement over baseline)
  Tournament: every 100 episodes, keep top 50%, mutate bottom 50%
  Mutation: insert / remove / replace one op

Kill: no program achieves score >1.5 after 200K steps.
Signal: any program >1.5 -> pipeline discovery confirmed.

5 seeds, 200K steps each, 5-min cap per seed.
"""
import time
import numpy as np
import sys
from scipy.ndimage import label as ndlabel

K = 16
DIM = 256    # avgpool16 (16x16=256) for SubLSH argmin fallback (step 546 encoding)
N_A = 4
N_POP = 20
EVOLVE_EVERY = 100   # episodes between tournament selections
BASELINE_EP = 2000   # reference steps-to-L1 for scoring (conservative)
MAX_EPISODE = 3000   # cap per episode (avoid infinite loops)
MAX_STEPS = 200_000
TIME_CAP = 280       # seconds per seed (5-min cap)

# ── primitive set ─────────────────────────────────────────────────────────────

# All possible ops: (name, *params). 20 total.
ALL_OPS = (
    [('accumulate',)] +
    [('argmax',)] +
    [('threshold', c) for c in range(1, 16)] +
    [('label_cc',)] +
    [('nearest',)] +
    [('navigate',)]
)
# Count: 1 + 1 + 15 + 1 + 1 + 1 = 20


def execute(ops, frame, buf, agent_yx):
    """
    Execute program ops against current frame.
    Returns (action_or_None, updated_buf).
    buf: (64,64,16) int32 accumulator (persistent across frames for this program).
    agent_yx: (y,x) float estimate of agent position, or None.
    """
    arr = np.array(frame[0], dtype=np.int32)
    val = arr   # current pipeline value

    for op in ops:
        name = op[0]
        try:
            if name == 'accumulate':
                r = np.arange(64)[:, None]; c = np.arange(64)[None, :]
                buf[r, c, val.clip(0, 15)] += 1
                val = arr   # side effect only; value unchanged

            elif name == 'argmax':
                if buf.max() == 0:
                    return None, buf   # buffer empty
                val = np.argmax(buf, axis=2).astype(np.int32)

            elif name == 'threshold':
                color = op[1]
                if not hasattr(val, 'ndim') or val.ndim != 2:
                    return None, buf
                val = (val == color).astype(np.uint8)

            elif name == 'label_cc':
                if not hasattr(val, 'ndim') or val.ndim != 2:
                    return None, buf
                labeled, n = ndlabel(val.astype(bool))
                ccs = []
                for cid in range(1, n + 1):
                    region = (labeled == cid)
                    sz = int(region.sum())
                    if 2 <= sz <= 60:
                        ys, xs = np.where(region)
                        ccs.append((float(ys.mean()), float(xs.mean()), sz))
                val = ccs

            elif name == 'nearest':
                if not isinstance(val, list) or len(val) == 0:
                    return None, buf
                if agent_yx is None:
                    return None, buf
                ay, ax = agent_yx
                best = min(val, key=lambda t: (t[0]-ay)**2 + (t[1]-ax)**2)
                val = (best[0], best[1])

            elif name == 'navigate':
                if not (isinstance(val, tuple) and len(val) == 2):
                    return None, buf
                if agent_yx is None:
                    return None, buf
                cy, cx = val
                ay, ax = agent_yx
                dy, dx = cy - ay, cx - ax
                if abs(dy) >= abs(dx):
                    return (0 if dy < 0 else 1), buf
                return (2 if dx < 0 else 3), buf

        except Exception:
            return None, buf

    return None, buf   # program didn't produce an action


# ── LSH argmin fallback ───────────────────────────────────────────────────────

class SubLSH:
    def __init__(self, k=K, dim=DIM, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.G = {}
        self._pn = self._pa = self._cn = None
        self.cells = set()
        self.prev_arr = None
        self.agent_yx = None

    def observe(self, frame):
        arr = np.array(frame[0], dtype=np.int32)
        if self.prev_arr is not None:
            diff = np.abs(arr - self.prev_arr) > 0
            nc = int(diff.sum())
            if 1 <= nc < 200:
                ys, xs = np.where(diff)
                self.agent_yx = (float(ys.mean()), float(xs.mean()))
        self.prev_arr = arr.copy()
        # avgpool16: 64x64 -> 16x16 = 256D (same encoding as step 546 Recode)
        x = arr.astype(np.float32).reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten() / 15.0
        x -= x.mean()
        n = int(np.packbits((self.H @ x > 0).astype(np.uint8),
                            bitorder='big').tobytes().hex(), 16)
        self.cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        a = int(np.argmin(counts))
        self._pn = self._cn; self._pa = a; return a

    def record_action(self, a):
        """Record that action a was taken (for programs that override argmin)."""
        self._pn = self._cn; self._pa = a

    def on_reset(self):
        self._pn = None; self.prev_arr = None; self.agent_yx = None


# ── population ────────────────────────────────────────────────────────────────

class Population:
    def __init__(self, n=N_POP, seed=0):
        self.rng = np.random.RandomState(seed)
        self.n = n
        self.ops = [self._random_ops() for _ in range(n)]
        self.bufs = [np.zeros((64, 64, 16), dtype=np.int32) for _ in range(n)]
        self.ep_scores = [[] for _ in range(n)]   # per-program episode scores
        self.current = 0
        self.total_episodes = 0
        self.best_score = 0.0
        self.best_ops = None
        self.evolve_count = 0

    def _random_ops(self):
        length = self.rng.randint(3, 8)
        return [ALL_OPS[self.rng.randint(len(ALL_OPS))] for _ in range(length)]

    def _mutate(self, ops):
        ops = list(ops)
        r = self.rng.randint(3)
        if r == 0 and len(ops) < 8:
            ops.insert(self.rng.randint(len(ops) + 1),
                       ALL_OPS[self.rng.randint(len(ALL_OPS))])
        elif r == 1 and len(ops) > 2:
            ops.pop(self.rng.randint(len(ops)))
        else:
            ops[self.rng.randint(len(ops))] = ALL_OPS[self.rng.randint(len(ALL_OPS))]
        return ops

    def record(self, prog_idx, steps_to_l1):
        score = float(BASELINE_EP) / steps_to_l1 if steps_to_l1 > 0 else 0.0
        self.ep_scores[prog_idx].append(score)
        self.total_episodes += 1
        if score > self.best_score:
            self.best_score = score
            self.best_ops = list(self.ops[prog_idx])

    def evolve(self):
        """Tournament selection: top 50% survive, bottom 50% are mutated copies."""
        avg = [float(np.mean(s)) if s else 0.0 for s in self.ep_scores]
        ranked = sorted(range(self.n), key=lambda i: avg[i], reverse=True)
        top = ranked[:self.n // 2]
        for idx in ranked[self.n // 2:]:
            parent = self.rng.choice(top)
            self.ops[idx] = self._mutate(list(self.ops[parent]))
            self.bufs[idx][:] = 0   # reset buffer for new program
            self.ep_scores[idx] = []
        self.evolve_count += 1


# ── tests ─────────────────────────────────────────────────────────────────────

def t0():
    buf = np.zeros((64, 64, 16), dtype=np.int32)
    frame = [np.zeros((64, 64), dtype=np.uint8)]
    frame[0][20:24, 30:34] = 9   # inject color-9 block

    # Test: accumulate -> argmax -> threshold_9 -> label_cc -> nearest -> navigate
    ops = [('accumulate',), ('argmax',), ('threshold', 9),
           ('label_cc',), ('nearest',), ('navigate',)]
    agent_yx = (10.0, 10.0)

    # Need enough frames to build mode
    for _ in range(100):
        _, buf = execute(ops, frame, buf, agent_yx)
    action, buf = execute(ops, frame, buf, agent_yx)
    assert action is not None, f"Expected action, got None. buf.max={buf.max()}"
    print(f"T0 PASS: action={action} (toward color-9 block at y=22,x=32 from y=10,x=10)")

    # Test random program returns None or valid action
    pop = Population(seed=0)
    frame2 = [np.random.RandomState(3).randint(0, 16, (64, 64), dtype=np.uint8)]
    for i in range(pop.n):
        buf2 = pop.bufs[i]
        a, buf2 = execute(pop.ops[i], frame2, buf2, (32.0, 32.0))
        assert a is None or 0 <= a < 4, f"Invalid action {a}"
    print("T0 PASS: population produces valid actions or None")


# ── experiment ────────────────────────────────────────────────────────────────

def run_seed(mk, seed, time_cap=TIME_CAP):
    env = mk()
    sub = SubLSH(k=K, dim=DIM, seed=seed * 1000)
    pop = Population(n=N_POP, seed=seed * 7)
    obs = env.reset(seed=seed)

    l1_total = go = ep_step = 0
    prev_cl = 0; t_start = time.time(); step = 0
    prog_idx = 0
    ep_l1_step = None   # steps to L1 this episode
    # LS20 persists cl from last episode for one step after reset.
    # fresh_episode=True skips L1 detection on the first step and syncs prev_cl.
    fresh_episode = True

    def do_episode_end():
        nonlocal prog_idx, ep_step, ep_l1_step, go, fresh_episode
        go += 1
        pop.record(prog_idx, ep_l1_step if ep_l1_step else 0)
        prog_idx = (prog_idx + 1) % N_POP; ep_step = 0; ep_l1_step = None
        if pop.total_episodes > 0 and pop.total_episodes % EVOLVE_EVERY == 0:
            pop.evolve()
        fresh_episode = True

    while step < MAX_STEPS and time.time() - t_start < time_cap:
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset()
            do_episode_end()
            continue

        sub.observe(obs)
        action, pop.bufs[prog_idx] = execute(
            pop.ops[prog_idx], obs, pop.bufs[prog_idx], sub.agent_yx)
        if action is None:
            action = sub.act()
        else:
            sub.record_action(action)

        obs, reward, done, info = env.step(action)
        step += 1; ep_step += 1

        if done:
            do_episode_end()
            obs = env.reset(seed=seed); sub.on_reset()
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0

        if fresh_episode:
            # First step after reset: sync prev_cl without counting L1.
            # LS20 may carry over cl from last episode for one step.
            prev_cl = cl
            fresh_episode = False
        elif cl >= 1 and prev_cl < 1:
            l1_total += 1
            if ep_l1_step is None:
                ep_l1_step = ep_step
                score = float(BASELINE_EP) / ep_step if ep_step > 0 else 0
                print(f"  s{seed} L1@{step} ep_steps={ep_step} prog={prog_idx} "
                      f"ops={pop.ops[prog_idx]} score={score:.2f} "
                      f"best={pop.best_score:.2f} go={go}", flush=True)
        if ep_step >= MAX_EPISODE:
            # Force end episode (no death signal from game)
            do_episode_end()
            obs = env.reset(seed=seed); sub.on_reset()
            continue
        prev_cl = cl

    elapsed = time.time() - t_start
    print(f"  s{seed}: l1={l1_total} go={go} step={step} "
          f"evolve={pop.evolve_count} best_score={pop.best_score:.2f} "
          f"cells={len(sub.cells)} {elapsed:.0f}s", flush=True)
    if pop.best_ops:
        print(f"    best_ops: {pop.best_ops}", flush=True)
    return dict(seed=seed, l1=l1_total, go=go, steps=step,
                best_score=pop.best_score, best_ops=pop.best_ops,
                evolve_count=pop.evolve_count, cells=len(sub.cells))


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 577: Self-discovering substrate (R3 attack)", flush=True)
    print(f"  N_POP={N_POP} EVOLVE_EVERY={EVOLVE_EVERY} "
          f"BASELINE_EP={BASELINE_EP} MAX_STEPS={MAX_STEPS}", flush=True)

    results = []
    t_total = time.time()
    for seed in range(5):
        if time.time() - t_total > 1380:
            print("TOTAL TIME CAP HIT"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(mk, seed, time_cap=TIME_CAP)
        results.append(r)

    best_overall = max((r['best_score'] for r in results), default=0.0)
    any_l1 = sum(1 for r in results if r['l1'] > 0)
    avg_evolves = float(np.mean([r['evolve_count'] for r in results])) if results else 0

    print(f"\n{'='*60}")
    print("STEP 577: Self-discovering substrate")
    for r in results:
        print(f"  s{r['seed']}: l1={r['l1']} best_score={r['best_score']:.2f} "
              f"evolve={r['evolve_count']} cells={r['cells']}")
    print(f"\nbest_score_overall: {best_overall:.2f}")
    print(f"seeds_with_L1: {any_l1}/{len(results)}")
    print(f"avg_evolve_rounds: {avg_evolves:.1f}")

    if best_overall > 1.5:
        print("SUCCESS: Program discovered pipeline with >1.5x improvement over baseline.")
        for r in results:
            if r['best_score'] > 1.5:
                print(f"  Discovered ops: {r['best_ops']}")
    elif best_overall > 1.0:
        print(f"PARTIAL: Best score {best_overall:.2f} (some improvement, not 1.5x).")
    else:
        print("FAIL: No program achieves >1.0x baseline. Primitives insufficient or "
              "search too slow. See addendum: iterate primitive decomposition.")


if __name__ == "__main__":
    main()
