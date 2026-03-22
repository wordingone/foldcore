"""
Step 577b -- Self-discovering substrate: exploration-proxy scoring.

Refinement of 577 (Spec, 2026-03-20):
  Score by exploration rate (unique hash cells / 1000 steps)
  rather than L1 completion time. Gives 15x more evaluations per budget.

Protocol:
  1. Baseline: argmin k=16 for EVAL_STEPS steps → BASELINE_CELLS
  2. Each evaluation: run program EVAL_STEPS steps → score = cells / BASELINE_CELLS
  3. Tournament selection every N_POP evaluations (1 generation)
  4. score > PROXY_THRESHOLD → full L1 validation (15K steps)

Kill: No program scores > 1.5 after full time budget.
Signal: score > 1.5 → pipeline discovery candidate.

Primitives (same as 577):
  accumulate, argmax, threshold_C (C=1..15), label_cc, nearest, navigate (20 total)

Budget: 5 seeds, TIME_CAP=280s per seed.
At ~300 steps/sec: ~84K steps/seed → ~84 evals/seed → ~4 generations.
"""
import time
import numpy as np
import sys
from scipy.ndimage import label as ndlabel

K = 16
DIM = 256
N_A = 4
N_POP = 20
EVAL_STEPS = 1000
TIME_CAP = 280         # seconds per seed
PROXY_THRESHOLD = 1.5
L1_VALIDATE_STEPS = 15_000
L1_BASELINE_STEPS = 2000   # reference for L1 score = L1_BASELINE_STEPS / steps_to_l1

ALL_OPS = (
    [('accumulate',)] +
    [('argmax',)] +
    [('threshold', c) for c in range(1, 16)] +
    [('label_cc',)] +
    [('nearest',)] +
    [('navigate',)]
)
# 1 + 1 + 15 + 1 + 1 + 1 = 20 ops


# ── primitive executor ────────────────────────────────────────────────────────

def execute(ops, frame, buf, agent_yx):
    """
    Execute program ops against current frame.
    buf: (64,64,16) int32 accumulator (persistent for this program).
    Returns (action_or_None, updated_buf).
    """
    arr = np.array(frame[0], dtype=np.int32)
    val = arr

    for op in ops:
        name = op[0]
        try:
            if name == 'accumulate':
                r = np.arange(64)[:, None]; c = np.arange(64)[None, :]
                buf[r, c, val.clip(0, 15)] += 1
                val = arr   # side effect only; value unchanged

            elif name == 'argmax':
                if buf.max() == 0:
                    return None, buf
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
                cy, cx = val; ay, ax = agent_yx
                dy, dx = cy - ay, cx - ax
                if abs(dy) >= abs(dx):
                    return (0 if dy < 0 else 1), buf
                return (2 if dx < 0 else 3), buf

        except Exception:
            return None, buf

    return None, buf


# ── LSH substrate ─────────────────────────────────────────────────────────────

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
        self.gen_scores = [[] for _ in range(n)]  # scores in current generation
        self.all_scores = [[] for _ in range(n)]  # all scores ever (for reporting)
        self.best_score = 0.0
        self.best_ops = None
        self.evolve_count = 0
        self.total_evals = 0

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

    def record(self, prog_idx, score):
        self.gen_scores[prog_idx].append(score)
        self.all_scores[prog_idx].append(score)
        self.total_evals += 1
        if score > self.best_score:
            self.best_score = score
            self.best_ops = list(self.ops[prog_idx])

    def evolve(self):
        """Tournament: top 50% survive, bottom 50% become mutated copies."""
        avg = [float(np.mean(s)) if s else 0.0 for s in self.gen_scores]
        ranked = sorted(range(self.n), key=lambda i: avg[i], reverse=True)
        top = ranked[:self.n // 2]
        for idx in ranked[self.n // 2:]:
            parent = self.rng.choice(top)
            self.ops[idx] = self._mutate(list(self.ops[parent]))
            self.bufs[idx][:] = 0    # reset buffer for new program
            self.gen_scores[idx] = []
            self.all_scores[idx] = []
        # Keep gen_scores for top half but clear for new generation tracking
        for idx in top:
            self.gen_scores[idx] = []
        self.evolve_count += 1
        return avg


# ── evaluation helpers ────────────────────────────────────────────────────────

def run_eval(ops, buf, env, seed, n_steps, time_budget, lsh_seed=0):
    """
    Run program for n_steps. Returns (cells_visited, buf).
    buf is persistent (accumulates across evals).
    """
    sub = SubLSH(k=K, dim=DIM, seed=lsh_seed)
    obs = env.reset(seed=seed)
    sub.on_reset()
    t_start = time.time()

    for _ in range(n_steps):
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset(); continue
        if time.time() - t_start > time_budget:
            break

        sub.observe(obs)
        action, buf = execute(ops, obs, buf, sub.agent_yx)
        if action is None:
            action = sub.act()
        else:
            sub.record_action(action)

        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset(seed=seed); sub.on_reset()

    return len(sub.cells), buf


LSH_SEED_BASE = 42   # fixed hash for all evals/baseline so scores are comparable


def measure_baseline(env, seed):
    """Argmin baseline: cells visited in EVAL_STEPS steps."""
    sub = SubLSH(k=K, dim=DIM, seed=LSH_SEED_BASE)
    obs = env.reset(seed=seed)
    sub.on_reset()

    for _ in range(EVAL_STEPS):
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset(); continue
        sub.observe(obs)
        action = sub.act()
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset(seed=seed); sub.on_reset()

    return len(sub.cells)


def validate_l1(ops, env, seed, time_budget=60):
    """
    Full L1 validation after proxy hit.
    Returns (steps_to_l1 or None, l1_score).
    """
    sub = SubLSH(k=K, dim=DIM, seed=LSH_SEED_BASE)
    buf = np.zeros((64, 64, 16), dtype=np.int32)
    obs = env.reset(seed=seed)
    sub.on_reset()
    prev_cl = 0; fresh = True; ep_step = 0
    t_start = time.time()

    for total_step in range(1, L1_VALIDATE_STEPS + 1):
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset(); fresh = True; ep_step = 0; continue
        if time.time() - t_start > time_budget:
            return None, 0.0

        sub.observe(obs)
        action, buf = execute(ops, obs, buf, sub.agent_yx)
        if action is None:
            action = sub.act()
        else:
            sub.record_action(action)

        obs, _, done, info = env.step(action)
        ep_step += 1

        if done:
            obs = env.reset(seed=seed); sub.on_reset(); fresh = True; ep_step = 0; continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if fresh:
            prev_cl = cl; fresh = False
        elif cl >= 1 and prev_cl < 1:
            score = float(L1_BASELINE_STEPS) / total_step
            return total_step, score
        prev_cl = cl

    return None, 0.0


# ── tests ─────────────────────────────────────────────────────────────────────

def t0():
    """Smoke test: program executes and exploration metric runs."""
    buf = np.zeros((64, 64, 16), dtype=np.int32)
    frame = [np.zeros((64, 64), dtype=np.uint8)]
    frame[0][20:24, 30:34] = 9

    ops = [('accumulate',), ('argmax',), ('threshold', 9),
           ('label_cc',), ('nearest',), ('navigate',)]
    agent_yx = (10.0, 10.0)
    for _ in range(100):
        _, buf = execute(ops, frame, buf, agent_yx)
    action, buf = execute(ops, frame, buf, agent_yx)
    assert action is not None, f"Expected action, got None"
    print(f"T0 PASS: action={action}")

    pop = Population(seed=0)
    frame2 = [np.random.RandomState(3).randint(0, 16, (64, 64), dtype=np.uint8)]
    for i in range(pop.n):
        a, _ = execute(pop.ops[i], frame2, pop.bufs[i], (32.0, 32.0))
        assert a is None or 0 <= a < 4, f"Invalid action {a}"
    print("T0 PASS: population produces valid actions")

    # Baseline measurement smoke test (no env, just structure)
    sub = SubLSH(seed=0)
    for _ in range(10):
        f = [np.random.RandomState(0).randint(0, 16, (64, 64), dtype=np.uint8)]
        sub.observe(f)
        sub.act()
    assert len(sub.cells) > 0
    print(f"T0 PASS: SubLSH cells={len(sub.cells)}")


# ── main seed loop ────────────────────────────────────────────────────────────

def run_seed(mk, seed, time_cap=TIME_CAP):
    env = mk()
    t_start = time.time()

    # Measure baseline
    baseline_cells = measure_baseline(env, seed)
    print(f"  s{seed} baseline: {baseline_cells} cells/{EVAL_STEPS}steps", flush=True)

    pop = Population(n=N_POP, seed=seed * 7)
    total_steps = EVAL_STEPS  # baseline consumed these
    eval_count = 0
    generation = 0
    prog_idx = 0
    evals_this_gen = 0
    validated = False
    best_score = 0.0
    best_ops = None

    while time.time() - t_start < time_cap:
        time_left = time_cap - (time.time() - t_start)
        if time_left < 5:
            break

        cells, pop.bufs[prog_idx] = run_eval(
            pop.ops[prog_idx], pop.bufs[prog_idx], env, seed,
            EVAL_STEPS, min(time_left - 2, EVAL_STEPS / 100 + 5),
            lsh_seed=LSH_SEED_BASE)

        score = cells / max(baseline_cells, 1)
        pop.record(prog_idx, score)
        eval_count += 1
        total_steps += EVAL_STEPS
        evals_this_gen += 1

        if score > best_score:
            best_score = score
            best_ops = list(pop.ops[prog_idx])
            print(f"  s{seed} gen={generation} NEW BEST: prog={prog_idx} "
                  f"score={score:.2f} cells={cells} ops={best_ops}", flush=True)

        # Validate if proxy threshold hit (once per seed)
        if score >= PROXY_THRESHOLD and not validated:
            validated = True
            print(f"  s{seed} PROXY HIT! prog={prog_idx} score={score:.2f} "
                  f"-> L1 validation...", flush=True)
            time_left2 = time_cap - (time.time() - t_start)
            l1_step, l1_score = validate_l1(pop.ops[prog_idx], env, seed,
                                             time_budget=min(time_left2 - 2, 90))
            if l1_step is not None:
                print(f"  s{seed} L1 CONFIRMED: steps={l1_step} l1_score={l1_score:.2f}",
                      flush=True)
            else:
                print(f"  s{seed} L1 validation: not reached in {L1_VALIDATE_STEPS} steps",
                      flush=True)

        prog_idx = (prog_idx + 1) % N_POP

        # Evolve after all N_POP programs evaluated (1 generation)
        if evals_this_gen >= N_POP:
            avg_scores = pop.evolve()
            gen_avg = float(np.mean(avg_scores))
            generation += 1
            evals_this_gen = 0
            print(f"  s{seed} GEN {generation}: avg={gen_avg:.2f} best={pop.best_score:.2f} "
                  f"evals={eval_count} evolves={pop.evolve_count}", flush=True)

    elapsed = time.time() - t_start
    print(f"  s{seed}: best_score={pop.best_score:.2f} gen={generation} "
          f"evals={eval_count} steps={total_steps} {elapsed:.0f}s", flush=True)
    if pop.best_ops:
        print(f"    best_ops: {pop.best_ops}", flush=True)

    return dict(seed=seed, best_score=pop.best_score, best_ops=pop.best_ops,
                generation=generation, evals=eval_count, steps=total_steps,
                baseline_cells=baseline_cells)


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 577b: Self-discovering substrate (exploration proxy)", flush=True)
    print(f"  N_POP={N_POP} EVAL_STEPS={EVAL_STEPS} "
          f"PROXY_THRESHOLD={PROXY_THRESHOLD}", flush=True)

    results = []
    t_total = time.time()
    for seed in range(5):
        if time.time() - t_total > 1380:
            print("TOTAL TIME CAP HIT"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(mk, seed, time_cap=TIME_CAP)
        results.append(r)

    best_overall = max((r['best_score'] for r in results), default=0.0)
    avg_gens = float(np.mean([r['generation'] for r in results])) if results else 0
    avg_evals = float(np.mean([r['evals'] for r in results])) if results else 0

    print(f"\n{'='*60}")
    print("STEP 577b: Self-discovering substrate (exploration proxy)")
    for r in results:
        print(f"  s{r['seed']}: best_score={r['best_score']:.2f} "
              f"gen={r['generation']} evals={r['evals']} "
              f"baseline_cells={r['baseline_cells']}")
    print(f"\nbest_score_overall: {best_overall:.2f}")
    print(f"avg_generations: {avg_gens:.1f}")
    print(f"avg_evals_per_seed: {avg_evals:.0f}")

    if best_overall >= PROXY_THRESHOLD:
        print(f"PROXY HIT: Program found with exploration score >= {PROXY_THRESHOLD}.")
        for r in results:
            if r['best_score'] >= PROXY_THRESHOLD:
                print(f"  Discovered ops: {r['best_ops']}")
    elif best_overall > 1.0:
        print(f"PARTIAL: Best exploration score {best_overall:.2f} "
              f"(some improvement, not {PROXY_THRESHOLD}x).")
    else:
        print("FAIL: No program exceeds baseline exploration. "
              "Primitives insufficient or mode map needs more frames. "
              "See addendum: reduce EVAL_STEPS or increase persistent buffer warmup.")


if __name__ == "__main__":
    main()
