"""
Step 609 -- GRN + Somatic Hypermutation on LS20.

Control: step 607 = selection-only (L1:0/5, agree=0.248, go=0).
This: selection + somatic hypermutation.

After selection picks best π_i (amplified), mutate it:
  h_new = h_old + N(0, sigma=0.1) per hyperplane row
Child encoding replaces the lowest-weight encoding (affinity maturation).

Mechanism:
- Same GRN substrate as 607: N=4 encodings, weighted vote, selection pressure
- After level-up (amplify best), mutate the best encoding:
  H_mutant = H_best + N(0, sigma=0.1) * |H_best|.mean() [scale-relative noise]
  The mutant replaces the lowest-weight encoding
- After death (suppress worst), mutate the second-best:
  H_mutant = H_survivor + N(0, sigma=0.1)
  Replaces the suppressed encoding (instead of random reset)

R3 test: does mutation+selection produce genuinely better encodings than selection alone?
Kill: action agreement > 0.95 at 10K (diversity collapse)
       OR L1=0/5 (same as 607 control)
Signal: L1 >= 1/5 with diverse encodings (agree < 0.7)

Compare: 607 (selection only) vs 609 (selection+mutation)
Control baseline: L1:0/5, agree=0.248

Protocol: 5 seeds x 50K steps (60s/seed cap), LS20.
"""
import time
import logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)

K = 12
DIM = 256
N_CLICKS = 64
N_ENCODINGS = 4
BIRTH_INTERVAL = 5000
MAX_ENCODINGS = 8
MAX_STEPS = 50_000
TIME_CAP = 60
SIGMA = 0.1  # mutation noise relative to row L2 norm

CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]


def encode(frame, H):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    x -= x.mean()
    bits = (H @ x > 0).astype(np.uint8)
    return int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)


def mutate_hyperplanes(H, rng, sigma=SIGMA):
    """Somatic hypermutation: add Gaussian noise scaled to each row's norm."""
    norms = np.linalg.norm(H, axis=1, keepdims=True)  # (K, 1)
    noise = rng.randn(*H.shape).astype(np.float32) * sigma * (norms + 1e-6)
    return H + noise


class Encoding:
    def __init__(self, seed):
        self.H = np.random.RandomState(seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self.cells_episode = set()
        self.cells_total = set()
        self.vote_weight = 1.0
        self.suppressed_steps = 0
        self.cn = None

    def observe(self, frame):
        n = encode(frame, self.H)
        self.cells_episode.add(n)
        self.cells_total.add(n)
        self.cn = n

    def vote(self, pn, pa):
        if self.cn is None:
            a = int(np.random.randint(N_CLICKS))
        else:
            counts = [sum(self.G.get((self.cn, a), {}).values())
                      for a in range(N_CLICKS)]
            min_c = min(counts)
            cands = [a for a, c in enumerate(counts) if c == min_c]
            a = cands[int(np.random.randint(len(cands)))]
        return a

    def record(self, pn, pa, cn):
        if pn is not None and pa is not None:
            d = self.G.setdefault((pn, pa), {})
            d[cn] = d.get(cn, 0) + 1

    def on_reset(self):
        self.cells_episode = set()
        self.cn = None

    def on_level_up(self):
        self.cells_episode = set()
        self.cn = None


class SubGRNMutate:
    def __init__(self, seed=0):
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        rng = np.random.RandomState(seed)
        self.encodings = [Encoding(rng.randint(0, 2**31)) for _ in range(N_ENCODINGS)]
        self.seed_counter = rng.randint(1000, 10000)
        self._pa = None
        self._pn_per_enc = [None] * len(self.encodings)
        self._pa_per_enc = [None] * len(self.encodings)
        self.step_count = 0
        self.game_level = 0
        self.action_log = []
        self.mutation_count = 0

    def observe(self, frame):
        for enc in self.encodings:
            enc.observe(frame)

    def act(self):
        self.step_count += 1

        # Birth: every BIRTH_INTERVAL steps, spawn new encoding
        if self.step_count % BIRTH_INTERVAL == 0 and len(self.encodings) < MAX_ENCODINGS:
            self.seed_counter += 1
            new_enc = Encoding(self.seed_counter)
            new_enc.vote_weight = 0.5
            self.encodings.append(new_enc)
            self._pn_per_enc.append(None)
            self._pa_per_enc.append(None)

        # Update suppression counters
        for enc in self.encodings:
            if enc.vote_weight < 0.1:
                enc.suppressed_steps += 1
                if enc.suppressed_steps > 1000:
                    # Mutate a surviving encoding instead of random reset
                    active = [e for e in self.encodings if e.vote_weight >= 0.1]
                    if active:
                        donor = max(active, key=lambda e: len(e.cells_total))
                        enc.H = mutate_hyperplanes(donor.H, self.rng)
                    else:
                        self.seed_counter += 1
                        enc.H = np.random.RandomState(self.seed_counter).randn(K, DIM).astype(np.float32)
                    enc.G = {}
                    enc.vote_weight = 0.5
                    enc.suppressed_steps = 0
                    self.mutation_count += 1
            else:
                enc.suppressed_steps = 0

        # Collect votes
        votes = []
        for i, enc in enumerate(self.encodings):
            a = enc.vote(self._pn_per_enc[i], self._pa_per_enc[i])
            votes.append((i, a, enc.vote_weight))

        # Weighted majority vote
        action_weights = np.zeros(N_CLICKS)
        for i, a, w in votes:
            action_weights[a] += w
        best_a = int(np.argmax(action_weights))

        # Record chosen action
        for i, enc in enumerate(self.encodings):
            if enc.cn is not None:
                enc.record(self._pn_per_enc[i], self._pa_per_enc[i], enc.cn)
            self._pn_per_enc[i] = enc.cn
            self._pa_per_enc[i] = best_a

        # Agreement metric
        active = [enc for enc in self.encodings if enc.vote_weight >= 0.1]
        if active:
            agree = sum(1 for i, a, w in votes
                       if a == best_a and self.encodings[i].vote_weight >= 0.1)
            agreement = agree / len(active)
        else:
            agreement = 1.0
        self.action_log.append(agreement)

        cx, cy = CLICK_GRID[best_a]
        return cx, cy, best_a, agreement

    def on_death(self):
        """Suppress worst. Mutate second-best to replace suppressed slot."""
        active = [enc for enc in self.encodings if enc.vote_weight >= 0.1]
        worst = None
        if len(active) > 1:
            worst = min(active, key=lambda e: len(e.cells_episode))
            worst.vote_weight = 0.0
            # Somatic hypermutation: mutate second-best to create replacement
            remaining = [e for e in active if e is not worst]
            if remaining:
                donor = max(remaining, key=lambda e: len(e.cells_episode))
                worst.H = mutate_hyperplanes(donor.H, self.rng)
                worst.G = {}
                worst.suppressed_steps = 0
                # Keep vote_weight at 0.0 until suppressed_steps > 1000 triggers revival
                # Actually: set weight to 0.3 so it competes weakly right away
                worst.vote_weight = 0.3
                self.mutation_count += 1

        for enc in self.encodings:
            enc.on_reset()
        self._pn_per_enc = [None] * len(self.encodings)
        self._pa_per_enc = [None] * len(self.encodings)
        self.game_level = 0

    def on_level_up(self, new_lvl):
        """Amplify best. Create mutant child of best to replace lowest-weight."""
        self.game_level = new_lvl
        active = [enc for enc in self.encodings if enc.vote_weight >= 0.1]
        best = None
        if active:
            best = max(active, key=lambda e: len(e.cells_episode))
            best.vote_weight = min(4.0, best.vote_weight * 2)
            # Somatic hypermutation: spawn mutant child of best
            if len(self.encodings) < MAX_ENCODINGS:
                mutant = Encoding(self.rng.randint(0, 2**31))
                mutant.H = mutate_hyperplanes(best.H, self.rng)
                mutant.vote_weight = 0.5
                self.encodings.append(mutant)
                self._pn_per_enc.append(None)
                self._pa_per_enc.append(None)
                self.mutation_count += 1
            else:
                # Replace lowest-weight with mutant
                worst = min(self.encodings, key=lambda e: e.vote_weight)
                worst.H = mutate_hyperplanes(best.H, self.rng)
                worst.G = {}
                worst.vote_weight = 0.5
                worst.suppressed_steps = 0
                self.mutation_count += 1

        for enc in self.encodings:
            enc.on_level_up()
        self._pn_per_enc = [None] * len(self.encodings)
        self._pa_per_enc = [None] * len(self.encodings)

    def agreement_stats(self):
        if not self.action_log:
            return 0.0, 0.0
        recent = self.action_log[-1000:] if len(self.action_log) >= 1000 else self.action_log
        return float(np.mean(recent)), float(np.std(recent))

    def weight_summary(self):
        return [f"{enc.vote_weight:.2f}" for enc in self.encodings]


def t0():
    sub = SubGRNMutate(seed=0)
    assert len(sub.encodings) == N_ENCODINGS
    assert all(enc.vote_weight == 1.0 for enc in sub.encodings)
    # Test mutation: after level-up, should have a mutant
    sub.on_level_up(1)
    assert len(sub.encodings) == N_ENCODINGS + 1, f"Expected {N_ENCODINGS+1}, got {len(sub.encodings)}"
    assert sub.mutation_count >= 1
    print("T0 PASS", flush=True)


def run_seed(arc, game_id, seed):
    from arcengine import GameState
    np.random.seed(seed)

    env = arc.make(game_id)
    action6 = env.action_space[0]
    sub = SubGRNMutate(seed=seed * 100)
    obs = env.reset()

    ts = go = 0
    prev_lvls = 0
    l1_step = l2_step = None
    t_start = time.time()
    kill_triggered = False

    while ts < MAX_STEPS:
        if obs is None:
            obs = env.reset(); sub.on_death(); prev_lvls = 0; continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); sub.on_death(); prev_lvls = 0; continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset(); sub.on_death(); prev_lvls = 0; continue

        sub.observe(obs.frame)
        cx, cy, a, agreement = sub.act()

        if ts == 10000 and not kill_triggered:
            mean_agree, _ = sub.agreement_stats()
            if mean_agree > 0.95:
                print(f"  s{seed} KILL@{ts}: agreement={mean_agree:.3f} > 0.95", flush=True)
                kill_triggered = True

        lvls_before = obs.levels_completed
        obs = env.step(action6, data={"x": cx, "y": cy})
        ts += 1

        if obs is None:
            break

        if obs.levels_completed > lvls_before:
            new_lvl = obs.levels_completed
            sub.on_level_up(new_lvl)
            if new_lvl >= 1 and l1_step is None:
                l1_step = ts
                print(f"  s{seed} L1@{ts} go={go} mut={sub.mutation_count} "
                      f"weights={sub.weight_summary()}", flush=True)
            if new_lvl >= 2 and l2_step is None:
                l2_step = ts
                print(f"  s{seed} L2@{ts}!! mut={sub.mutation_count} "
                      f"weights={sub.weight_summary()}", flush=True)
            prev_lvls = new_lvl

        if time.time() - t_start > TIME_CAP:
            mean_agree, std_agree = sub.agreement_stats()
            print(f"  s{seed} cap@{ts} go={go} agree={mean_agree:.3f}±{std_agree:.3f} "
                  f"mut={sub.mutation_count} weights={sub.weight_summary()}", flush=True)
            break

    mean_agree, std_agree = sub.agreement_stats()
    status = f"L2@{l2_step}" if l2_step else (f"L1@{l1_step}" if l1_step else "---")
    print(f"  s{seed}: {status}  go={go}  agree={mean_agree:.3f}  "
          f"mut={sub.mutation_count}  encs={len(sub.encodings)}  "
          f"weights={sub.weight_summary()}", flush=True)
    return dict(seed=seed, l1=l1_step, l2=l2_step, go=go, ts=ts,
                agreement=mean_agree, n_encs=len(sub.encodings),
                mutations=sub.mutation_count)


def main():
    t0()

    import arc_agi
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    ls20 = next((e for e in envs if 'ls20' in e.game_id.lower()), None)
    if ls20 is None:
        print("SKIP -- LS20 not found"); return

    print(f"Step 609: GRN + Somatic Hypermutation on LS20", flush=True)
    print(f"  game={ls20.game_id}  K={K}  sigma={SIGMA}  N_enc={N_ENCODINGS}", flush=True)
    print(f"  Control (step 607): L1:0/5, agree=0.248", flush=True)
    print(f"  R3 test: does mutation+selection beat selection alone?", flush=True)

    results = []
    t_total = time.time()

    for seed in range(5):
        if time.time() - t_total > 295:
            print("TOTAL TIME CAP"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(arc, ls20.game_id, seed)
        results.append(r)

    l1_wins = sum(1 for r in results if r['l1'])
    l2_wins = sum(1 for r in results if r['l2'])
    avg_agree = np.mean([r['agreement'] for r in results]) if results else 0
    avg_mut = np.mean([r['mutations'] for r in results]) if results else 0

    print(f"\n{'='*60}", flush=True)
    print(f"Step 609: GRN + Somatic Hypermutation (LS20)", flush=True)
    print(f"  L1: {l1_wins}/{len(results)}  L2: {l2_wins}/{len(results)}", flush=True)
    print(f"  avg agreement: {avg_agree:.3f}  avg mutations: {avg_mut:.1f}", flush=True)
    print(f"  Compare 607: L1:0/5, agree=0.248, mutations=0", flush=True)
    if avg_agree > 0.95:
        print("  KILL: Diversity collapse — mutation converged encodings.", flush=True)
    elif l1_wins >= 1:
        print("  SIGNAL: Mutation+selection reaches L1. Better than selection alone.", flush=True)
    elif l1_wins == 0 and avg_agree < 0.5:
        print("  NULL: Diverse but no wins — same as 607. Mutation doesn't help navigation.", flush=True)
    else:
        print("  L1 not reached.", flush=True)


if __name__ == "__main__":
    main()
