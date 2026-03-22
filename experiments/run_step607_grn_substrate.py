"""
Step 607 -- GRN Substrate on LS20 (Leo's step 606 spec).

Gene Regulatory Network substrate: N=4 competing LSH encodings with
selection pressure and mutual inhibition.

Mechanism:
- N=4 independent LSH projections {H_1..H_N} (different random hyperplanes, same k=12)
- Each H_i builds its own argmin graph G_i
- Action selection: majority vote across H_i's (weighted by vote_weight[i])
- Selection pressure after death: suppress H_i with FEWEST unique cells last episode
  (its vote_weight drops to 0)
- After level-up: amplify H_i with MOST unique cells (vote_weight doubles)
- Suppressed H_i's still observe/update but don't vote
- Birth: every 5K steps, spawn new H_i with fresh random hyperplanes
- Prune: H_i with weight < 0.1 for > 1K steps (implemented as reset)

R3 test: do encodings select different actions for different game states?
Kill: action agreement > 95% across all H_i's by step 10K
Signal: diverse actions AND level-up achieved

Not codebook: no cosine match, no attract update, no spawn-on-similarity.
Self-modification is which encoding is active, not the encoding itself.

Protocol: 5 seeds x 50K steps (5-min cap), LS20
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

CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]


def encode(frame, H):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    x -= x.mean()
    bits = (H @ x > 0).astype(np.uint8)
    return int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)


class Encoding:
    def __init__(self, seed):
        self.H = np.random.RandomState(seed).randn(K, DIM).astype(np.float32)
        self.G = {}         # argmin graph
        self.cells_episode = set()  # unique cells this episode
        self.cells_total = set()    # unique cells ever
        self.vote_weight = 1.0
        self.suppressed_steps = 0
        self.cn = None      # current node

    def observe(self, frame):
        n = encode(frame, self.H)
        self.cells_episode.add(n)
        self.cells_total.add(n)
        self.cn = n

    def vote(self, pn, pa):
        """Return (cx, cy, action_index) for argmin."""
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


class SubGRN:
    def __init__(self, seed=0):
        np.random.seed(seed)
        rng = np.random.RandomState(seed)
        self.encodings = [Encoding(rng.randint(0, 2**31)) for _ in range(N_ENCODINGS)]
        self.seed_counter = rng.randint(1000, 10000)
        self._pa = None
        self._pn_per_enc = [None] * len(self.encodings)
        self._pa_per_enc = [None] * len(self.encodings)
        self.step_count = 0
        self.game_level = 0

        # Tracking
        self.action_log = []  # (step, active_enc_idx, action) for agreement analysis

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
                    # Reset suppressed encoding with fresh hyperplanes
                    self.seed_counter += 1
                    enc.H = np.random.RandomState(self.seed_counter).randn(K, DIM).astype(np.float32)
                    enc.G = {}
                    enc.vote_weight = 0.5
                    enc.suppressed_steps = 0
            else:
                enc.suppressed_steps = 0

        # Collect votes from each encoding
        votes = []
        for i, enc in enumerate(self.encodings):
            a = enc.vote(self._pn_per_enc[i], self._pa_per_enc[i])
            votes.append((i, a, enc.vote_weight))

        # Weighted majority vote (argmax of summed weights per action)
        action_weights = np.zeros(N_CLICKS)
        for i, a, w in votes:
            action_weights[a] += w

        # Win: action with highest total weight
        best_a = int(np.argmax(action_weights))

        # Record chosen action in each encoding's graph
        for i, enc in enumerate(self.encodings):
            if enc.cn is not None:
                enc.record(self._pn_per_enc[i], self._pa_per_enc[i], enc.cn)
            self._pn_per_enc[i] = enc.cn
            self._pa_per_enc[i] = best_a  # all use same action

        # Agreement metric: fraction of encodings that voted for best_a
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
        """Suppress encoding with fewest unique cells this episode."""
        active = [enc for enc in self.encodings if enc.vote_weight >= 0.1]
        if len(active) > 1:
            worst = min(active, key=lambda e: len(e.cells_episode))
            worst.vote_weight = 0.0
        for enc in self.encodings:
            enc.on_reset()
        self._pn_per_enc = [None] * len(self.encodings)
        self._pa_per_enc = [None] * len(self.encodings)
        self.game_level = 0

    def on_level_up(self, new_lvl):
        """Amplify encoding with most unique cells in this episode."""
        self.game_level = new_lvl
        active = [enc for enc in self.encodings if enc.vote_weight >= 0.1]
        if active:
            best = max(active, key=lambda e: len(e.cells_episode))
            best.vote_weight = min(4.0, best.vote_weight * 2)
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
    sub = SubGRN(seed=0)
    assert len(sub.encodings) == N_ENCODINGS
    assert all(enc.vote_weight == 1.0 for enc in sub.encodings)
    sub.on_death()
    # After death: one encoding should be suppressed
    # (not guaranteed with only empty episode sets, but check state resets)
    assert all(enc.cn is None for enc in sub.encodings)
    print("T0 PASS", flush=True)


def run_seed(arc, game_id, seed):
    from arcengine import GameState
    np.random.seed(seed)

    env = arc.make(game_id)
    action6 = env.action_space[0]
    sub = SubGRN(seed=seed * 100)
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

        # Kill criterion check at step 10K
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
                print(f"  s{seed} L1@{ts} go={go} weights={sub.weight_summary()}",
                      flush=True)
            if new_lvl >= 2 and l2_step is None:
                l2_step = ts
                print(f"  s{seed} L2@{ts}!! weights={sub.weight_summary()}", flush=True)
            prev_lvls = new_lvl

        if time.time() - t_start > TIME_CAP:
            mean_agree, std_agree = sub.agreement_stats()
            print(f"  s{seed} cap@{ts} go={go} agree={mean_agree:.3f}±{std_agree:.3f} "
                  f"weights={sub.weight_summary()}", flush=True)
            break

    mean_agree, std_agree = sub.agreement_stats()
    status = f"L2@{l2_step}" if l2_step else (f"L1@{l1_step}" if l1_step else "---")
    print(f"  s{seed}: {status}  go={go}  agree={mean_agree:.3f}  "
          f"encs={len(sub.encodings)}  weights={sub.weight_summary()}", flush=True)
    return dict(seed=seed, l1=l1_step, l2=l2_step, go=go, ts=ts,
                agreement=mean_agree, n_encs=len(sub.encodings))


def main():
    t0()

    import arc_agi
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    ls20 = next((e for e in envs if 'ls20' in e.game_id.lower()), None)
    if ls20 is None:
        print("SKIP -- LS20 not found"); return

    print(f"Step 607: GRN Substrate on LS20", flush=True)
    print(f"  game={ls20.game_id}  N_enc={N_ENCODINGS}  birth@{BIRTH_INTERVAL}steps",
          flush=True)
    print(f"  R3 test: does selection pressure diversify encodings?", flush=True)

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
    killed = sum(1 for r in results if r.get('kill', False))

    print(f"\n{'='*60}", flush=True)
    print(f"Step 607: GRN Substrate (LS20)", flush=True)
    print(f"  L1: {l1_wins}/{len(results)}  L2: {l2_wins}/{len(results)}", flush=True)
    print(f"  avg agreement: {avg_agree:.3f}", flush=True)
    if avg_agree > 0.95:
        print("  KILL: Selection pressure not working — all encodings agree.", flush=True)
    elif l1_wins >= 3:
        print("  SIGNAL: GRN reaches L1 with diverse encodings.", flush=True)
    else:
        print("  L1 not reached.", flush=True)


if __name__ == "__main__":
    main()
