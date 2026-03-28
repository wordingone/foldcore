"""
Step 1279 — I3 artifact diagnostic: random action permutation on LS20.

Tests whether CTL I3=0.64 on LS20 is an index-ordering artifact.

Standard CTL (argmin) breaks ties by returning the lowest-index action.
On LS20 with 7 actions, at step 200 this produces counts [29,29,29,29,28,28,28].
If kb_delta for LS20 is monotone in action index, I3=0.64 is an artifact.

PermutedCTL: same argmin but tie-breaking follows a random permutation order
instead of index order. If I3 drops to ~0, the criterion is measuring index
correlation, not coverage quality.

Conditions:
  control_c         — standard pure argmin (index-ordered ties)
  control_c_permuted — argmin with random tie-break permutation

Game: LS20 only.
Draws: 5 per condition (10 runs).

Spec: Leo mail 3564, 2026-03-28.
"""
import sys, os, time, json
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')

import numpy as np
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
MAX_STEPS = 10_000
MAX_SECONDS = 300
I3_STEP = 200

DIAG_DIR = os.path.join('B:/M/the-search/experiments', 'results', 'game_diagnostics')
RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1279')


def make_game(game_name: str):
    try:
        import arcagi3
        return arcagi3.make(game_name.upper())
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game_name.upper())


def spearman_rho(x, y):
    if len(x) < 2 or len(y) < 2:
        return None
    rx = np.argsort(np.argsort(x)).astype(np.float32)
    ry = np.argsort(np.argsort(y)).astype(np.float32)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom < 1e-8:
        return None
    return float(np.dot(rx, ry) / denom)


class ControlC:
    """Standard pure argmin — lowest-index tie breaking."""
    def __init__(self, n_actions, seed):
        self.n_actions = n_actions
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.action_counts = np.zeros(n_actions, np.float32)
        self._prev_repr = None
        self._prev_enc = None

    def _encode(self, obs):
        x = _enc_frame(np.asarray(obs, dtype=np.float32))
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def process(self, obs_raw):
        enc = self._encode(obs_raw)
        self._prev_repr = enc.copy()
        self._prev_enc = enc.copy()
        action = int(np.argmin(self.action_counts))
        self.action_counts[action] += 1
        return action

    def update_flow(self, next_obs): pass
    def on_level_transition(self): pass


class PermutedControlC:
    """Pure argmin with random permutation tie-breaking.

    Tie-breaking order = random permutation of action indices.
    Among tied minimum-count actions, picks the one that comes
    first in perm (random order) rather than the lowest index.
    action_counts still indexed by game action space for I3 comparison.
    """
    def __init__(self, n_actions, seed):
        self.n_actions = n_actions
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.action_counts = np.zeros(n_actions, np.float32)
        self._prev_repr = None
        self._prev_enc = None
        # Random permutation determines tie-breaking order
        rng_p = np.random.RandomState(seed + 77777)
        self._perm = rng_p.permutation(n_actions)  # perm[internal_rank] = game_action
        # Build inverse: which rank does each game_action have in perm?
        self._perm_rank = np.empty(n_actions, dtype=np.int64)
        for rank, ga in enumerate(self._perm):
            self._perm_rank[ga] = rank

    def _encode(self, obs):
        x = _enc_frame(np.asarray(obs, dtype=np.float32))
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def process(self, obs_raw):
        enc = self._encode(obs_raw)
        self._prev_repr = enc.copy()
        self._prev_enc = enc.copy()
        # Among minimum-count game actions, pick the one with lowest rank in perm
        min_count = self.action_counts.min()
        candidates = np.where(self.action_counts == min_count)[0]
        # Rank each candidate by their position in perm
        ranks = self._perm_rank[candidates]
        action = int(candidates[ranks.argmin()])
        self.action_counts[action] += 1
        return action

    def update_flow(self, next_obs): pass
    def on_level_transition(self): pass


def run_single(condition, draw, seed, n_actions, kb_delta):
    if condition == 'control_c':
        substrate = ControlC(n_actions, seed)
    elif condition == 'control_c_permuted':
        substrate = PermutedControlC(n_actions, seed)
    else:
        raise ValueError(condition)

    env = make_game('ls20')
    obs = env.reset(seed=seed)
    steps = 0
    level = 0
    max_level = 0
    level_first_step = {}
    i3_action_counts = None
    fresh_episode = True
    t_start = time.time()

    while steps < MAX_STEPS:
        if time.time() - t_start > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            continue

        obs_arr = np.asarray(obs, dtype=np.float32)
        if steps == I3_STEP:
            i3_action_counts = substrate.action_counts[:min(7, n_actions)].copy()

        action = substrate.process(obs_arr) % n_actions
        obs, reward, done, info = env.step(action)
        steps += 1

        if fresh_episode:
            fresh_episode = False
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl not in level_first_step:
                level_first_step[cl] = steps
            if cl > max_level:
                max_level = cl
            level = cl
            substrate.on_level_transition()

        if done:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            level = 0

    elapsed = time.time() - t_start

    i3_rho, i3_pass = None, None
    if i3_action_counts is not None:
        n = min(len(i3_action_counts), 7)
        freq = i3_action_counts[:n].astype(np.float32)
        ref = kb_delta[:n].astype(np.float32)
        if ref.max() - ref.min() > 1e-6:
            rho = spearman_rho(freq, ref)
            if rho is not None:
                i3_rho = round(rho, 4)
                i3_pass = bool(rho > 0.5)

    label = 'CTLp' if 'permuted' in condition else 'CTL'
    # Print action count distribution at step 200
    counts_str = str(i3_action_counts[:7].astype(int).tolist()) if i3_action_counts is not None else "null"
    print(f"  LS20 | {condition} | draw={draw} | [{label}] I3ρ={i3_rho} pass={i3_pass} counts={counts_str} | {elapsed:.1f}s")

    return {
        'game': 'ls20',
        'condition': condition,
        'draw': draw,
        'seed': seed,
        'max_level': max_level,
        'L1_solved': bool(level_first_step.get(1) is not None),
        'I3_spearman_rho': i3_rho,
        'I3_pass': i3_pass,
        'action_counts_at_200': i3_action_counts[:7].tolist() if i3_action_counts is not None else None,
        'elapsed_seconds': round(elapsed, 2),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(os.path.join(DIAG_DIR, 'ls20_diagnostic.json')) as f:
        diag = json.load(f)
    kb = diag.get('kb_responsiveness', {})
    kb_delta = np.zeros(7, np.float32)
    for i, key in enumerate([f'ACTION{j}' for j in range(1, 8)]):
        if key in kb:
            kb_delta[i] = kb[key].get('delta_mean', 0.0)

    n_actions = diag.get('n_actions', 7)

    print("=" * 60)
    print("STEP 1279 — I3 ARTIFACT DIAGNOSTIC (LS20 only)")
    print("Testing: CTL I3=0.64 is index-ordering artifact?")
    print(f"LS20 kb_delta[:7] = {kb_delta.tolist()}")
    print("=" * 60)

    results = []
    for condition in ['control_c', 'control_c_permuted']:
        print(f"\nCondition: {condition}")
        for draw in range(1, 6):
            seed = draw * 100 + (0 if condition == 'control_c' else 50)
            r = run_single(condition, draw, seed, n_actions, kb_delta)
            results.append(r)

    print("\n" + "=" * 60)
    print("SUMMARY")
    for cond in ['control_c', 'control_c_permuted']:
        runs = [r for r in results if r['condition'] == cond]
        rhos = [r['I3_spearman_rho'] for r in runs if r['I3_spearman_rho'] is not None]
        passes = sum(1 for r in runs if r.get('I3_pass') == True)
        mean_rho = float(np.mean(rhos)) if rhos else 0.0
        label = 'CTL (index order)' if cond == 'control_c' else 'CTL (random perm)'
        print(f"  {label}: I3={mean_rho:.3f} ({passes}/{len(runs)} pass)")
        for r in runs:
            print(f"    draw={r['draw']} I3={r['I3_spearman_rho']} counts={r['action_counts_at_200']}")

    with open(os.path.join(RESULTS_DIR, 'step1279_permutation_diag.json'), 'w') as f:
        json.dump({'results': results, 'kb_delta': kb_delta.tolist()}, f, indent=2)

    print("\nDIAGNOSTIC DONE")


if __name__ == '__main__':
    main()
