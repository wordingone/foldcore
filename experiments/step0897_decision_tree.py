"""
step0897_decision_tree.py -- Online Decision Tree forward predictor.

R3 hypothesis: symbolic tree structure captures transition dynamics.
Tree growth (splits on prediction error) IS self-modification of operations.

Architecture:
- Binary decision tree, keyed by binarized enc (enc > 0 → 1, else 0).
- Each leaf: {action → (sum_next, count)}, sample buffer for split decisions.
- Split condition: leaf has >MIN_SAMPLES samples AND mean prediction error >SPLIT_THRESH.
- Split feature: highest variance-reduction feature over buffered samples.
- Tree depth = complexity of learned dynamics model.

Action selection: for each action, find tree leaf for predicted next obs (momentum step).
Pick action whose leaf has fewest samples per action (least familiar = most novel).
R2 compliant: splits driven by prediction error, fully self-supervised.

R3_cf protocol: cold (fresh tree) vs warm (full tree transfer), seeds 6-10, 10K steps.
Pretrain: seeds 1-5, 5K steps each.
Kill criterion: tree depth < 4 at 10K steps.
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

PRETRAIN_SEEDS = list(range(1, 6))
TEST_SEEDS = list(range(6, 11))
PRETRAIN_STEPS = 5_000
TEST_STEPS = 10_000
N_ACTIONS = 4
SPLIT_THRESH = 0.3
MIN_SAMPLES = 20
MAX_DEPTH = 16
ENC_DIM = 256
EPSILON = 0.20


class TreeNode:
    """Binary decision tree node."""
    def __init__(self):
        self.is_leaf = True
        self.feature = None; self.threshold = 0.0
        self.left = None; self.right = None
        # Leaf data: per-action prediction sums and counts
        self.action_sums = [np.zeros(ENC_DIM, np.float32) for _ in range(N_ACTIONS)]
        self.action_counts = np.zeros(N_ACTIONS, int)
        self.samples = []  # (obs_bin, action, next_enc) for split decisions
        self.depth = 0

    def predict(self, action):
        if self.action_counts[action] == 0:
            return None
        return self.action_sums[action] / self.action_counts[action]

    def add_sample(self, obs_bin, action, next_enc):
        self.action_sums[action] += next_enc
        self.action_counts[action] += 1
        if len(self.samples) < 200:
            self.samples.append((obs_bin, action, next_enc))

    def total_count(self):
        return int(self.action_counts.sum())

    def mean_pred_error(self):
        errors = []
        for obs_bin, action, next_enc in self.samples:
            pred = self.predict(action)
            if pred is not None:
                errors.append(float(np.mean(np.abs(next_enc - pred))))
        return np.mean(errors) if errors else 0.0

    def best_split_feature(self):
        if len(self.samples) < MIN_SAMPLES:
            return None, None
        obs_arr = np.array([s[0] for s in self.samples], dtype=np.float32)
        next_arr = np.array([s[2] for s in self.samples], dtype=np.float32)
        baseline_var = np.var(next_arr, axis=0).sum()
        best_gain = 0.0; best_f = 0
        # Sample a subset of features to check (speed optimization)
        features_to_check = np.random.choice(ENC_DIM, min(32, ENC_DIM), replace=False)
        for f in features_to_check:
            threshold = np.median(obs_arr[:, f])
            mask = obs_arr[:, f] >= threshold
            if mask.sum() < 3 or (~mask).sum() < 3:
                continue
            var_left = np.var(next_arr[~mask], axis=0).sum() if (~mask).sum() > 1 else 0
            var_right = np.var(next_arr[mask], axis=0).sum() if mask.sum() > 1 else 0
            n = len(self.samples)
            gain = baseline_var - (var_left * (~mask).sum() + var_right * mask.sum()) / n
            if gain > best_gain:
                best_gain = gain; best_f = f
                best_thresh = threshold
        return (best_f, best_thresh) if best_gain > 0 else (None, None)


class OnlineDecisionTree897(BaseSubstrate):
    """Online decision tree predictor. Splits on prediction error."""

    def __init__(self, n_actions=N_ACTIONS, seed=0, epsilon=EPSILON):
        self._n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self._epsilon = epsilon
        self._root = TreeNode()
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_enc_bin = None
        self._prev_action = None; self._last_enc = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self._running_mean = (1 - alpha) * self._running_mean + alpha * enc_raw
        return enc_raw - self._running_mean

    def _binarize(self, enc):
        return (enc > 0).astype(np.float32)

    def _traverse(self, obs_bin):
        node = self._root
        while not node.is_leaf:
            node = node.right if obs_bin[node.feature] >= node.threshold else node.left
        return node

    def _maybe_split(self, node):
        if (node.depth >= MAX_DEPTH or node.total_count() < MIN_SAMPLES):
            return
        if node.mean_pred_error() > SPLIT_THRESH:
            f, t = node.best_split_feature()
            if f is None:
                return
            node.is_leaf = False; node.feature = f; node.threshold = t
            node.left = TreeNode(); node.right = TreeNode()
            node.left.depth = node.right.depth = node.depth + 1
            # Redistribute samples
            for obs_bin, action, next_enc in node.samples:
                child = node.right if obs_bin[f] >= t else node.left
                child.add_sample(obs_bin, action, next_enc)
            node.samples = []

    def max_depth(self):
        def _depth(n):
            if n is None or n.is_leaf: return n.depth if n else 0
            return max(_depth(n.left), _depth(n.right))
        return _depth(self._root)

    def n_leaves(self):
        def _count(n):
            if n is None: return 0
            if n.is_leaf: return 1
            return _count(n.left) + _count(n.right)
        return _count(self._root)

    def process(self, observation):
        enc = self._encode(observation)
        self._last_enc = enc
        enc_bin = self._binarize(enc)

        # Update tree with previous transition
        if self._prev_enc_bin is not None and self._prev_action is not None:
            leaf = self._traverse(self._prev_enc_bin)
            leaf.add_sample(self._prev_enc_bin, self._prev_action, enc)
            self._maybe_split(leaf)

        # Action selection: momentum prediction → leaf → fewest samples
        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            # Simple momentum: next ≈ current
            best_a = 0; best_score = -1.0
            for a in range(self._n_actions):
                # Predict next enc: use tree prediction if available, else current enc
                leaf = self._traverse(enc_bin)
                pred = leaf.predict(a)
                if pred is None:
                    score = 1.0 / 1.0  # unknown = novel
                else:
                    pred_bin = self._binarize(pred)
                    pred_leaf = self._traverse(pred_bin)
                    score = 1.0 / (pred_leaf.total_count() + 1.0)
                if score > best_score:
                    best_score = score; best_a = a
            action = best_a

        self._prev_enc = enc.copy(); self._prev_enc_bin = enc_bin.copy()
        self._prev_action = action
        return action

    @property
    def n_actions(self): return self._n_actions

    def reset(self, seed):
        self._rng = np.random.RandomState(seed)
        self._root = TreeNode()
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None; self._prev_enc_bin = None
        self._prev_action = None; self._last_enc = None

    def on_level_transition(self):
        self._prev_enc = None; self._prev_enc_bin = None; self._prev_action = None

    def get_state(self): return {"root": self._root, "rm": self._running_mean.copy(), "n": self._n_obs}
    def set_state(self, s): self._root = s["root"]; self._running_mean = s["rm"].copy(); self._n_obs = s["n"]
    def frozen_elements(self): return []


def make_game():
    try:
        import arcagi3; return arcagi3.make("LS20")
    except:
        import util_arcagi3; return util_arcagi3.make("LS20")


def run_phase(substrate, env_seed, n_steps):
    env = make_game(); obs = env.reset(seed=env_seed)
    step = 0; completions = 0; current_level = 0; pred_errors = []
    while step < n_steps:
        if obs is None:
            obs = env.reset(seed=env_seed); substrate.on_level_transition(); continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        prev_enc = substrate._prev_enc; prev_action = substrate._prev_action
        action = substrate.process(obs_arr) % N_ACTIONS
        obs_next, _, done, info = env.step(action); step += 1
        if prev_enc is not None and prev_action is not None and obs_next is not None:
            actual = substrate._last_enc
            prev_leaf = substrate._traverse(substrate._binarize(prev_enc))
            pred = prev_leaf.predict(prev_action)
            if pred is not None and actual is not None:
                err = float(np.sum((pred - actual)**2))
                norm = float(np.sum(actual**2)) + 1e-8
                pred_errors.append((err, norm))
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            completions += (cl - current_level); current_level = cl
            substrate.on_level_transition()
        if done:
            obs_next = env.reset(seed=env_seed); current_level = 0
            substrate.on_level_transition()
        obs = obs_next
    pred_acc = None
    if pred_errors:
        te = sum(e for e, n in pred_errors); tn = sum(n for e, n in pred_errors)
        pred_acc = float(1.0 - te / tn) * 100.0
    return completions, pred_acc


print("=" * 70)
print("STEP 897 — ONLINE DECISION TREE PREDICTOR")
print("=" * 70)
print(f"Split on pred_error>{SPLIT_THRESH} AND count>{MIN_SAMPLES}. Max depth={MAX_DEPTH}.")
print(f"Action: argmax(1/(leaf_count+1)). Tree depth = R3 self-modification.")
print(f"Kill criterion: max_depth < 4 at 10K steps.")

t0 = time.time()

# Pretrain
sub_p = OnlineDecisionTree897(n_actions=N_ACTIONS, seed=0)
sub_p.reset(0)
for ps in PRETRAIN_SEEDS:
    sub_p.on_level_transition()
    env = make_game(); obs = env.reset(seed=ps * 1000); s = 0
    while s < PRETRAIN_STEPS:
        if obs is None:
            obs = env.reset(seed=ps * 1000); sub_p.on_level_transition(); continue
        action = sub_p.process(np.asarray(obs, dtype=np.float32)) % N_ACTIONS
        obs, _, done, _ = env.step(action); s += 1
        if done:
            obs = env.reset(seed=ps * 1000); sub_p.on_level_transition()
saved = sub_p.get_state()
print(f"Pretrain done ({time.time()-t0:.1f}s). tree_depth={sub_p.max_depth()} n_leaves={sub_p.n_leaves()}")

cold_comps = []; cold_accs = []
warm_comps = []; warm_accs = []

for ts in TEST_SEEDS:
    sub_c = OnlineDecisionTree897(n_actions=N_ACTIONS, seed=0)
    sub_c.reset(0)
    c_comp, c_acc = run_phase(sub_c, ts * 1000, TEST_STEPS)
    cold_comps.append(c_comp); cold_accs.append(c_acc)

    sub_w = OnlineDecisionTree897(n_actions=N_ACTIONS, seed=0)
    sub_w.reset(0)
    sub_w.set_state({"root": saved["root"], "rm": saved["rm"].copy(), "n": saved["n"]})
    w_comp, w_acc = run_phase(sub_w, ts * 1000, TEST_STEPS)
    warm_comps.append(w_comp); warm_accs.append(w_acc)

mc = np.mean(cold_comps); mw = np.mean(warm_comps)
vc = [a for a in cold_accs if a is not None]; vw = [a for a in warm_accs if a is not None]
mc_acc = np.mean(vc) if vc else None; mw_acc = np.mean(vw) if vw else None

print(f"\nRESULTS (Decision Tree):")
print(f"  cold: L1={mc:.1f}/seed  pred_acc={mc_acc:.2f}%  tree_depth={sub_c.max_depth()}" if mc_acc else f"  cold: L1={mc:.1f}/seed  tree_depth={sub_c.max_depth()}")
print(f"  warm: L1={mw:.1f}/seed  pred_acc={mw_acc:.2f}%  tree_depth={sub_w.max_depth()}" if mw_acc else f"  warm: L1={mw:.1f}/seed  tree_depth={sub_w.max_depth()}")
if mc_acc and mw_acc:
    print(f"  R3_cf: {'PASS' if mw_acc > mc_acc else 'FAIL'} ({mw_acc-mc_acc:+.2f}%)")
max_cold_depth = sub_c.max_depth()
print(f"  Kill check: depth={max_cold_depth}. {'ALIVE' if max_cold_depth >= 4 else 'KILLED (depth<4)'}")
print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
print("STEP 897 DONE")
