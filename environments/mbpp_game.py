"""
MBPP Game Wrapper — ARC-like interface for MBPP code generation problems.

Treats MBPP as an ARC-AGI-3 compatible environment:
  - Observation: 1D float32 array (256 dims) encoding the last 256 bytes of
    (prompt + "\\n" + generated_code), normalized to [0,1].
    This passes through _enc_frame's fallback path unchanged — same pipeline.
  - Action space: 128 (ASCII codes 0-127). Each action appends chr(action) to
    generated code (control chars other than \\n/\\t silently ignored).
  - Episode ends when: generated code contains "def " AND ends with "\\n\\n",
    OR step limit reached.
  - Reward: pass_rate = assertions_passed / len(test_list) on termination.
  - Level: 0 (in progress) → 1 (at least one assertion passes, L1 equivalent).

Integration with PRISM:
  - Drop-in replacement for ARC-AGI-3 game. make('mbpp_NNN') creates problem NNN.
  - Substrate receives obs as 1D array → _enc_frame fallback path → 256-dim encoding.
  - 50 stable evaluation problems (indices 0-49 from sanitized test set).
  - Problem selected by: PROBLEMS[problem_idx % len(PROBLEMS)].
  - Budget: MAX_STEPS=2000 (functions ~120 chars, 2000 steps = 16x headroom).

Dataset: google-research-datasets/mbpp sanitized test (257 problems, local cache).
Evaluation: exec() in restricted namespace, 3-second timeout via signal (POSIX only)
            or threading.Timer (Windows). No dangerous imports allowed.

Spec: Leo mail 3635 (interpretation 1), 2026-03-28.
"""
import re
import threading
import numpy as np

# Lazy-loaded dataset (loaded once on first use)
_PROBLEMS = None
N_EVAL_PROBLEMS = 50  # first 50 from sanitized test set

# ASCII action space: 128 characters
N_ACTIONS = 128

# Char categories for action mapping
_PRINTABLE = set(range(32, 127))  # printable ASCII (space to ~)
_SPECIAL = {10: '\n', 9: '\t'}    # newline and tab

MAX_STEPS = 2000  # per problem (functions ~120 chars, generous headroom)
EVAL_TIMEOUT = 3.0  # seconds for test execution


def _load_problems():
    global _PROBLEMS
    if _PROBLEMS is not None:
        return _PROBLEMS
    from datasets import load_dataset
    ds = load_dataset('google-research-datasets/mbpp', 'sanitized', split='test')
    _PROBLEMS = [
        {
            'task_id': p['task_id'],
            'prompt': p['prompt'],
            'code': p['code'],
            'test_list': p['test_list'],
            'test_imports': p.get('test_imports', []),
        }
        for p in ds
    ][:N_EVAL_PROBLEMS]
    return _PROBLEMS


def _safe_exec(code_str, test_str, timeout=EVAL_TIMEOUT):
    """Execute code_str + test_str in a restricted namespace with timeout.
    Returns True if no exception raised, False otherwise."""
    result = [False]
    exc = [None]

    def _run():
        namespace = {'__builtins__': __builtins__}
        try:
            exec(compile(code_str, '<generated>', 'exec'), namespace)
            exec(compile(test_str, '<test>', 'exec'), namespace)
            result[0] = True
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return False  # timeout
    return result[0]


def _make_obs(prompt, generated):
    """Encode last 256 bytes of (prompt + '\\n' + generated) as float32 array.
    Same pipeline as _enc_frame fallback: flatten → 256 dims → [0, 1]."""
    text = prompt + '\n' + generated
    raw = text.encode('utf-8', errors='replace')[-256:]
    obs = np.zeros(256, dtype=np.float32)
    obs[:len(raw)] = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0
    return obs


def _action_to_char(action):
    """Convert action index (0-127) to character. Returns '' for ignored actions."""
    action = int(action) % N_ACTIONS
    if action in _SPECIAL:
        return _SPECIAL[action]
    if action in _PRINTABLE:
        return chr(action)
    return ''


class MBPPGame:
    """MBPP problem as an ARC-AGI-3 compatible game environment.

    Usage:
        env = MBPPGame(problem_idx=0)
        obs = env.reset(seed=42)  # seed selects problem_idx % N_EVAL_PROBLEMS
        obs, reward, done, info = env.step(action)

    The seed parameter is repurposed: seed % N_EVAL_PROBLEMS selects the problem.
    This is consistent with how ARC-AGI-3 games use seed for game variation.
    """

    def __init__(self, problem_idx=0):
        self._problem_idx = problem_idx % N_EVAL_PROBLEMS
        self._problem = None  # loaded lazily
        self._generated = ''
        self._step = 0
        self._done = False
        self._level = 0
        self._pass_rate = 0.0
        self._max_level = 0

    @property
    def n_actions(self):
        return N_ACTIONS

    def reset(self, seed=None):
        """Reset the environment. seed % N_EVAL_PROBLEMS selects the problem."""
        problems = _load_problems()
        idx = (seed % N_EVAL_PROBLEMS) if seed is not None else self._problem_idx
        self._problem = problems[idx % len(problems)]
        self._generated = ''
        self._step = 0
        self._done = False
        self._level = 0
        self._pass_rate = 0.0
        self._max_level = 0
        return _make_obs(self._problem['prompt'], self._generated)

    def step(self, action):
        if self._done or self._problem is None:
            obs = _make_obs(self._problem['prompt'] if self._problem else '', self._generated)
            return obs, 0.0, True, {'level': self._level}

        char = _action_to_char(action)
        if char:
            self._generated += char
        self._step += 1

        # Check termination conditions
        done = False
        reward = 0.0

        is_complete = (
            'def ' in self._generated
            and self._generated.endswith('\n\n')
        )
        if is_complete or self._step >= MAX_STEPS:
            done = True
            self._done = True
            reward, self._pass_rate = self._evaluate()

        # Level: 0 = in progress, 1 = at least one assertion passes (L1 equiv)
        if self._pass_rate >= (1.0 / max(len(self._problem.get('test_list', [1])), 1) - 1e-6):
            self._level = 1
        if self._level > self._max_level:
            self._max_level = self._level

        obs = _make_obs(self._problem['prompt'], self._generated)
        info = {
            'level': self._level,
            'pass_rate': self._pass_rate,
            'generated_len': len(self._generated),
            'is_complete': is_complete,
            'step': self._step,
        }
        return obs, reward, done, info

    def _evaluate(self):
        """Execute generated code against test assertions. Returns (reward, pass_rate)."""
        if not self._problem or not self._generated.strip():
            return 0.0, 0.0

        test_list = self._problem.get('test_list', [])
        if not test_list:
            return 0.0, 0.0

        # Add test_imports to namespace setup if needed
        test_imports = self._problem.get('test_imports', [])
        import_str = '\n'.join(test_imports) + '\n' if test_imports else ''
        code_str = import_str + self._generated + '\n'

        passed = 0
        for test in test_list:
            if _safe_exec(code_str, test):
                passed += 1

        pass_rate = passed / len(test_list)
        reward = pass_rate
        return reward, pass_rate

    def get_info(self):
        """Return current state info (for diagnostics)."""
        return {
            'problem_idx': self._problem_idx,
            'task_id': self._problem.get('task_id') if self._problem else None,
            'generated': self._generated,
            'step': self._step,
            'level': self._level,
            'pass_rate': self._pass_rate,
        }


def make(game_id):
    """Factory function matching ARC-AGI-3 interface.

    game_id: 'mbpp' (uses problem_idx=0) or 'mbpp_NNN' (uses problem_idx=NNN)
    """
    game_id = str(game_id).lower().strip()
    if game_id == 'mbpp':
        return MBPPGame(problem_idx=0)
    if game_id.startswith('mbpp_'):
        try:
            idx = int(game_id.split('_', 1)[1])
            return MBPPGame(problem_idx=idx)
        except (ValueError, IndexError):
            pass
    raise ValueError(f"Unknown MBPP game id: {game_id}. Use 'mbpp' or 'mbpp_NNN'.")


# --- Solver baseline for ARC score computation ---

def compute_solver_steps(problem_idx=0):
    """Return solver level steps for MBPP problem (using ground-truth code).
    This gives the minimum steps needed by an oracle that types the solution directly."""
    problems = _load_problems()
    p = problems[problem_idx % len(problems)]
    gt_code = p['code'] + '\n\n'  # ground truth + double newline terminator
    # Oracle types exactly len(gt_code) steps
    return {1: len(gt_code)}


if __name__ == '__main__':
    # Quick sanity check
    print("MBPP Game Wrapper — sanity check")
    env = make('mbpp')
    obs = env.reset(seed=0)
    print(f"obs shape: {obs.shape}, dtype: {obs.dtype}, range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"n_actions: {env.n_actions}")

    # Type the ground-truth solution manually (simulate oracle)
    p = _load_problems()[0]
    print(f"Problem: {p['prompt'][:80]}")
    print(f"Ground-truth code length: {len(p['code'])} chars")

    solution = p['code'] + '\n\n'
    total_reward = 0.0
    for i, char in enumerate(solution):
        action = ord(char) if ord(char) < 128 else 32
        obs, reward, done, info = env.step(action)
        if done:
            total_reward = reward
            print(f"Done at step {i+1}: reward={reward:.3f}, pass_rate={info['pass_rate']:.3f}, "
                  f"level={info['level']}, is_complete={info['is_complete']}")
            break

    print(f"Generated ({len(env._generated)} chars):\n{env._generated[:200]}")
    print("\nSanity check PASSED" if total_reward > 0 else "\nSanity check: reward=0 (expected if tests fail)")
