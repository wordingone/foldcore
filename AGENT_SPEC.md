# Agent Spec: The Search Condensed

**One agent. One codebase. 1000+ experiments compressed into a game-playing entity.**

For the ARC-AGI-3 Community Leaderboard. Not a substrate — a scored agent.

---

## 1. API Contract

```python
import arc_agi
from arcengine.enums import GameAction

arcade = arc_agi.Arcade()
env = arcade.make(game_id, seed=seed, scorecard_id=scorecard_id, include_frame_data=True)
obs = env.reset()  # → FrameDataRaw

# FrameDataRaw fields:
#   obs.frame           : list[np.ndarray]  — list of (64,64) int8 arrays. Usually len=1. Multi-frame games: len>1.
#   obs.game_id         : str               — e.g., "ls20-9607627b" (includes version hash)
#   obs.state           : GameState         — NOT_FINISHED, FINISHED, etc.
#   obs.levels_completed: int               — levels cleared so far
#   obs.win_levels      : int               — total levels to win the game
#   obs.available_actions: list[int]         — which actions this game supports (e.g., [1,2,3,4])
#   obs.full_reset      : bool              — True on first frame or full restart

# Step:
obs = env.step(GameAction.ACTION1)                          # keyboard
obs = env.step(GameAction.ACTION6, data={"x": 32, "y": 16}) # click at pixel (32, 16)
```

## 2. Frame Normalization

```python
def normalize_frame(obs) -> np.ndarray:
    """Extract single 64x64 frame from observation."""
    frames = obs.frame  # list of np.ndarray
    if len(frames) == 0:
        return np.zeros((64, 64), dtype=np.int8)
    frame = np.array(frames[-1])  # ALWAYS take last frame (multi-frame fix)
    assert frame.shape == (64, 64), f"Expected (64,64), got {frame.shape}"
    return frame
```

**Multi-frame games (4 known: bp35, lf52, sc25, sk48):** Return multiple frames in `obs.frame`. We always take the last one — it's the current visual state.

## 3. Architecture

```
Agent
├── normalize_frame(obs) → np.ndarray(64,64)
├── get_game_key(obs.game_id) → str       # "ls20" from "ls20-9607627b"
├── SOLVERS: dict[str, Solver]             # per-game analytical solvers
├── ExplorationModule                      # for unknown games
│   ├── state_hash(frame) → int
│   ├── state_graph: dict[(hash,action), hash]
│   ├── action_deltas: dict[action, float]
│   ├── visited_from: dict[hash, set[action]]
│   └── path: list[(hash, action)]         # for backtracking
└── act(obs) → (GameAction, dict|None)
```

## 4. Game Dispatch

```python
def get_game_key(game_id: str) -> str:
    """Extract base game name: 'ls20-9607627b' → 'ls20'"""
    return game_id.split('-')[0]

def act(self, obs):
    game_key = get_game_key(obs.game_id)

    if game_key in self.SOLVERS:
        return self.SOLVERS[game_key].act(obs)
    else:
        return self.exploration.act(obs)
```

**CRITICAL:** Dispatch is by game_key (first part of game_id). Version hashes may change across draws — solvers MUST read the current frame and compute actions dynamically. Never replay hardcoded action sequences.

## 5. Solver Interface

Every solver implements:

```python
class Solver:
    def reset(self):
        """Called on game start and level transitions (obs.full_reset or levels_completed changes)."""
        pass

    def act(self, obs: FrameDataRaw) -> tuple[GameAction, dict | None]:
        """
        Returns (action, data).
        KB action: (GameAction.ACTION1, None)
        Click action: (GameAction.ACTION6, {"x": int, "y": int})
        """
        pass
```

**Solvers must be FRAME-REACTIVE.** They read the current frame, extract game state, and compute the next optimal action. They NEVER store hardcoded action sequences. Game draws vary — tile positions change, maze layouts change, obstacle positions change.

### Solver implementation pattern:

```python
class LS20Solver(Solver):
    def act(self, obs):
        frame = normalize_frame(obs)
        # 1. Extract state: player position, goal position, walls
        player = find_player(frame)  # e.g., find color cluster
        goal = find_goal(frame)
        walls = find_walls(frame)
        # 2. BFS from player to goal
        path = bfs(player, goal, walls)
        # 3. Return next action along path
        if not path:
            return (GameAction.ACTION1, None)  # fallback
        next_pos = path[0]
        action = direction_to_action(player, next_pos)
        return (action, None)
```

### Tier 1 — Analytical solvers (EXIST, verified):

| Game | Mechanism | Actions L1 | Multi-level |
|------|-----------|-----------|-------------|
| FT09 | Click tile puzzle: read frame → identify tiles → compute click sequence | ~4 clicks | Yes, 6L |
| VC33 | Click canal lock: read frame → identify valves → click sequence | ~3 clicks | Yes, 5-7L |
| LS20 | KB maze: read frame → extract maze graph → BFS | ~13 moves | Yes, 7L |
| TU93 | KB maze: read frame → extract topology → BFS | ~18 moves | Yes, multi |
| SB26 | Click placement: read target config → click items into slots | ~9 clicks | Yes, multi |
| SU15 | Click physics: compute pull trajectory → click along path | ~8 clicks | Yes, multi |

### Tier 2 — Need analytical solvers (search-cracked, action counts known):

Each game below was solved by brute-force search. Eli has source code access. Build frame-reactive solvers.

| Game | Modality | L1 actions (search) | Load-bearing | Dead | Notes |
|------|----------|--------------------|--------------|----|-------|
| sp80 | KB-dom | 13 | A4, A5 | A1-3, A6-7, clicks | available_actions likely [4,5] or superset |
| cd82 | KB-dom | 24 | A2-A5 | A1, A6-7, clicks | |
| cn04 | KB-dom | 52 | A1-A4 | A5-7, clicks | |
| r11l | Click-dom | 55 | CLICK(30) | all KB | Pure click game |
| re86 | KB-only | 93 | A1-A5 | — | n_actions=7 (no clicks) |
| m0r0 | KB-dom | 138 | A1, A3, A4 | A2, A5-7, clicks | Mirrored twin blocks (Jun's play data) |
| tr87 | KB-dom | 180 | A1-A4 | A5-7 | n_actions=7 (no clicks) |
| ar25 | KB-dom | 205 | A1-A4, A7 | A5-6, clicks | |
| lp85 | Click-dom | 446 | CLICK(230) | all KB | Pure click game |
| sk48 | KB-dom | 447 | A1-A4, A7 | A5-6, clicks | |
| tn36 | Click-only | 55 | CLICK(55) | — | Only ACTION6 in diagnostic |
| ka59 | KB-dom | 54 | A1-A4 | A5-7 | Sokoban with push |

**FOR EACH TIER 2 GAME:** Eli reads the game source code, identifies mechanics, writes a frame-reactive solver that:
1. Extracts game state from pixel frame
2. Plans via BFS/search/heuristic
3. Returns next action

If a game's source code is too complex for a reliable analytical solver within the timeframe, fall through to the Exploration Module.

### Tier 3 — No solver (7 games, use Exploration Module):

| Game | Type | Estimated difficulty |
|------|------|---------------------|
| dc22 | Maze navigation | LOW — BFS-solvable by exploration |
| sc25 | Spell casting + nav | MEDIUM — needs click + KB |
| wa30 | Sokoban grab/carry | MEDIUM — planning puzzle |
| bp35 | Gravity platformer + descending ceiling | HIGH — timing-dependent |
| g50t | Ghost replay / recording mechanic | HIGH — needs space bar timing |
| lf52 | Peg solitaire (8 precise clicks) | HIGH — combinatorial |
| s5i5 | Arm rotation/extension | HIGH — unknown mechanics |

## 6. Exploration Module (Unknown Games)

For games without an analytical solver. Combines graph exploration (3rd place approach) with search insights.

### 6.1 State Hashing

```python
def state_hash(frame: np.ndarray) -> int:
    """Hash 64x64 frame to a single int for state identity."""
    # Downsample to 16x16 via mean pooling (same as avgpool4)
    pooled = frame.reshape(16, 4, 16, 4).mean(axis=(1, 3))
    # Quantize to 16 levels (4 bits per cell)
    quantized = np.clip((pooled * 16 / 256).astype(int), 0, 15)
    # Hash
    return hash(quantized.tobytes())
```

**Why this hash:** avgpool4 (16x16) was proven by encoding research (Steps 378-895) to preserve signal while removing noise. 16-level quantization handles minor pixel variations across draws.

### 6.2 Action Space Selection

```python
def get_action_set(obs) -> list:
    """Determine which actions to try."""
    available = obs.available_actions  # e.g., [1,2,3,4] or [1,2,3,4,5,6,7]

    actions = []
    for a in available:
        if a != 6:
            actions.append(("kb", a, None))  # (type, action_id, data)
        else:
            # ACTION6 = click. Add grid of click positions.
            # Start with 4x4 coarse grid (16 positions)
            for gx in range(4):
                for gy in range(4):
                    x = gx * 16 + 8  # centers: 8, 24, 40, 56
                    y = gy * 16 + 8
                    actions.append(("click", 6, {"x": x, "y": y}))

    return actions
```

**Why 4x4 grid:** Full 64x64 = 4096 positions is too many. 4x4 coarse grid (16 positions) covers the space. When a click at (x,y) causes frame change, refine: add (x±8, y), (x, y±8) to explore finer positions.

### 6.3 Core Loop

```python
class ExplorationModule:
    def __init__(self):
        self.state_graph = {}      # (hash, action_key) → next_hash
        self.action_deltas = {}    # action_key → EMA of frame delta
        self.visited = {}          # hash → set of tried action_keys
        self.score_states = {}     # hash → best_score_seen_here
        self.prev_frame = None
        self.prev_hash = None
        self.prev_action = None
        self.current_level = 0
        self.step_count = 0
        self.action_set = None
        self.action_budget = 1000  # per-level budget

    def act(self, obs):
        frame = normalize_frame(obs)
        h = state_hash(frame)

        # Detect level transition
        if obs.levels_completed > self.current_level:
            self.on_level_up()
            self.current_level = obs.levels_completed

        # Initialize action set on first call
        if self.action_set is None:
            self.action_set = get_action_set(obs)

        # Record previous transition
        if self.prev_hash is not None and self.prev_action is not None:
            delta = np.sum(np.abs(frame.astype(float) - self.prev_frame.astype(float)))
            self.state_graph[(self.prev_hash, self.prev_action)] = h
            # Update delta tracker
            key = self.prev_action
            old = self.action_deltas.get(key, delta)
            self.action_deltas[key] = 0.9 * old + 0.1 * delta  # EMA

        # Mark hash as visited for this action
        if h not in self.visited:
            self.visited[h] = set()

        # Find untested actions from current state
        untested = [a for a in self.action_set if self._action_key(a) not in self.visited[h]]

        if untested:
            # Sort by expected delta (prefer high-delta actions)
            untested.sort(key=lambda a: self.action_deltas.get(self._action_key(a), 1.0), reverse=True)
            chosen = untested[0]
        else:
            # All actions tested from this state. Pick highest-delta action to move to new state.
            all_actions = list(self.action_set)
            all_actions.sort(key=lambda a: self.action_deltas.get(self._action_key(a), 0), reverse=True)
            chosen = all_actions[0]  # Best known action to make progress

        # Record and execute
        action_key = self._action_key(chosen)
        self.visited[h].add(action_key)
        self.prev_frame = frame.copy()
        self.prev_hash = h
        self.prev_action = action_key
        self.step_count += 1

        # Convert to API format
        action_type, action_id, data = chosen
        game_action = GameAction(action_id)
        return (game_action, data)

    def _action_key(self, action_tuple):
        """Hashable key for an action."""
        atype, aid, data = action_tuple
        if data:
            return (aid, data.get("x", 0), data.get("y", 0))
        return (aid,)

    def on_level_up(self):
        """Reset state graph but keep action delta knowledge."""
        self.state_graph.clear()
        self.visited.clear()
        self.score_states.clear()
        self.prev_frame = None
        self.prev_hash = None
        self.prev_action = None
        self.step_count = 0
        # action_deltas PRESERVED — priors transfer across levels
```

### 6.4 Click Refinement

When a coarse-grid click at (x, y) produces delta > threshold:

```python
def refine_click(self, x, y):
    """Add fine-grained clicks around a responsive location."""
    for dx in [-8, -4, 0, 4, 8]:
        for dy in [-8, -4, 0, 4, 8]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 64 and 0 <= ny < 64 and (nx, ny) != (x, y):
                new_action = ("click", 6, {"x": nx, "y": ny})
                if new_action not in self.action_set:
                    self.action_set.append(new_action)
```

**Trigger:** After any click where `delta > 100` (pixel sum change), call `refine_click(x, y)`.

## 7. Main Agent Class

```python
class SearchAgent:
    """The Search, condensed into one agent."""

    def __init__(self):
        self.SOLVERS = {
            "ft09": FT09Solver(),
            "vc33": VC33Solver(),
            "ls20": LS20Solver(),
            "tu93": TU93Solver(),
            "sb26": SB26Solver(),
            "su15": SU15Solver(),
            # Tier 2: add as Eli builds them
            "sp80": SP80Solver(),
            "cd82": CD82Solver(),
            # ... etc
        }
        self.exploration = ExplorationModule()
        self.current_game = None
        self.solver = None

    def act(self, obs):
        game_key = obs.game_id.split('-')[0]

        # On game change, reset
        if game_key != self.current_game:
            self.current_game = game_key
            self.exploration = ExplorationModule()
            self.solver = self.SOLVERS.get(game_key, None)

        # Dispatch
        if self.solver:
            return self.solver.act(obs)
        else:
            return self.exploration.act(obs)
```

## 8. Testing Protocol

```python
def test_all_games():
    """Run agent on all 25 games. Compare to predictions."""
    arcade = arc_agi.Arcade()
    scorecard_id = arcade.open_scorecard(source_url="https://github.com/wordingone/the-search")

    results = {}
    for env_info in arcade.get_environments():
        game_key = env_info.game_id.split('-')[0]
        env = arcade.make(env_info.game_id, seed=0, scorecard_id=scorecard_id)

        agent = SearchAgent()
        obs = env.reset()
        total_actions = 0
        max_level = 0

        while obs.state.name == "NOT_FINISHED":
            action, data = agent.act(obs)
            obs = env.step(action, data=data)
            total_actions += 1
            max_level = max(max_level, obs.levels_completed)

            if total_actions > 5000:  # safety cap
                break

        results[game_key] = {
            "levels": max_level,
            "win_levels": obs.win_levels,
            "actions": total_actions,
        }
        print(f"{game_key}: {max_level}/{obs.win_levels}L in {total_actions} actions")

    arcade.close_scorecard(scorecard_id)
    return results
```

## 9. Predictions

### Tier 1 — Analytical Solvers (HIGH confidence, ≥95%)

| Game | Levels | Actions | RHAE estimate | Notes |
|------|--------|---------|---------------|-------|
| FT09 | 6/6 | ~75 | 0.85-1.0 | Click puzzle. Human baseline: 163 total |
| VC33 | 5/7 | ~178 | 0.70-0.85 | Click canal. Human: 307. L6-7 may fail (complex) |
| LS20 | 7/7 | ~311 | 0.90-1.0 | KB maze. Human: 546. BFS optimal |
| TU93 | ≥3/9 | ~50-100 | 0.50-0.80 | KB maze. Human: 378. Later levels complex |
| SB26 | ≥3/8 | ~30-50 | 0.60-0.90 | Click placement. Human: 153. Later levels may fail |
| SU15 | ≥3/9 | ~30-50 | 0.60-0.90 | Click physics. Human: 566. Later levels complex |

### Tier 2 — Frame-Reactive Solvers (MEDIUM confidence, 60-80%)

Confidence depends on whether Eli successfully builds frame-reactive solvers from source code analysis. If source code is too complex → falls to exploration.

| Game | L1 prediction | Actions L1 | L2+ possible | Risk |
|------|--------------|-----------|-------------|------|
| sp80 | YES | ~13 | YES (54 L2) | Low — simple KB game |
| cd82 | YES | ~24 | YES (78 L2) | Low — KB navigation |
| cn04 | YES | ~52 | Maybe | Medium — more actions |
| r11l | YES | ~55 clicks | Maybe | Medium — click precision |
| re86 | YES | ~93 | Maybe | Medium — 5 KB actions, no clicks |
| m0r0 | YES | ~138 | Unlikely | HIGH — mirrored twin blocks, complex |
| tr87 | YES | ~180 | Maybe | Medium — KB navigation |
| ar25 | YES | ~205 | Maybe | Medium — KB + A7 needed |
| lp85 | YES | ~446 clicks | Unlikely | HIGH — 230 clicks, precision needed |
| sk48 | YES | ~447 | Unlikely | HIGH — many actions |
| tn36 | YES | ~55 clicks | Maybe | Medium — click-only |
| ka59 | YES | ~54 | Maybe | Medium — Sokoban push |

### Tier 3 — Exploration Module (LOW confidence, 10-50%)

| Game | L1 prediction | Confidence | Reasoning |
|------|--------------|-----------|-----------|
| dc22 | LIKELY | 50% | Maze navigation. A1-A4 all responsive (delta=21). Graph exploration handles mazes. Budget: 1192 human baseline = room for 3-5x more. |
| sc25 | POSSIBLE | 30% | A1-A5 all responsive (delta=2500+). High delta = visible changes. But spell casting may need specific sequences. |
| wa30 | POSSIBLE | 30% | A1-A4 responsive (delta=130-320). Sokoban = planning puzzle. Graph exploration may stumble into solution. Human: 1564 actions = very complex. |
| bp35 | UNLIKELY | 15% | Gravity platformer. Timing-dependent. Diagnostic had NO data (pre multi-frame fix). |
| g50t | UNLIKELY | 10% | Ghost replay requires recording mechanic (space bar = A5? or A7?). Graph exploration can't plan what to RECORD. |
| lf52 | UNLIKELY | 20% | Peg solitaire. 8 precise clicks from 4096 positions. Graph exploration with click refinement MIGHT find responsive positions. |
| s5i5 | UNLIKELY | 15% | Arm rotation. Only ACTION6 responsive (delta=1.2, very low). Extremely subtle changes. |

### Aggregate Prediction

**Best case** (all Tier 2 solvers built): 18 games L1, 3-5 Tier 3 games L1 = **21-23/25 L1**
**Realistic case** (12/12 Tier 2 + 2/7 Tier 3): **20/25 L1**
**Conservative case** (9/12 Tier 2 + 1/7 Tier 3): **16/25 L1**

**RHAE score estimate** (realistic case):
- 6 Tier 1 games × ~0.80 avg = 4.8
- 12 Tier 2 games × ~0.30 avg (L1 only, suboptimal) = 3.6
- 2 Tier 3 games × ~0.05 avg = 0.1
- 5 failed games × 0 = 0
- Total: 8.5 / 25 = **0.34 average RHAE**

With multi-level on Tier 1 games boosting the average, realistic RHAE ≈ **0.35-0.45**.

## 10. Failure Modes

| # | Mode | Impact | Mitigation |
|---|------|--------|------------|
| 1 | Game version changes → solver breaks | Solver produces wrong actions | All solvers MUST read frame, never replay sequences. Test across multiple seeds. |
| 2 | Multi-frame games (bp35, lf52, sc25, sk48) | Wrong frame extracted | normalize_frame() always takes frames[-1]. Verified working. |
| 3 | Click coordinate format | API error or wrong position | Always int(x), int(y). Always 0 ≤ x,y < 64. |
| 4 | Hash collision in exploration | Agent thinks it's in a visited state when it's not | 16x16 × 16-level quantization = 2^256 theoretical space. Collision unlikely but possible. Mitigate: on unexpected result, rehash at higher resolution. |
| 5 | Solver infinite loop | Agent wastes all budget on one state | Every solver has a per-step counter. If >2x expected actions for current level, return random action. |
| 6 | Game has no solution from current state | Agent stuck | Exploration module: after 500 actions with no level progress, switch to uniform random. |
| 7 | available_actions changes mid-game | New actions unlock at higher levels | Re-check obs.available_actions every step. Update action_set if changed. |
| 8 | Level death/reset | Frame returns to start but levels_completed unchanged | Detect via state_hash matching a known "start" hash for this level. State graph naturally handles revisitation. |
| 9 | Scorecard timeout | Too many actions burns the clock | Per-game budget: min(5000, 10 × human_baseline_total). Abandon game if exceeded. |

## 11. File Structure

```
the-search/
├── agent/
│   ├── __init__.py
│   ├── agent.py          # SearchAgent class (Section 7)
│   ├── exploration.py    # ExplorationModule (Section 6)
│   ├── frame.py          # normalize_frame, state_hash (Sections 2, 6.1)
│   └── solvers/
│       ├── __init__.py
│       ├── base.py       # Solver interface
│       ├── ft09.py       # Tier 1
│       ├── vc33.py       # Tier 1
│       ├── ls20.py       # Tier 1
│       ├── tu93.py       # Tier 1
│       ├── sb26.py       # Tier 1
│       ├── su15.py       # Tier 1
│       ├── sp80.py       # Tier 2
│       ├── cd82.py       # Tier 2
│       └── ...           # remaining Tier 2
├── test_agent.py         # Testing protocol (Section 8)
└── AGENT_SPEC.md         # This file
```

## 12. What This Channels From The Search

| Research finding | How it's used |
|-----------------|---------------|
| Encoding: avgpool4 > avgpool16 (Steps 378-895) | State hash uses 16×16 pooling |
| Frame diff amplifies signal (Steps 942+) | Delta tracking in exploration module |
| Dead actions: A5-7 dead in KB games (ablation 1019-1250) | Solver action pruning |
| Modality: KB-dom / click-dom / click-only (25-game classification) | Solver design per game type |
| Simplicity load-bearing: 0-param reactive > complex (Prop 32, debate) | Exploration module is simple reactive, no learning |
| Prescriptions = minimum action sequences (D2 catalog) | Analytical solver targets |
| Graph exploration works training-free (3rd place competitor) | Exploration module architecture |
| StochasticGoose spatial head for clicks (1st place) | Click refinement around responsive regions |
| Human play data: M0R0 mirrored blocks, G50T ghost replay, VC33 canals | Solver mechanics for these games |
| Theorem 4: global running mean SNR→0 | Exploration uses per-state tracking, not global |
| 0% wall = zero pixel change on ~40% of draws (PB32) | Exploration module doesn't waste budget on unresponsive actions |
| available_actions from API (discovered during spec) | No modality probe needed — API tells us which actions exist |
| Human baselines from API (extracted per game per level) | RHAE scoring denominators |
| Attention-over-trajectory breaks Theorem 4 (Family 8) | NOT used — too complex for score-maximizing agent. Insight retained for substrate research |
| Alpha self-modification (Steps 895+) | NOT used — same reason. This is a scoring agent, not a substrate |

**What's deliberately EXCLUDED:** R3 self-modification, alpha attention, attention-over-trajectory, codebook learning, Hebbian updates, oscillatory dynamics, forward models, population-based methods. These are substrate research — valuable for understanding, but they REDUCE score compared to analytical solvers. The search proved this empirically: 1000+ experiments of substrate < 1 analytical solver.

---

*The search found that intelligence (R3) is the open problem. This agent doesn't solve R3. It solves the games using everything the search learned ABOUT the games. Honest.*
