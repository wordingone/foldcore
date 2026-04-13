"""
Step 1399 — Dolphin v4: KB action boost from observed frame changes.
Leo mail 4559, 2026-04-13.

Single variable over v3 (step1398):
  Track per-KB-action EMA frame-change rate.
  After each (state, action, next_state): record if frame hash changed.
  change_rate[action] = EMA(changed, alpha=0.1), init=0.5.
  On KB-only action selection: sort by change_rate descending (epsilon=0.1 random).
  Click games: v3 segment boost unchanged.
  KB-only games: v4 action ordering (new).

Targets: WA30, G50T, RE86 (KB-only, growing, v3-zero).

Constitutional audit:
  R1: change_rate from frame deltas, not external reward.
  R2: tracking is part of exploration computation, no separate optimizer.
  R6: remove KB boost → falls back to v3 equal KB priority. No gain → KILL.
"""

import numpy as np
import hashlib
import sys
import os

sys.path.insert(0, 'B:/M/the-search/experiments/steps')
sys.path.insert(0, 'B:/M/the-search/experiments/environments')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search')

SALIENT_COLORS = frozenset({6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
MEDIUM_MIN = 2
MEDIUM_MAX = 32
STATUS_BAR_COLOR = 16
INF_DISTANCE = 10_000_000

G_BOOST = -1
G0 = 0
G1 = 1
G2 = 2
G3 = 3
G4 = 4

N_KB = 7
DECAY_WINDOW = 10   # v3: steps before segment boost expires
ALPHA_KB = 0.1      # v4: EMA alpha for KB change rate
EPSILON_KB = 0.1    # v4: exploration rate for KB action selection
INIT_CHANGE_RATE = 0.5  # v4: neutral init (neither known good nor bad)


def _flood_fill_segments(frame_int):
    """BFS flood-fill segmentation on 64x64 int frame."""
    H, W = frame_int.shape
    visited = np.zeros((H, W), dtype=bool)
    segments = []

    for sy in range(H):
        for sx in range(W):
            if visited[sy, sx]:
                continue
            color = int(frame_int[sy, sx])
            pixels = []
            queue = [(sx, sy)]
            visited[sy, sx] = True
            while queue:
                x, y = queue.pop()
                pixels.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and not visited[ny, nx]:
                        if int(frame_int[ny, nx]) == color:
                            visited[ny, nx] = True
                            queue.append((nx, ny))

            xs = [p[0] for p in pixels]
            ys = [p[1] for p in pixels]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            area = len(pixels)
            segments.append({
                'pixels': pixels,
                'color': color,
                'area': area,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'is_rectangle': (area == w * h),
                'is_status_bar': False,
            })

    return segments


def _detect_status_bars(segments, frame_h=64, frame_w=64, edge_px=3, min_aspect=5.0):
    """Detect and mark status bar segments."""
    def touching_edges(seg):
        edges = set()
        if seg['y1'] <= edge_px:
            edges.add('top')
        if seg['y2'] >= frame_h - 1 - edge_px:
            edges.add('bottom')
        if seg['x1'] <= edge_px:
            edges.add('left')
        if seg['x2'] >= frame_w - 1 - edge_px:
            edges.add('right')
        return edges

    edge_segs = []
    for i, seg in enumerate(segments):
        e = touching_edges(seg)
        if e:
            w = seg['x2'] - seg['x1'] + 1
            h = seg['y2'] - seg['y1'] + 1
            aspect = max(w, h) / max(min(w, h), 1)
            edge_segs.append((i, seg, e, aspect))

    for i, seg, edges, aspect in edge_segs:
        if aspect >= min_aspect:
            seg['is_status_bar'] = True

    from collections import defaultdict
    groups = defaultdict(list)
    for i, seg, edges, aspect in edge_segs:
        if seg['is_status_bar']:
            continue
        for edge in edges:
            key = (edge, seg['color'], seg['area'], seg['is_rectangle'])
            groups[key].append(i)

    for key, idxs in groups.items():
        if len(idxs) >= 3:
            for i in idxs:
                segments[i]['is_status_bar'] = True

    return segments


def _build_masked_frame(frame_int, segments):
    masked = frame_int.copy()
    for seg in segments:
        if seg['is_status_bar']:
            for x, y in seg['pixels']:
                masked[y, x] = STATUS_BAR_COLOR
    return masked


def _hash_frame(masked_frame):
    H, W = masked_frame.shape
    flat = masked_frame.flatten().astype(np.uint8)
    n = len(flat)
    if n % 2:
        flat = np.append(flat, 0)
    packed = flat[0::2] << 4 | flat[1::2]
    tag = f"{H}x{W}".encode()
    return hashlib.blake2b(packed.tobytes(), digest_size=16, person=tag[:16]).hexdigest()


def _seg_key(seg):
    """Canonical key for segment identity comparison across frames."""
    return (seg['color'], seg['x1'], seg['y1'], seg['x2'], seg['y2'], seg['area'])


def _static_priority_group(seg):
    """Static G0-G4 priority (identical to v1/v3)."""
    if seg['is_status_bar']:
        return G4
    color = seg['color']
    w = seg['x2'] - seg['x1'] + 1
    h = seg['y2'] - seg['y1'] + 1
    is_salient = color in SALIENT_COLORS
    is_medium = (MEDIUM_MIN <= w <= MEDIUM_MAX) and (MEDIUM_MIN <= h <= MEDIUM_MAX)
    if is_salient and is_medium:
        return G0
    if is_medium and not is_salient:
        return G1
    if is_salient and not is_medium:
        return G2
    return G3


def _compute_distances(nodes, edges, rev_edges, active_group):
    """Backward BFS from frontier (nodes with untested in active_group)."""
    frontier = set()
    for h, node in nodes.items():
        for i, g in enumerate(node['groups']):
            if g == active_group and i not in node['tested']:
                frontier.add(h)
                break

    if not frontier:
        return {}

    distances = {h: 0 for h in frontier}
    queue = list(frontier)
    head = 0
    while head < len(queue):
        h = queue[head]
        head += 1
        d = distances[h]
        for src_hash, local_idx in rev_edges.get(h, []):
            if src_hash not in distances:
                distances[src_hash] = d + 1
                queue.append(src_hash)
    return distances


CONFIG = {
    'step': 1399,
    'mechanism': 'dolphin_v4_kb_action_boost',
    'description': 'v3 + per-KB-action EMA change rate ordering (alpha=0.1, eps=0.1)',
}


class DolphinV4:
    """Dolphin v4: KB action boost from observed frame changes.

    Same as v3 (segment boost), plus:
    - Track per-KB-action EMA change rate (alpha=0.1, init=0.5)
    - After each step where a KB action was taken: update change_rate[action]
    - On KB-only G0 selection: sort by change_rate desc (epsilon=0.1 random)
    - Click actions: unchanged (v3 segment boost)
    """

    def __init__(self):
        self._supports_click = False
        self._n_actions = N_KB
        self._pending_transition = False
        self._reset_graph()

    def _reset_graph(self):
        self._nodes = {}
        self._edges = {}
        self._rev_edges = {}
        self._active_group = G0
        self._current_hash = None
        self._last_local_action = None
        self._step = 0
        # v3: change tracking
        self._prev_seg_keys = set()
        self._boosted = {}              # seg_key → step when boosted
        # v4: KB action change rate
        self._kb_change_rate = np.full(N_KB, INIT_CHANGE_RATE, dtype=np.float32)

    def set_game(self, n_actions):
        self._supports_click = (n_actions > N_KB)
        self._n_actions = n_actions

    def on_level_transition(self):
        """v2 GAME_OVER persistence: defer reset to next process() call."""
        self._pending_transition = True
        self._current_hash = None
        self._last_local_action = None
        self._prev_seg_keys = set()
        self._boosted = {}
        # Do NOT reset _kb_change_rate on transition — accumulated KB statistics
        # are valid across GAME_OVER restarts (same game, same KB actions)

    def process(self, obs):
        """Select action with KB change-rate ordering + segment boost."""
        self._step += 1

        frame_int = self._extract_frame(obs)
        if frame_int is None:
            return np.random.randint(0, max(self._n_actions, 1))

        segments = _flood_fill_segments(frame_int)
        segments = _detect_status_bars(segments)
        masked_frame = _build_masked_frame(frame_int, segments)
        node_hash = _hash_frame(masked_frame)

        current_seg_keys = set(
            _seg_key(s) for s in segments if not s['is_status_bar']
        )

        # v4: update KB change rate before anything else
        if (self._current_hash is not None
                and self._last_local_action is not None):
            prev_node = self._nodes.get(self._current_hash)
            if prev_node is not None:
                lact = self._last_local_action
                if lact < len(prev_node['actions']):
                    enc = prev_node['actions'][lact]
                    if enc < N_KB:
                        changed = float(node_hash != self._current_hash)
                        self._kb_change_rate[enc] = (
                            (1.0 - ALPHA_KB) * self._kb_change_rate[enc]
                            + ALPHA_KB * changed
                        )

        # v3: update segment boosts
        if self._prev_seg_keys:
            changed_keys = current_seg_keys - self._prev_seg_keys
            for key in changed_keys:
                self._boosted[key] = self._step

        expired = [k for k, t in self._boosted.items() if self._step - t >= DECAY_WINDOW]
        for k in expired:
            del self._boosted[k]

        # v2: resolve pending transition
        if self._pending_transition:
            self._pending_transition = False
            if node_hash not in self._nodes:
                self._reset_graph()

        # Record transition
        if (self._current_hash is not None
                and self._last_local_action is not None
                and self._current_hash != node_hash):
            self._record_edge(self._current_hash, self._last_local_action, node_hash)

        # Add node if new
        if node_hash not in self._nodes:
            actions, groups, seg_keys = self._build_action_space(segments)
            self._nodes[node_hash] = {
                'actions': actions,
                'groups': groups,
                'seg_keys': seg_keys,
                'tested': set(),
            }
            self._rev_edges.setdefault(node_hash, [])

        self._current_hash = node_hash
        self._prev_seg_keys = current_seg_keys

        # Group advancement
        distances = self._recompute_distances()
        if node_hash not in distances and self._active_group < G4:
            self._active_group += 1
            distances = self._recompute_distances()

        # Select action
        local_action = self._select_action(node_hash, distances)
        self._last_local_action = local_action

        node = self._nodes[node_hash]
        if local_action is not None and local_action < len(node['actions']):
            global_action = node['actions'][local_action]
            node['tested'].add(local_action)
            return global_action

        return np.random.randint(0, N_KB)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_frame(self, obs):
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-2:] == (64, 64):
            return arr[0].astype(np.int32)
        elif arr.ndim == 2 and arr.shape == (64, 64):
            return arr.astype(np.int32)
        return None

    def _build_action_space(self, segments):
        actions = list(range(N_KB))
        groups = [G0] * N_KB
        seg_keys = [None] * N_KB

        if self._supports_click:
            for seg in segments:
                g = _static_priority_group(seg)
                pixels = seg['pixels']
                x, y = pixels[np.random.randint(len(pixels))]
                click_idx = N_KB + y * 64 + x
                actions.append(click_idx)
                groups.append(g)
                seg_keys.append(_seg_key(seg))

        return actions, groups, seg_keys

    def _record_edge(self, src_hash, local_idx, dst_hash):
        key = (src_hash, local_idx)
        if key not in self._edges:
            self._edges[key] = dst_hash
            self._rev_edges.setdefault(dst_hash, [])
            self._rev_edges[dst_hash].append((src_hash, local_idx))

    def _recompute_distances(self):
        return _compute_distances(
            self._nodes, self._edges, self._rev_edges, self._active_group
        )

    def _select_action(self, node_hash, distances):
        """v4 action selection: BOOSTED → G0(KB-ordered) → G1 → G2 → G3 → G4.

        For KB actions in active group: sort by change_rate descending (eps=0.1).
        For click actions in active group: random (v3 unchanged).
        Boosted = untested click action with recently-changed segment key (v3).
        """
        node = self._nodes[node_hash]
        actions = node['actions']
        groups = node['groups']
        seg_keys = node.get('seg_keys', [None] * len(actions))
        tested = node['tested']

        # BOOSTED: untested click actions with recently-changed segments (v3)
        boosted_untested = [
            i for i in range(len(actions))
            if i not in tested
            and seg_keys[i] is not None
            and seg_keys[i] in self._boosted
            and self._step - self._boosted[seg_keys[i]] < DECAY_WINDOW
        ]
        if boosted_untested:
            return np.random.choice(boosted_untested)

        # Active group: split KB vs click
        untested_in_group = [
            i for i, g in enumerate(groups)
            if g == self._active_group and i not in tested
        ]

        if untested_in_group:
            kb_candidates = [i for i in untested_in_group if actions[i] < N_KB]
            click_candidates = [i for i in untested_in_group if actions[i] >= N_KB]

            if click_candidates:
                # Click actions available: random among all (clicks are prioritized by segment group)
                return np.random.choice(untested_in_group)
            elif kb_candidates:
                # KB-only: v4 change-rate ordering
                if np.random.random() < EPSILON_KB:
                    return np.random.choice(kb_candidates)
                best = max(kb_candidates, key=lambda i: float(self._kb_change_rate[actions[i]]))
                return best

        # Navigate toward frontier via lowest-distance edge
        best_local = None
        best_dist = INF_DISTANCE
        for (src_hash, local_idx), dst_hash in self._edges.items():
            if src_hash != node_hash:
                continue
            d = distances.get(dst_hash, INF_DISTANCE)
            if d < best_dist:
                best_dist = d
                best_local = local_idx

        if best_local is not None:
            return best_local

        # Fallback: any untested action
        all_untested = [i for i in range(len(actions)) if i not in tested]
        if all_untested:
            return np.random.choice(all_untested)

        return np.random.randint(0, max(len(actions), 1))
