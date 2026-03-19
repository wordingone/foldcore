#!/usr/bin/env python3
"""
Paper Compiler — Generates paper/paper.html from the unified knowledge/ system.

ALL content is derived dynamically from:
  - knowledge/entries/*.json (all 78 entries: SS sessions 1-23 + FoldCore steps 97-105)
  - knowledge/constraints.json (66 merged constraints: SS c001-c051 + FoldCore fc001-fc015)
  - knowledge/frozen_frame.json (frozen/adaptive element tracking)
  - CONSTITUTION.md (the five principles and eight stages)

No values are hardcoded. Running this script regenerates the paper
to reflect the current state of the unified knowledge base.

Usage:
    python paper/compile_paper.py

Output:
    paper/paper.html — self-contained, viewable in any browser
"""

import json
import html as html_mod
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
ENTRIES_DIR = KNOWLEDGE_DIR / "entries"
PAPER_DIR = PROJECT_ROOT / "paper"
OUTPUT_FILE = PAPER_DIR / "paper.html"


# ============================================================
# DATA LOADING
# ============================================================

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def load_all_entries():
    entries = []
    if ENTRIES_DIR.exists():
        for f in sorted(ENTRIES_DIR.glob("*.json")):
            entries.append(load_json(f))
    return entries


# ============================================================
# DYNAMIC VALUE COMPUTATION
# ============================================================

def parse_state(state_md):
    """Extract current stage, session, and status from state.md."""
    info = {'session': 0, 'stage_str': 'unknown', 'stage_num': 0, 'status': ''}
    for line in state_md.split('\n'):
        if line.startswith('**Session:**'):
            try:
                info['session'] = int(line.split('**Session:**')[1].strip())
            except ValueError:
                pass
        elif line.startswith('**Stage:**'):
            info['stage_str'] = line.split('**Stage:**')[1].strip()
            m = re.search(r'(\d+)', info['stage_str'])
            if m:
                info['stage_num'] = int(m.group(1))
        elif line.startswith('**Status:**'):
            info['status'] = line.split('**Status:**')[1].strip()
    return info


def compute_progress(frozen_frame):
    """Compute frozen frame reduction from frozen_frame.json."""
    total = len(frozen_frame)
    adaptive = sum(1 for e in frozen_frame if e['status'].startswith('adaptive'))
    calibrated = sum(1 for e in frozen_frame if e['status'] == 'calibrated')
    frozen = total - adaptive - calibrated
    pct = round((adaptive + calibrated) / total * 100) if total > 0 else 0
    return {
        'total_elements': total,
        'adaptive': adaptive,
        'calibrated': calibrated,
        'frozen': frozen,
        'pct_reduced': pct,
    }


def count_by_type(entries, entry_type):
    return sum(1 for e in entries if e.get('type') == entry_type)


def max_session(entries):
    sessions = [e.get('session', 0) for e in entries if e.get('session')]
    return max(sessions) if sessions else 0


# ============================================================
# CONSTITUTION PARSING (from CLAUDE.md)
# ============================================================

def extract_constitution(claude_md):
    """Extract constitution sections from CLAUDE.md.
    Returns ordered list of (name, content) tuples."""
    sections = []
    current_name = None
    current_lines = []

    for line in claude_md.split('\n'):
        if line.startswith('## '):
            if current_name is not None:
                sections.append((current_name, '\n'.join(current_lines)))
            current_name = line[3:].strip()
            current_lines = []
        elif current_name is not None:
            current_lines.append(line)

    if current_name is not None:
        sections.append((current_name, '\n'.join(current_lines)))

    # Keep only the constitution sections (stop at operational/meta sections)
    keep = [
        'What This Is', 'Definitions', 'The Five Principles',
        'The Stages', 'Operational Discipline', 'The One Guarantee'
    ]
    return [(name, content) for name, content in sections if name in keep]


def md_to_html(md_text):
    """Convert Markdown to HTML."""
    if HAS_MARKDOWN:
        return markdown.markdown(
            md_text,
            extensions=['tables', 'fenced_code', 'nl2br']
        )
    # Fallback: basic conversion
    text = html_mod.escape(md_text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'^### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = text.replace('\n\n', '</p><p>')
    return f'<p>{text}</p>'


# ============================================================
# CONTENT RENDERERS
# ============================================================

def esc(text):
    """Escape HTML."""
    return html_mod.escape(str(text))


def status_badge(status):
    """Colored badge for entry status."""
    colors = {
        'active': '#2196F3', 'succeeded': '#4CAF50', 'failed': '#f44336',
        'partial': '#FF9800', 'superseded': '#9E9E9E', 'retracted': '#9E9E9E',
        'frozen': '#f44336', 'adaptive': '#4CAF50',
        'adaptive (vacuous)': '#FF9800', 'calibrated': '#7B1FA2',
    }
    color = colors.get(status, '#757575')
    return (f'<span style="background:{color};color:white;padding:2px 8px;'
            f'border-radius:3px;font-size:0.8em;font-weight:bold;">'
            f'{esc(status.upper())}</span>')


def render_value(value):
    """Render a content value as HTML (handles str, list, dict)."""
    if isinstance(value, list):
        items = ''.join(f'<li>{esc(v)}</li>' for v in value)
        return f'<ul>{items}</ul>'
    elif isinstance(value, dict):
        items = ''.join(
            f'<li><strong>{esc(k)}:</strong> {render_value(v)}</li>'
            for k, v in value.items()
        )
        return f'<ul>{items}</ul>'
    else:
        return esc(value)


def render_entry_content(content):
    """Render an entry's content dict as HTML, priority keys first."""
    priority = [
        'description', 'hypothesis', 'finding', 'choice',
        'method', 'result', 'evidence', 'rationale', 'root_cause',
        'key_insight', 'implications',
    ]
    parts = []
    rendered = set()
    for key in priority:
        if key in content:
            label = key.replace('_', ' ').title()
            parts.append(f'<p><strong>{label}:</strong> {render_value(content[key])}</p>')
            rendered.add(key)
    for key, value in content.items():
        if key not in rendered:
            label = key.replace('_', ' ').title()
            parts.append(f'<p><strong>{label}:</strong> {render_value(value)}</p>')
    return '\n'.join(parts)


# ============================================================
# SECTION GENERATORS
# ============================================================

def generate_abstract(state_info, progress, constraints, entries):
    stage = state_info['stage_num']
    n_sessions = max_session(entries)
    n_constraints = len(constraints)
    n_experiments = count_by_type(entries, 'experiment')
    n_discoveries = count_by_type(entries, 'discovery')
    n_frozen = progress['frozen']
    n_adaptive = progress['adaptive']
    total = progress['total_elements']
    pct = progress['pct_reduced']

    n_calibrated = progress.get('calibrated', 0)
    resolved = n_adaptive + n_calibrated

    return f"""<div class="abstract">
<h2>Abstract</h2>
<p>This paper documents an ongoing research program in recursive self-improvement
through monotonic frozen frame reduction. We present a constitutional framework of
five principles and eight stages that defines the path to recursive self-improvement
architecture-independently, with empirical tests at each step.</p>
<p><strong>Progress to date ({resolved}/{total} frozen elements resolved,
{pct}% reduced):</strong>
Through {n_sessions} experimental session{"s" if n_sessions != 1 else ""} comprising
{n_experiments} experiments, we have accumulated {n_constraints} constraints (what
does not work and why) and {n_discoveries} discoveries. The frozen frame currently
contains {n_frozen} element{"s" if n_frozen != 1 else ""}; {n_adaptive}
element{"s" if n_adaptive != 1 else ""} ha{"ve" if n_adaptive != 1 else "s"} become
adaptive{f", {n_calibrated} calibrated (binding but not adaptive)" if n_calibrated else ""}.
Currently working on Stage {stage} of 8.</p>
<p><strong>Current status:</strong> {esc(state_info["status"])}</p>
<p>This is a work in progress. We present the constitution as a reusable framework,
the experimental log as an honest record of what works and what doesn't, and the
open problems as an invitation for review and collaboration.</p>
</div>"""


def generate_constitution(claude_md):
    sections = extract_constitution(claude_md)
    parts = ['<h2>The Constitution</h2>',
             '<p><em>Extracted verbatim from the project constitution (CLAUDE.md). '
             'This is the theoretical framework governing all experiments.</em></p>']
    for name, content in sections:
        parts.append(f'<h3>{esc(name)}</h3>')
        parts.append(md_to_html(content))
    return '\n'.join(parts)


def generate_architecture(entries):
    arch = [e for e in entries if e['type'] == 'architecture']
    if not arch:
        return ''
    parts = ['<h2>Architecture</h2>']
    for a in arch:
        parts.append(f'<div class="entry architecture">')
        parts.append(f'<h4>{esc(a["title"])}</h4>')
        parts.append(render_entry_content(a['content']))
        parts.append('</div>')
    return '\n'.join(parts)


def generate_methodology(entries):
    """Methodology from methodology-tagged entries."""
    method_ids = set()
    for e in entries:
        if 'methodology' in e.get('tags', []) or 'protocol' in e.get('tags', []):
            method_ids.add(e['id'])

    method_entries = [e for e in entries if e['id'] in method_ids]
    method_entries.sort(key=lambda e: e.get('session', 0))

    if not method_entries:
        return ''

    parts = ['<h2>Methodology</h2>',
             '<p>The experimental methodology evolved through systematic discovery '
             'of variance sources and protocol refinements.</p>']
    for m in method_entries:
        parts.append(f'<div class="entry">')
        parts.append(f'<h4>{esc(m["title"])} {status_badge(m["status"])}</h4>')
        parts.append(f'<div class="tags">Session {m.get("session", "?")} &middot; '
                     f'{", ".join(m.get("tags", []))}</div>')
        parts.append(render_entry_content(m['content']))
        parts.append('</div>')
    return '\n'.join(parts)


def generate_experiments(entries):
    experiments = [e for e in entries if e['type'] == 'experiment']
    by_session = defaultdict(list)
    for exp in experiments:
        by_session[exp.get('session', 0)].append(exp)

    if not experiments:
        return ''

    parts = ['<h2>Experimental Program</h2>',
             f'<p>{len(experiments)} experiments across '
             f'{len(by_session)} sessions.</p>']

    for session_num in sorted(by_session.keys()):
        exps = by_session[session_num]
        parts.append(f'<h3>Session {session_num}</h3>')
        for exp in exps:
            parts.append(f'<div class="entry experiment">')
            parts.append(f'<h4>{esc(exp["title"])} {status_badge(exp["status"])}</h4>')
            parts.append(f'<div class="tags">{", ".join(exp.get("tags", []))}</div>')
            parts.append(render_entry_content(exp['content']))
            if exp.get('constraints_implied'):
                parts.append('<p><strong>Constraints discovered:</strong></p><ul>')
                for c in exp['constraints_implied']:
                    parts.append(f'<li>{esc(c)}</li>')
                parts.append('</ul>')
            parts.append('</div>')

    return '\n'.join(parts)


def generate_discoveries(entries):
    discoveries = [e for e in entries
                   if e['type'] == 'discovery' and e.get('status') != 'superseded']
    discoveries.sort(key=lambda e: e.get('session', 0))

    if not discoveries:
        return ''

    parts = ['<h2>Discoveries</h2>',
             f'<p>{len(discoveries)} active discoveries.</p>']
    for disc in discoveries:
        parts.append(f'<div class="entry discovery">')
        parts.append(f'<h4>Session {disc.get("session", "?")} &mdash; '
                     f'{esc(disc["title"])} {status_badge(disc["status"])}</h4>')
        parts.append(f'<div class="tags">{", ".join(disc.get("tags", []))}</div>')
        parts.append(render_entry_content(disc['content']))
        if disc.get('constraints_implied'):
            parts.append('<p><strong>Constraints implied:</strong></p><ul>')
            for c in disc['constraints_implied']:
                parts.append(f'<li>{esc(c)}</li>')
            parts.append('</ul>')
        parts.append('</div>')

    return '\n'.join(parts)


def generate_constraints(constraints):
    if not constraints:
        return ''
    parts = ['<h2>Constraints</h2>',
             f'<p>{len(constraints)} experimentally-derived rules defining what '
             'approaches do not work.</p>',
             '<table><thead><tr><th>ID</th><th>Constraint</th><th>Tags</th>'
             '<th>Source</th></tr></thead><tbody>']
    for c in constraints:
        tags = ', '.join(c.get('tags', [])[:3])
        sources = ', '.join(c.get('source_entries', []))
        parts.append(f'<tr><td>{esc(c["id"])}</td><td>{esc(c["rule"])}</td>'
                     f'<td class="tags">{esc(tags)}</td>'
                     f'<td>{esc(sources)}</td></tr>')
    parts.append('</tbody></table>')
    return '\n'.join(parts)


def compute_frozen_timeline(frozen_frame, entries):
    """Derive frozen frame progression over time from existing data.

    Data sources (nothing hardcoded):
    - frozen_frame.json: each element has session_changed (when status changed)
    - entries/*.json: each has session number (gives the full session list)

    The timeline is fully derived. Plateaus are visible because the compiler
    generates a row for every session that has entries, and the frozen count
    carries forward from the last thawing event.
    """
    all_sessions = sorted(set(
        e.get('session') for e in entries if e.get('session')
    ))
    if not all_sessions:
        return []

    total = len(frozen_frame)

    # Build thawing/resolution events by session from frozen_frame.json
    thaw_events = defaultdict(list)
    for elem in frozen_frame:
        if elem['status'].startswith('adaptive') or elem['status'] == 'calibrated':
            s = elem.get('session_changed', 1)
            label = elem['element']
            if elem['status'] == 'calibrated':
                label += ' (calibrated)'
            thaw_events[s].append(label)

    # Compute per-session experiment stats from entries
    session_stats = defaultdict(lambda: {
        'experiments': 0, 'failed': 0, 'succeeded': 0,
        'partial': 0, 'constraints_new': 0
    })
    for entry in entries:
        s = entry.get('session')
        if s is None:
            continue
        if entry['type'] == 'experiment':
            session_stats[s]['experiments'] += 1
            status = entry.get('status', '')
            if status in ('failed', 'succeeded', 'partial'):
                session_stats[s][status] += 1
        session_stats[s]['constraints_new'] += len(
            entry.get('constraints_implied', [])
        )

    # Build cumulative timeline
    timeline = []
    cumulative_thawed = 0
    cumulative_constraints = 0

    for session in all_sessions:
        thawed_this = thaw_events.get(session, [])
        cumulative_thawed += len(thawed_this)
        cumulative_constraints += session_stats[session]['constraints_new']

        timeline.append({
            'session': session,
            'frozen': total - cumulative_thawed,
            'adaptive': cumulative_thawed,
            'pct_reduced': round(cumulative_thawed / total * 100, 1),
            'thawed_elements': thawed_this,
            'experiments': session_stats[session]['experiments'],
            'exp_failed': session_stats[session]['failed'],
            'exp_succeeded': session_stats[session]['succeeded'],
            'exp_partial': session_stats[session]['partial'],
            'constraints_total': cumulative_constraints,
        })

    return timeline


def generate_progress_timeline(frozen_frame, entries):
    """Generate the Progress Over Time section for the paper.

    Entirely derived from frozen_frame.json and entries/*.json.
    No hardcoded values. Plateaus emerge automatically because the
    frozen count carries forward through sessions with no thawing events.
    """
    timeline = compute_frozen_timeline(frozen_frame, entries)
    if not timeline:
        return ''

    total = len(frozen_frame)

    parts = [
        '<h2>Progress Over Time</h2>',
        '<p>Frozen frame reduction trajectory, derived from experimental entries. '
        'Each row shows the cumulative state at the end of that session. '
        'Green = adaptive, red = frozen.</p>'
    ]

    # Visual bar chart
    parts.append('<div class="timeline-chart">')
    for point in timeline:
        adaptive_pct = point['pct_reduced']
        frozen_pct = 100 - adaptive_pct

        annotation = ''
        if point['thawed_elements']:
            elements = ', '.join(point['thawed_elements'])
            annotation = f'<span class="timeline-event">{esc(elements)} thawed</span>'

        parts.append(
            f'<div class="timeline-row">'
            f'<span class="timeline-label">S{point["session"]}</span>'
            f'<div class="timeline-bar">'
            f'<div class="bar-adaptive" style="width:{max(adaptive_pct, 0)}%"></div>'
            f'<div class="bar-frozen" style="width:{frozen_pct}%"></div>'
            f'</div>'
            f'<span class="timeline-count">{point["frozen"]}/{total}</span>'
            f'{annotation}'
            f'</div>'
        )
    parts.append('</div>')

    # Summary table
    parts.append(
        '<table class="timeline-table"><thead><tr>'
        '<th>Session</th><th>Frozen</th><th>Reduced</th>'
        '<th>Experiments</th><th>Outcomes</th>'
        '<th>Constraints</th><th>Event</th>'
        '</tr></thead><tbody>'
    )

    for point in timeline:
        outcomes = []
        if point['exp_succeeded']:
            outcomes.append(f'{point["exp_succeeded"]} pass')
        if point['exp_partial']:
            outcomes.append(f'{point["exp_partial"]} partial')
        if point['exp_failed']:
            outcomes.append(f'{point["exp_failed"]} fail')
        outcome_str = ', '.join(outcomes) if outcomes else '&mdash;'

        event = (', '.join(point['thawed_elements']) + ' thawed'
                 if point['thawed_elements'] else '&mdash;')

        row_class = ' class="timeline-change"' if point['thawed_elements'] else ''

        parts.append(
            f'<tr{row_class}>'
            f'<td>{point["session"]}</td>'
            f'<td>{point["frozen"]}/{total}</td>'
            f'<td>{point["pct_reduced"]}%</td>'
            f'<td>{point["experiments"]}</td>'
            f'<td>{outcome_str}</td>'
            f'<td>{point["constraints_total"]}</td>'
            f'<td>{event}</td>'
            f'</tr>'
        )

    parts.append('</tbody></table>')

    # Detect and report plateau
    if len(timeline) >= 2:
        current_frozen = timeline[-1]['frozen']
        # Find the first session where frozen count reached current level
        plateau_start = None
        for point in timeline:
            if point['frozen'] == current_frozen:
                plateau_start = point['session']
                break

        current_session = timeline[-1]['session']
        if plateau_start and plateau_start < current_session:
            sessions_on_plateau = sum(
                1 for p in timeline
                if p['session'] > plateau_start
            )
            parts.append(
                f'<p class="plateau-note"><strong>Current plateau:</strong> '
                f'Frozen frame unchanged at {current_frozen}/{total} for '
                f'{sessions_on_plateau} session'
                f'{"s" if sessions_on_plateau != 1 else ""} '
                f'since Session {plateau_start}.</p>'
            )

    return '\n'.join(parts)


def generate_frozen_frame(frozen_frame, progress):
    parts = ['<h2>Current Frozen Frame</h2>']
    pct = progress['pct_reduced']
    parts.append(
        f'<p><strong>Frozen frame reduction:</strong> '
        f'{progress["adaptive"]}/{progress["total_elements"]} elements adapted '
        f'({pct}% reduced). {progress["frozen"]} element'
        f'{"s" if progress["frozen"] != 1 else ""} remain frozen.</p>'
    )
    parts.append(f'<div class="progress-bar"><div class="fill" '
                 f'style="width:{max(pct, 5)}%">{pct}% reduced</div></div>')
    parts.append('<table><thead><tr><th>Element</th><th>Description</th>'
                 '<th>Stage</th><th>Status</th></tr></thead><tbody>')
    for elem in frozen_frame:
        status = elem['status']
        if status == 'adaptive':
            stage_info = f'Thawed at Stage {elem.get("stage_thawed", "?")}'
        elif status == 'adaptive (vacuous)':
            stage_info = f'Vacuously passed at Stage {elem.get("stage_thawed", "?")}'
        elif status == 'calibrated':
            stage_info = (f'Calibrated at Stage {elem.get("stage_target", "?")} '
                         f'(Session {elem.get("session_changed", "?")})')
        else:
            stage_info = f'Target: Stage {elem.get("stage_target", "?")}'
        parts.append(f'<tr><td>{esc(elem["element"])}</td>'
                     f'<td>{esc(elem["description"])}</td>'
                     f'<td>{stage_info}</td>'
                     f'<td>{status_badge(status)}</td></tr>')
    parts.append('</tbody></table>')
    return '\n'.join(parts)


def generate_decisions(entries):
    decisions = [e for e in entries if e['type'] == 'decision']
    decisions.sort(key=lambda e: e.get('session', 0))

    if not decisions:
        return ''

    parts = ['<h2>Key Decisions</h2>']
    for dec in decisions:
        parts.append(f'<div class="entry decision">')
        parts.append(f'<h4>{esc(dec["title"])} {status_badge(dec["status"])}</h4>')
        parts.append(f'<div class="tags">Session {dec.get("session", "?")} &middot; '
                     f'{", ".join(dec.get("tags", []))}</div>')
        parts.append(render_entry_content(dec['content']))
        parts.append('</div>')

    return '\n'.join(parts)


# ============================================================
# HTML TEMPLATE
# ============================================================

CSS = """
body {
    font-family: 'Georgia', 'Times New Roman', serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 40px 20px;
    color: #222;
    line-height: 1.7;
    background: #fafafa;
}
h1 {
    text-align: center;
    border-bottom: 2px solid #333;
    padding-bottom: 20px;
    margin-bottom: 10px;
}
h1 .subtitle {
    display: block;
    font-size: 0.5em;
    font-style: italic;
    color: #555;
    margin-top: 8px;
}
h1 .date {
    display: block;
    font-size: 0.4em;
    color: #888;
    font-weight: normal;
    margin-top: 5px;
}
h2 {
    border-bottom: 1px solid #ccc;
    padding-bottom: 5px;
    margin-top: 40px;
}
h3 { margin-top: 30px; }
h4 { margin-top: 15px; margin-bottom: 5px; }
.abstract {
    background: #f0f0f0;
    padding: 20px;
    border-left: 4px solid #333;
    margin: 20px 0;
}
.abstract h2 { border: none; margin-top: 0; }
.entry {
    margin: 15px 0;
    padding: 15px;
    border-left: 3px solid #ddd;
    background: #fff;
}
.entry.experiment { border-left-color: #2196F3; }
.entry.discovery { border-left-color: #4CAF50; }
.entry.decision { border-left-color: #FF9800; }
.entry.architecture { border-left-color: #9C27B0; }
.tags {
    font-size: 0.85em;
    color: #888;
    font-style: italic;
    margin-bottom: 8px;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
    font-size: 0.9em;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
    vertical-align: top;
}
th { background: #f5f5f5; font-weight: bold; }
tr:nth-child(even) { background: #fafafa; }
.constitution h3 { color: #333; }
.constitution h4 { color: #444; }
.progress-bar {
    background: #e0e0e0;
    border-radius: 10px;
    height: 24px;
    margin: 10px 0;
    overflow: hidden;
}
.progress-bar .fill {
    background: linear-gradient(90deg, #4CAF50, #2196F3);
    height: 100%;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    font-size: 0.85em;
    min-width: 60px;
}
.meta-info {
    text-align: center;
    color: #666;
    font-size: 0.9em;
    margin-bottom: 30px;
}
ul { margin: 5px 0; padding-left: 25px; }
li { margin: 3px 0; }
.timeline-chart { margin: 20px 0; }
.timeline-row {
    display: flex;
    align-items: center;
    margin: 3px 0;
}
.timeline-label {
    width: 35px;
    font-size: 0.85em;
    font-weight: bold;
    text-align: right;
    margin-right: 8px;
    color: #555;
}
.timeline-bar {
    flex: 1;
    height: 18px;
    display: flex;
    border-radius: 3px;
    overflow: hidden;
    background: #e0e0e0;
}
.bar-adaptive { background: #4CAF50; }
.bar-frozen { background: #e57373; }
.timeline-count {
    width: 35px;
    font-size: 0.8em;
    color: #666;
    margin-left: 8px;
    text-align: center;
}
.timeline-event {
    font-size: 0.8em;
    color: #4CAF50;
    font-weight: bold;
    margin-left: 8px;
}
.timeline-table { font-size: 0.85em; }
tr.timeline-change { background: #e8f5e9 !important; }
.plateau-note {
    background: #fff3e0;
    padding: 10px 15px;
    border-left: 3px solid #FF9800;
    margin: 15px 0;
    font-size: 0.9em;
}
@media print {
    body { background: white; max-width: 100%; }
    .entry { break-inside: avoid; }
    .progress-bar { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    .timeline-bar { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    .plateau-note { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
}
"""


def build_html(sections, generated_at):
    """Assemble the complete HTML document from section strings."""
    body = '\n\n'.join(s for s in sections if s)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>The Singularity Search</title>
<style>{CSS}</style>
</head>
<body>
{body}
<hr>
<p style="text-align:center;color:#999;font-size:0.8em;">
Auto-generated from <code>knowledge/</code> on {generated_at}.<br>
Do not edit this file. Update <code>knowledge/entries/</code> and recompile.
</p>
</body>
</html>"""


# ============================================================
# MAIN
# ============================================================

def compile_paper():
    print("=== Paper Compiler ===")
    print(f"Source: {KNOWLEDGE_DIR}")

    # --- Load all data ---
    constraints = load_json(KNOWLEDGE_DIR / "constraints.json")
    frozen_frame = load_json(KNOWLEDGE_DIR / "frozen_frame.json")
    claude_md = load_text(PROJECT_ROOT / "CONSTITUTION.md")
    # Synthesize state_md from RESEARCH_STATE.md for backward compat
    state_md_path = PROJECT_ROOT / "RESEARCH_STATE.md"
    state_md = load_text(state_md_path) if state_md_path.exists() else ""
    entries = load_all_entries()

    # --- Compute dynamic values ---
    state_info = parse_state(state_md)
    progress = compute_progress(frozen_frame)
    stage = state_info['stage_num']
    session = state_info['session']
    pct_reduced = progress['pct_reduced']

    resolved = progress['adaptive'] + progress.get('calibrated', 0)
    print(f"Loaded: {len(entries)} entries, {len(constraints)} constraints, "
          f"{len(frozen_frame)} frame elements")
    print(f"State:  Stage {stage}/8, Session {session}")
    print(f"Frame:  {resolved}/{progress['total_elements']} resolved "
          f"({progress['adaptive']} adaptive, {progress.get('calibrated', 0)} calibrated) "
          f"({pct_reduced}% reduced)")

    # --- Title ---
    title_html = (
        f'<h1>The Singularity Search'
        f'<span class="subtitle">Recursive Self-Improvement Through '
        f'Monotonic Frozen Frame Reduction</span>'
        f'<span class="subtitle">Progress Report: Stage {stage} of 8</span>'
        f'<span class="date">{datetime.now().strftime("%B %Y")} '
        f'(Work in Progress)</span></h1>'
    )

    # --- Meta info bar ---
    resolved = progress['adaptive'] + progress.get('calibrated', 0)
    meta_html = (
        f'<div class="meta-info">'
        f'<div class="progress-bar"><div class="fill" '
        f'style="width:{max(pct_reduced, 5)}%">'
        f'{resolved}/{progress["total_elements"]} resolved '
        f'({pct_reduced}%)</div></div>'
        f'<p>Stage {stage}/8 &middot; '
        f'{len(entries)} knowledge entries &middot; '
        f'{len(constraints)} constraints &middot; '
        f'{max_session(entries)} sessions &middot; '
        f'{progress["frozen"]} frozen elements remaining</p></div>'
    )

    # --- Generate all sections ---
    abstract_html = generate_abstract(state_info, progress, constraints, entries)
    timeline_html = generate_progress_timeline(frozen_frame, entries)
    constitution_html = generate_constitution(claude_md)
    architecture_html = generate_architecture(entries)
    methodology_html = generate_methodology(entries)
    experiments_html = generate_experiments(entries)
    discoveries_html = generate_discoveries(entries)
    constraints_html = generate_constraints(constraints)
    frozen_html = generate_frozen_frame(frozen_frame, progress)
    decisions_html = generate_decisions(entries)

    # --- Assemble ---
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    full_html = build_html([
        title_html,
        meta_html,
        abstract_html,
        timeline_html,
        constitution_html,
        architecture_html,
        methodology_html,
        experiments_html,
        discoveries_html,
        constraints_html,
        frozen_html,
        decisions_html,
    ], generated_at)

    # --- Write ---
    PAPER_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(full_html)

    print(f"\n[OK] {OUTPUT_FILE}")
    print(f"     {len(full_html):,} bytes")


if __name__ == "__main__":
    compile_paper()
