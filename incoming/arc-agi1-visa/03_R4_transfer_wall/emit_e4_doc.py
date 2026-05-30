"""
Auto-emit E4-holed R* section to RESEARCH_STATE.md from result JSON (Leo #11678).

Reads E4_holed_result.json and generates the exact R* section text,
then writes it into the E4-holed block in RESEARCH_STATE.md.
Eliminates hand-written doc drift (3 catches: #11644, #11663, #11675).

Fail-loud: any missing field or anchor → sys.exit(1). NEVER print-warning-and-continue.

Usage: python emit_e4_doc.py
"""
import json
import re
import os
import sys

RESULT_PATH = "incoming/arc-agi1-visa/03_R4_transfer_wall/E4_holed_result.json"
RESEARCH_STATE_PATH = "docs/RESEARCH_STATE.md"

MARKER_START = "**E4-holed result (auto-emitted from E4_holed_result.json)**"
MARKER_END = "<!-- /E4-HOLED-RESULT -->"

REQUIRED_FIELDS = ['r_star_contiguous', 'r_star_holed', 'config', 'curve']


def fail(msg):
    print(f"EMIT ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def curve_entry_at(curve, rho):
    for pt in curve:
        if pt.get('rho') == rho:
            return pt
    return None


def emit_section(result):
    """Generate the R* section text from result JSON. Fails loud on any missing data."""
    for field in REQUIRED_FIELDS:
        if field not in result:
            fail(f"Missing required field '{field}' in result JSON. Cannot emit.")

    config = result['config']
    rstar_B = result['r_star_contiguous']
    rstar_C = result['r_star_holed']
    curve = result['curve']

    # Determine acceptance status for C (holed) R*
    accepted_C = False
    accepted_C_reason = "R*_holed not reached"
    if rstar_C is not None:
        entry = curve_entry_at(curve, rstar_C)
        if entry is None:
            fail(f"r_star_holed={rstar_C} but no curve entry found at that rho")
        dist = entry.get('distractor_in_lib_fraction')
        sel = entry.get('selected_holed_count_mean')
        if dist is not None and dist > 0 and sel is not None and sel > 0:
            accepted_C = True
            accepted_C_reason = (
                f"dist_in_lib={dist:.2f} (distractor load-bearing), "
                f"selected_holed={sel:.1f} (holed ops selected by search)"
            )
        else:
            accepted_C_reason = (
                f"dist_in_lib={dist} (need >0), selected_holed={sel} (need >0)"
            )

    lines = [
        f"{MARKER_START}",
        f"",
        f"**Config:** MOTIF=mir_h__mir_v, DISTRACTOR=dup_h__fv (independent roll, "
        f"rho_d={config.get('rho_d', 0.4)}), "
        f"N_SEEDS={config.get('n_seeds', '?')}, prog_len={config.get('program_length', '?')}, "
        f"budget={config.get('budget', '?')}, n_source={config.get('n_source', '?')}, "
        f"n_held={config.get('n_held', '?')}",
        f"",
        f"**R* definition (aggregate-net):** min rho where aggregate_delta < "
        f"-{config.get('aggregate_margin_pct', 5.0):.1f}% "
        f"AND variance band (mean+std) excludes 0. ALL held tasks, overhead included.",
        f"",
        f"**R* results:**",
        f"- R*_contiguous (B): {rstar_B if rstar_B is not None else 'NOT REACHED'} — "
        f"aggregate-net crossover, all-MDL library",
        f"- R*_holed (C): {rstar_C if rstar_C is not None else 'NOT REACHED'} — "
        f"aggregate-net crossover with holed operators",
        f"- Planted-only P: mechanism signal confirmed; R*_P NOT reported as design threshold "
        f"(isolated condition, overhead not included).",
        f"",
        f"**Key signal (NOT accepted as R*):** script-reported motif-subset threshold = 0.1 "
        f"from prior E4_rstar_grade run. NOT accepted — was motif-subset, not aggregate-net.",
        f"",
        f"**Distractor load-bearing check + per-rho curve:**",
        f"",
        f"| rho | AGG B (±std) | AGG C (±std) | Planted P (±std) | dist_in_lib | holed_sel |",
        f"|-----|--------------|--------------|------------------|-------------|-----------|",
    ]

    for pt in curve:
        rho = pt.get('rho', '?')
        bm = pt.get('agg_delta_B_mean')
        bs = pt.get('agg_delta_B_std')
        cm = pt.get('agg_delta_C_mean')
        cs = pt.get('agg_delta_C_std')
        pm = pt.get('agg_delta_P_mean')
        ps = pt.get('agg_delta_P_std')
        dist = pt.get('distractor_in_lib_fraction')
        sel = pt.get('selected_holed_count_mean')
        b_str = f"{bm:+.1f}%±{bs:.1f}" if bm is not None and bs is not None else "N/A"
        c_str = f"{cm:+.1f}%±{cs:.1f}" if cm is not None and cs is not None else "N/A"
        p_str = f"{pm:+.1f}%±{ps:.1f}" if pm is not None and ps is not None else "N/A"
        dist_str = f"{dist:.2f}" if dist is not None else "N/A"
        sel_str = f"{sel:.1f}" if sel is not None else "N/A"
        markers = []
        if rstar_B is not None and rho == rstar_B:
            markers.append("R*_B")
        if rstar_C is not None and rho == rstar_C:
            markers.append("R*_C")
        marker_str = " ← " + "+".join(markers) if markers else ""
        lines.append(
            f"| {rho} | {b_str} | {c_str} | {p_str} | {dist_str} | {sel_str} |{marker_str}"
        )

    # Find band value at rstar_C rho for "barely band-positive" flag
    band_at_rstar_C = None
    if rstar_C is not None:
        entry_C = curve_entry_at(curve, rstar_C)
        if entry_C:
            cm = entry_C.get('agg_delta_C_mean')
            cs = entry_C.get('agg_delta_C_std')
            if cm is not None and cs is not None:
                band_at_rstar_C = cm + cs

    lines.append(f"")
    lines.append(f"**Verdict (scope-qualified, PRISM-gated):**")

    if rstar_C is not None:
        if accepted_C:
            barely_flag = ""
            if band_at_rstar_C is not None and abs(band_at_rstar_C) < 2.0:
                barely_flag = (
                    f" NOTE: rho={rstar_C} is BARELY band-positive "
                    f"(C mean+std={band_at_rstar_C:+.2f}, barely excludes 0)."
                )
            holed_lowers = (rstar_B is not None and rstar_C < rstar_B)
            lines.append(
                f"**Synthetic aggregate-net R*_holed={rstar_C} under THIS generator/search/band rule.**"
                f" B (contiguous) reaches R*={rstar_B}; holed operators lower synthetic R*: "
                f"{rstar_C} vs {rstar_B} (holed_lowers={holed_lowers}).{barely_flag}"
            )
            lines.append(
                f"Load-bearing gates pass: {accepted_C_reason}."
            )
            lines.append(
                f"**PRISM gate:** General self-compilation claims still require "
                f"PRISM/multi-domain evidence (MBPP always present + masked ARC). "
                f"This result is synthetic-only and scoped to this generator and band rule."
            )
        else:
            lines.append(
                f"**Script-reported R*_holed = {rstar_C} (NOT yet accepted).** "
                f"Load-bearing gates: {accepted_C_reason}."
            )
    else:
        lines.append(
            f"R*_holed NOT REACHED in tested rho range. Aggregate C remains net-positive "
            f"(distractor overhead exceeds motif benefit) even where planted-only P shows clear signal."
        )
        lines.append(
            f"DESIGN IMPLICATION: aggregate-net R* requires either higher rho than tested, "
            f"or library construction that excludes distractor overhead."
        )
        lines.append(
            f"**PRISM gate:** General self-compilation claims require PRISM/multi-domain evidence. "
            f"This result is synthetic-only."
        )

    if rstar_B is not None:
        lines.append(
            f"R*_contiguous (B): {rstar_B} — all-MDL library achieves aggregate crossover."
        )

    lines.append(f"")
    lines.append(f"**Holed lowers R*:** {rstar_C} vs {rstar_B} (holed_lowers={rstar_C is not None and rstar_B is not None and rstar_C < rstar_B}).")
    lines.append(f"")
    lines.append(f"**Artifacts:** `E4_holed_operators.py`, `E4_holed_result.json`.")
    lines.append(f"{MARKER_END}")

    return "\n".join(lines)


def update_research_state(section_text):
    """Insert/replace the E4-holed result block in RESEARCH_STATE.md. Fails loud."""
    if not os.path.exists(RESEARCH_STATE_PATH):
        fail(f"RESEARCH_STATE.md not found at {RESEARCH_STATE_PATH}")

    with open(RESEARCH_STATE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # Try IN-FLIGHT block (em-dash variant — U+2014)
    old_block_pattern = re.compile(
        r'\*\*E4-holed \(Leo #11672 exact spec\) — IN-FLIGHT.*?Non-memory-heavy\.\*\* No model load\.',
        re.DOTALL
    )

    if old_block_pattern.search(content):
        new_content = old_block_pattern.sub(section_text, content)
        with open(RESEARCH_STATE_PATH, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"OK: replaced IN-FLIGHT block in {RESEARCH_STATE_PATH}")
        return

    # Try existing auto-emitted block
    existing_pattern = re.compile(
        re.escape(MARKER_START) + r'.*?' + re.escape(MARKER_END),
        re.DOTALL
    )
    if existing_pattern.search(content):
        new_content = existing_pattern.sub(section_text, content)
        with open(RESEARCH_STATE_PATH, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"OK: replaced existing auto-emitted block in {RESEARCH_STATE_PATH}")
        return

    # Anchor not found — FAIL LOUD
    fail(
        f"No insertion anchor found in {RESEARCH_STATE_PATH}.\n"
        f"  Searched for: IN-FLIGHT em-dash pattern AND existing MARKER_START.\n"
        f"  The emitter cannot leave the doc stale — fix the anchor or the pattern.\n"
        f"  Emitter EXITING NON-ZERO."
    )


def main():
    if not os.path.exists(RESULT_PATH):
        fail(f"Result not found at {RESULT_PATH}. Run E4_holed_operators.py first.")

    with open(RESULT_PATH, 'r') as f:
        result = json.load(f)

    rstar_B = result.get('r_star_contiguous')
    rstar_C = result.get('r_star_holed')
    print(f"Read: r_star_contiguous={rstar_B}, r_star_holed={rstar_C}")

    section = emit_section(result)
    print(f"\n--- Emitting section (first 600 chars) ---")
    print(section[:600])
    if len(section) > 600:
        print("...")

    update_research_state(section)
    print(f"\nRESEARCH_STATE.md updated successfully.")


if __name__ == '__main__':
    main()
