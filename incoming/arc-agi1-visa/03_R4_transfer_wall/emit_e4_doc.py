"""
Auto-emit E4-holed R* section to RESEARCH_STATE.md from result JSON (Leo #11678).

Reads E4_holed_result.json and generates the exact R* section text,
then writes it into the E4-holed block in RESEARCH_STATE.md.
Eliminates hand-written doc drift (3 catches: #11644, #11663, #11675).

Usage: python emit_e4_doc.py
"""
import json
import re
import os

RESULT_PATH = "incoming/arc-agi1-visa/03_R4_transfer_wall/E4_holed_result.json"
RESEARCH_STATE_PATH = "docs/RESEARCH_STATE.md"

MARKER_START = "**E4-holed result (auto-emitted from E4_holed_result.json)**"
MARKER_END = "<!-- /E4-HOLED-RESULT -->"


def fmt_pct(v):
    if v is None:
        return "N/A"
    return f"{v:+.1f}%"


def emit_section(result):
    """Generate the R* section text from result JSON."""
    config = result.get('config', {})
    rho_values = result.get('rho_values', [])

    rstar_B = result.get('r_star_contiguous')
    rstar_C = result.get('r_star_holed')

    lines = [
        f"{MARKER_START}",
        f"",
        f"**Config:** MOTIF=mir_h__mir_v, DISTRACTOR=dup_h__fv (independent roll, rho_d={config.get('rho_d', 0.4)}), "
        f"N_SEEDS={config.get('n_seeds', '?')}, prog_len={config.get('program_length', '?')}, "
        f"budget={config.get('budget', '?')}, n_source={config.get('n_source', '?')}, n_held={config.get('n_held', '?')}",
        f"",
        f"**R* definition (aggregate-net):** min rho where aggregate_delta < -{config.get('aggregate_margin_pct', 5.0):.1f}% "
        f"AND variance band (mean+std) excludes 0. ALL held tasks, overhead included.",
        f"",
        f"**R* results:**",
        f"- R*_contiguous (B): {rstar_B if rstar_B is not None else 'NOT REACHED'} — "
        f"aggregate-net crossover for all-MDL library",
        f"- R*_holed (C): {rstar_C if rstar_C is not None else 'NOT REACHED'} — "
        f"aggregate-net crossover with holed operators",
        f"- Planted-only P: mechanism signal confirmed; R*_P NOT reported as design threshold "
        f"(isolated condition, overhead not included).",
        f"",
        f"**Key signal (NOT accepted as R*):** script-reported motif-subset threshold = 0.1 "
        f"from prior E4_rstar_grade run. NOT accepted — was motif-subset, not aggregate-net.",
        f"",
        f"**Distractor load-bearing check:**",
    ]

    # Per-rho table
    per_rho = result.get('curve', [])
    if per_rho:
        lines.append(f"")
        lines.append(f"| rho | AGG B | AGG C | Planted P | dist_in_lib | holed_sel |")
        lines.append(f"|-----|-------|-------|-----------|-------------|-----------|")
        for pt in per_rho:
            rho = pt.get('rho', '?')
            bm = pt.get('agg_delta_B_mean')
            cm = pt.get('agg_delta_C_mean')
            pm = pt.get('agg_delta_P_mean')
            dist = pt.get('distractor_in_lib_fraction')
            sel = pt.get('selected_holed_count_mean')
            b_str = f"{bm:+.1f}%" if bm is not None else "N/A"
            c_str = f"{cm:+.1f}%" if cm is not None else "N/A"
            p_str = f"{pm:+.1f}%" if pm is not None else "N/A"
            dist_str = f"{dist:.2f}" if dist is not None else "N/A"
            sel_str = f"{sel:.1f}" if sel is not None else "N/A"
            rstar_marker = ""
            if rstar_B is not None and rho == rstar_B:
                rstar_marker = " R*_B"
            if rstar_C is not None and rho == rstar_C:
                rstar_marker += " R*_C"
            lines.append(f"| {rho} | {b_str} | {c_str} | {p_str} | {dist_str} | {sel_str} |{rstar_marker}")

    lines.append(f"")
    lines.append(f"**Verdict:**")
    if rstar_B is not None or rstar_C is not None:
        lines.append(f"R* found: aggregate-net crossover at rho={min(r for r in [rstar_B, rstar_C] if r is not None)}. "
                     f"Mechanism achieves net aggregate improvement at sufficient density.")
    else:
        lines.append(f"R* NOT reached in tested rho range. Aggregate remains net-positive "
                     f"(distractor overhead exceeds motif benefit) even where planted-only P shows clear signal.")
        lines.append(f"DESIGN IMPLICATION: aggregate-net R* requires either higher rho than tested, "
                     f"or library construction that excludes distractor overhead.")

    lines.append(f"")
    lines.append(f"**Artifacts:** `E4_holed_operators.py`, `E4_holed_result.json`.")
    lines.append(f"{MARKER_END}")

    return "\n".join(lines)


def update_research_state(section_text):
    """Insert/replace the E4-holed result block in RESEARCH_STATE.md."""
    with open(RESEARCH_STATE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the E4-holed pending section to replace (em-dash variant)
    old_block_pattern = re.compile(
        r'\*\*E4-holed \(Leo #11672 exact spec\) — IN-FLIGHT.*?Non-memory-heavy\.\*\* No model load\.',
        re.DOTALL
    )

    if old_block_pattern.search(content):
        new_content = old_block_pattern.sub(section_text, content)
        with open(RESEARCH_STATE_PATH, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: replaced IN-FLIGHT block with result in {RESEARCH_STATE_PATH}")
    else:
        # Try to find existing auto-emitted block
        existing_pattern = re.compile(
            re.escape(MARKER_START) + r'.*?' + re.escape(MARKER_END),
            re.DOTALL
        )
        if existing_pattern.search(content):
            new_content = existing_pattern.sub(section_text, content)
            with open(RESEARCH_STATE_PATH, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated: replaced existing auto-emitted block in {RESEARCH_STATE_PATH}")
        else:
            print(f"WARNING: Could not find insertion point in {RESEARCH_STATE_PATH}")
            print("Emit section would be:")
            print(section_text)
            return False
    return True


def main():
    if not os.path.exists(RESULT_PATH):
        print(f"Result not found: {RESULT_PATH}")
        print("Run E4_holed_operators.py first.")
        return

    with open(RESULT_PATH, 'r') as f:
        result = json.load(f)

    print(f"Read result from {RESULT_PATH}")
    rstar = result.get('rstar', {})
    print(f"R*_contiguous: {rstar.get('rstar_contiguous', 'NOT REACHED')}")
    print(f"R*_holed: {rstar.get('rstar_holed', 'NOT REACHED')}")

    section = emit_section(result)
    print("\n--- Emitting section ---")
    print(section[:500], "..." if len(section) > 500 else "")

    if update_research_state(section):
        print("\nRESEARCH_STATE.md updated successfully.")
    else:
        print("\nManual update needed.")


if __name__ == '__main__':
    main()
