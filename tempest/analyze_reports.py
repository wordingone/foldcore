#!/usr/bin/env python3
"""
Meta-analysis of Tempest reports for ground truth criteria correlation.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

def extract_metrics(report_path: str) -> Dict:
    """Extract key metrics from a report."""
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    result = {
        'path': report_path,
        'f_number': None,
        'language': None,
        'initial_density': None,
        'final_density': None,
        'cycles': None,
        'physics_description': '',
        'score_distinguishability': 0,
        'score_energy': 0,
        'score_self_reference': 0,
        'score_memory': 0,
        'total_score': 0,
        'extinction_cycle': None,
        'stable_density': None
    }

    # Extract f-number and language from path
    parts = report_path.replace('\\', '/').split('/')
    if 'reports' in parts:
        idx = parts.index('reports')
        if idx + 1 < len(parts):
            result['language'] = parts[idx + 1]

    # Extract f-number from filename or header
    fn_match = re.search(r'f(\d+)', content[:500], re.IGNORECASE)
    if fn_match:
        result['f_number'] = int(fn_match.group(1))

    # Extract cycles
    cycles_match = re.search(r'\|\s*C\s*\|\s*(\d+)\s*cycles', content[:2000])
    if cycles_match:
        result['cycles'] = int(cycles_match.group(1))

    # Extract physics description
    phys_match = re.search(r'## The Physics \(f\)(.*?)(?=##|\Z)', content, re.DOTALL)
    if phys_match:
        result['physics_description'] = phys_match.group(1)[:500]

    # Find all density observations
    density_pattern = r'(?:Density|density)[:\s]*([0-9.]+)'
    densities = re.findall(density_pattern, content)

    if densities:
        try:
            # Clean density values (remove trailing dots, validate format)
            clean_densities = []
            for d in densities:
                d_clean = d.rstrip('.')
                if d_clean and d_clean != '.':
                    clean_densities.append(float(d_clean))

            if clean_densities:
                result['initial_density'] = clean_densities[0]
                result['final_density'] = clean_densities[-1]
                densities = [str(d) for d in clean_densities]  # Update for later use
        except ValueError:
            pass  # Keep None values if parsing fails

    # Check for extinction (final density = 0)
    if result['final_density'] == 0.0:
        # Find when extinction occurred
        cycle_density_pattern = r'(?:Cycle|cycle)\s+(\d+).*?(?:density|Density)[:\s]*0\.0+\b'
        matches = re.findall(cycle_density_pattern, content)
        if matches:
            result['extinction_cycle'] = int(matches[0])

    # Check for stable density (last 3-4 observations identical)
    if len(densities) >= 4:
        last_densities = [float(d) for d in densities[-4:]]
        if len(set(last_densities)) == 1 and last_densities[0] > 0:
            result['stable_density'] = last_densities[0]

    # Score criteria
    result = score_criteria(result, content)

    return result

def score_criteria(result: Dict, content: str) -> Dict:
    """Score the 4 ground truth criteria."""
    physics = result['physics_description'].lower()
    full_content_lower = content.lower()

    # 1. Distinguishability preservation
    # Score 1 if final density > 0 and distinct patterns survive
    # Score 0.5 if density decreases but nonzero
    # Score 0 if extinction

    if result['final_density'] is not None and result['initial_density'] is not None:
        if result['final_density'] == 0:
            result['score_distinguishability'] = 0
        elif result['final_density'] >= result['initial_density'] * 0.5:
            # Maintained >50% of initial diversity
            result['score_distinguishability'] = 1
        elif result['final_density'] > 0:
            # Some survival but < 50%
            result['score_distinguishability'] = 0.5

    # 2. Energy conservation or bounded fluctuation
    # Look for conservation keywords, or stable 1s count
    # Score 0 if mentions "decay", "damping", "dissipation" without compensation

    if 'conserv' in physics or 'constant' in physics:
        result['score_energy'] = 1
    elif any(word in physics for word in ['decay', 'damp', 'dissipat', 'loss']):
        # Check if there's compensation
        if any(word in physics for word in ['flow', 'birth', 'input', 'creation']):
            result['score_energy'] = 0.5
        else:
            result['score_energy'] = 0
    elif result['stable_density'] is not None:
        # Density stabilized = some energy balance
        result['score_energy'] = 0.5

    # 3. Self-reference capability
    # Look for address-based operations, indirect reference
    # Score 0 for pure neighbor-count threshold rules

    if any(word in physics for word in ['address', 'reference', 'pointer', 'indirect']):
        result['score_self_reference'] = 1
    elif any(word in physics for word in ['neighbor', 'adjacent', 'local', 'count']):
        # Pure local threshold rule - no self-reference
        result['score_self_reference'] = 0
    else:
        # Unclear - give 0.5 if not obviously threshold-based
        if 'threshold' not in physics:
            result['score_self_reference'] = 0.5

    # 4. Accumulation/memory
    # Look for multi-field state, history dependence, integration
    # Score 0 if state depends only on immediate neighbors at t-1

    if any(word in physics for word in ['accumul', 'integrat', 'history', 'memory', 'sum']):
        result['score_memory'] = 1
    elif any(word in full_content_lower for word in ['pressure', 'tension', 'vorticity', 'amplitude', 'velocity', 'field']):
        # Multi-field state = some accumulation
        result['score_memory'] = 0.5
    elif 'neighbor' in physics and 'count' in physics:
        # Pure neighbor count = no memory
        result['score_memory'] = 0

    result['total_score'] = (
        result['score_distinguishability'] +
        result['score_energy'] +
        result['score_self_reference'] +
        result['score_memory']
    )

    return result

def main():
    reports_dir = Path("./reports")

    # Find all report files
    report_files = list(reports_dir.rglob("*.md"))
    print(f"Found {len(report_files)} report files")

    results = []
    for report_path in sorted(report_files):
        try:
            metrics = extract_metrics(str(report_path))
            results.append(metrics)
            print(f"Processed: f{metrics['f_number']} ({metrics['language']}) - Score: {metrics['total_score']:.1f}/4")
        except Exception as e:
            print(f"Error processing {report_path}: {e}")

    # Statistical analysis
    print(f"\n=== ANALYSIS OF {len(results)} REPORTS ===\n")

    # Group by score
    high_score = [r for r in results if r['total_score'] >= 3.0]
    low_score = [r for r in results if r['total_score'] <= 1.0]
    mid_score = [r for r in results if 1.0 < r['total_score'] < 3.0]

    print(f"High score (>=3/4): {len(high_score)} reports")
    print(f"Mid score (1-3): {len(mid_score)} reports")
    print(f"Low score (<=1/4): {len(low_score)} reports\n")

    # Test prediction: high-score maintains distinguishability longer
    def avg_final_density(group):
        valid = [r['final_density'] for r in group if r['final_density'] is not None]
        return sum(valid) / len(valid) if valid else 0

    def avg_extinction_cycle(group):
        valid = [r['extinction_cycle'] for r in group if r['extinction_cycle'] is not None]
        return sum(valid) / len(valid) if valid else None

    print("PREDICTION TEST:")
    print(f"High-score avg final density: {avg_final_density(high_score):.4f}")
    print(f"Low-score avg final density: {avg_final_density(low_score):.4f}")

    high_extinct = [r for r in high_score if r['final_density'] == 0]
    low_extinct = [r for r in low_score if r['final_density'] == 0]

    high_ext_rate = len(high_extinct)/len(high_score)*100 if high_score else 0
    low_ext_rate = len(low_extinct)/len(low_score)*100 if low_score else 0
    print(f"\nHigh-score extinction rate: {len(high_extinct)}/{len(high_score)} = {high_ext_rate:.1f}%")
    print(f"Low-score extinction rate: {len(low_extinct)}/{len(low_score)} = {low_ext_rate:.1f}%")

    # Identify high-scorers with survival
    survivors_high = [r for r in high_score if r['final_density'] and r['final_density'] > 0]
    print(f"\n=== HIGH-SCORE SURVIVORS ({len(survivors_high)}) ===")
    for r in survivors_high[:10]:
        print(f"f{r['f_number']:03d} ({r['language']:12s}): score={r['total_score']:.1f}, final_density={r['final_density']:.4f}")

    # Write detailed CSV
    with open("./analysis_results.csv", 'w') as f:
        f.write("f_number,language,cycles,init_density,final_density,extinction_cycle,stable_density,")
        f.write("score_dist,score_energy,score_selfref,score_memory,total_score\n")

        for r in results:
            f.write(f"{r['f_number'] or 0},{r['language'] or 'unknown'},{r['cycles'] or 0},")
            f.write(f"{r['initial_density'] or 0},{r['final_density'] or 0},")
            f.write(f"{r['extinction_cycle'] or ''},{r['stable_density'] or ''},")
            f.write(f"{r['score_distinguishability']},{r['score_energy']},")
            f.write(f"{r['score_self_reference']},{r['score_memory']},{r['total_score']}\n")

    print(f"\nDetailed results written to analysis_results.csv")

    return results

if __name__ == "__main__":
    main()
