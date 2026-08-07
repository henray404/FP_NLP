#!/usr/bin/env python3
"""
Validate easy_clean.jsonl for training: check cara logically produces jawaban.
Heuristic + LLM-free first pass; flag suspicious records for manual review.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
import sys

def extract_boxed(text):
    """Extract \\boxed{...} from text if exists."""
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    return match.group(1) if match else None

def extract_final_number(text):
    """Extract last numeric value from text."""
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else None

def check_answer_format_consistency(cara, jawaban):
    """
    Check if jawaban format is consistent with cara's conclusion.
    Returns (is_consistent, reason).
    """
    issues = []

    # Extract boxed from cara
    boxed_in_cara = extract_boxed(cara)

    # Check 1: if cara has \\boxed, jawaban should match it closely
    if boxed_in_cara:
        # Normalize: remove spaces, LaTeX
        boxed_norm = re.sub(r'\\.*?[{}]', '', boxed_in_cara).strip()
        jawaban_norm = re.sub(r'\\.*?[{}]', '', jawaban).strip()

        # Allow minor variations (case, spacing, unit)
        if boxed_norm.lower() not in jawaban_norm.lower() and jawaban_norm.lower() not in boxed_norm.lower():
            issues.append(f"boxed '{boxed_in_cara}' ≠ jawaban '{jawaban}'")

    # Check 2: if cara ends with numeric result, jawaban should contain similar number
    last_num_cara = extract_final_number(cara)
    last_num_jaw = extract_final_number(jawaban)

    if last_num_cara and last_num_jaw:
        # Allow up to 1% difference (rounding)
        try:
            num_c = float(last_num_cara)
            num_j = float(last_num_jaw)
            if num_c != 0 and abs(num_c - num_j) / abs(num_c) > 0.01:
                issues.append(f"final number in cara={last_num_cara} vs jawaban={last_num_jaw}")
        except ValueError:
            pass

    # Check 3: if cara is very short (<50 chars) but jawaban is reasonable
    if len(cara) < 50 and len(jawaban) < 10:
        issues.append("very short cara and jawaban")

    # Check 4: soal-cara overlap (should share math keywords)
    return len(issues) == 0, issues

def check_soal_cara_alignment(soal, cara):
    """Check if cara addresses the soal."""
    # Extract keywords: numbers, math terms, variable names
    soal_tokens = set(re.findall(r'\b[a-zA-Z]+\b|\d+', soal.lower()))
    cara_tokens = set(re.findall(r'\b[a-zA-Z]+\b|\d+', cara.lower()))

    # Should overlap on at least some key terms
    overlap = soal_tokens & cara_tokens

    # Common words to ignore
    stop = {'adalah', 'yang', 'dari', 'dan', 'atau', 'jika', 'maka', 'untuk', 'dengan', 'pada', 'di', 'ke', 'se', 'telah', 'akan', 'dapat', 'ini', 'itu', 'menjadi', 'dimana'}
    overlap = overlap - stop

    if len(overlap) < 2:
        return False, "low term overlap between soal and cara"
    return True, None

def validate_jsonl(input_path, output_flagged_path, output_report_path):
    """Validate all records. Output flagged ones to separate file."""

    input_path = Path(input_path)
    output_flagged_path = Path(output_flagged_path)
    output_report_path = Path(output_report_path)

    stats = {
        'total': 0,
        'valid': 0,
        'flagged': 0,
        'issues': defaultdict(int),
    }

    flagged_records = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                stats['issues']['json_parse_error'] += 1
                flagged_records.append({
                    'line': line_no,
                    'reason': f"JSON parse error: {e}",
                    'record': line[:100]
                })
                stats['flagged'] += 1
                continue

            stats['total'] += 1
            soal = record.get('soal', '')
            cara = record.get('cara', '')
            jawaban = record.get('jawaban', '')

            issues_found = []

            # Check 1: field presence and length
            if not soal or len(soal) < 10:
                issues_found.append(f"soal too short ({len(soal)} chars)")
                stats['issues']['soal_too_short'] += 1

            if not cara or len(cara) < 20:
                issues_found.append(f"cara too short ({len(cara)} chars)")
                stats['issues']['cara_too_short'] += 1

            if not jawaban or len(jawaban) < 1:
                issues_found.append("jawaban empty")
                stats['issues']['jawaban_empty'] += 1

            # Check 2: soal-cara alignment
            if soal and cara:
                aligned, reason = check_soal_cara_alignment(soal, cara)
                if not aligned:
                    issues_found.append(reason)
                    stats['issues']['low_soal_cara_overlap'] += 1

            # Check 3: answer format consistency
            if cara and jawaban:
                consistent, reasons = check_answer_format_consistency(cara, jawaban)
                if not consistent:
                    for r in reasons:
                        issues_found.append(r)
                        stats['issues']['answer_format_inconsistency'] += 1

            # Flag or mark valid
            if issues_found:
                stats['flagged'] += 1
                flagged_records.append({
                    'line': line_no,
                    'soal': soal[:60] + '...' if len(soal) > 60 else soal,
                    'cara': cara[:60] + '...' if len(cara) > 60 else cara,
                    'jawaban': jawaban,
                    'reasons': issues_found
                })
            else:
                stats['valid'] += 1

    # Write flagged records
    output_flagged_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_flagged_path, 'w', encoding='utf-8') as f:
        for rec in flagged_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # Write report
    report = f"""TRAINING DATA VALIDATION REPORT
================================

Input: {input_path}
Total records: {stats['total']}
Valid records: {stats['valid']} ({100*stats['valid']/stats['total']:.1f}%)
Flagged records: {stats['flagged']} ({100*stats['flagged']/stats['total']:.1f}%)

ISSUES BREAKDOWN
================
"""
    for issue_type, count in sorted(stats['issues'].items(), key=lambda x: -x[1]):
        report += f"{issue_type:40s} {count:5d}\n"

    report += f"""
Flagged records saved to: {output_flagged_path}
Total flagged: {len(flagged_records)}

Recommendation:
- Manually review flagged records for correctness
- Consider filtering out records with persistent format issues before training
"""

    with open(output_report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"Flagged records: {output_flagged_path}")
    print(f"Report: {output_report_path}")

if __name__ == '__main__':
    input_file = Path('data/Final/easy_clean.jsonl')
    output_flagged = Path('data/Final/easy_clean_flagged.jsonl')
    output_report = Path('data/Final/validation_report.txt')

    if not input_file.exists():
        print(f"Error: {input_file} not found")
        sys.exit(1)

    validate_jsonl(input_file, output_flagged, output_report)
