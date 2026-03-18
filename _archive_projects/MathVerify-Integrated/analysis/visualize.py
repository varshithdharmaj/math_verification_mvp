"""
Visualization and analysis tools for MathVerify-Integrated.

Generates charts and reports for error analysis and pipeline performance.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Dict, List, Any
import json

# Use non-interactive backend for server environments
matplotlib.use('Agg')

# Professional styling
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']  # Color-blind friendly


def plot_error_distribution(errors: Dict[str, Any], save_path: str = "error_distribution.png"):
    """
    Plot bar chart of error types with counts and percentages.
    
    Args:
        errors: Error dictionary with 'by_type' and 'percentage' keys
        save_path: Path to save the chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    by_type = errors.get('by_type', {})
    percentage = errors.get('percentage', {})
    
    if not by_type:
        ax.text(0.5, 0.5, 'No errors to display', 
                ha='center', va='center', fontsize=16)
        ax.set_title('Error Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    error_types = list(by_type.keys())
    counts = list(by_type.values())
    percentages = [percentage.get(et, 0) for et in error_types]
    
    # Create bar chart
    bars = ax.bar(error_types, counts, color=COLORS[:len(error_types)])
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(counts[i])}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Error Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Error Distribution by Type', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Error distribution chart saved to {save_path}")


def plot_verification_pipeline(results: List[Dict], save_path: str = "pipeline_performance.png"):
    """
    Plot pipeline stage success rates.
    
    Shows: Input → Reasoning → Verification pipeline performance.
    
    Args:
        results: List of result dictionaries from pipeline
        save_path: Path to save the chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if not results:
        ax.text(0.5, 0.5, 'No results to display', 
                ha='center', va='center', fontsize=16)
        ax.set_title('Pipeline Performance', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Calculate success rates for each stage
    total = len(results)
    
    input_valid = sum(1 for r in results if r.get('solution', {}).get('steps'))
    reasoning_success = sum(1 for r in results if r.get('final_answer') and 'error' not in r.get('final_answer', '').lower())
    verification_success = sum(1 for r in results if r.get('errors', {}).get('total_errors', 0) == 0)
    
    stages = ['Input\nProcessing', 'Reasoning\nGeneration', 'Verification\nPass']
    success_rates = [
        (input_valid / total * 100) if total > 0 else 0,
        (reasoning_success / total * 100) if total > 0 else 0,
        (verification_success / total * 100) if total > 0 else 0
    ]
    
    # Create line chart
    ax.plot(stages, success_rates, marker='o', linewidth=2, markersize=10, 
            color=COLORS[0], label='Success Rate')
    
    # Add value labels
    for i, rate in enumerate(success_rates):
        ax.text(i, rate, f'{rate:.1f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Pipeline Performance by Stage', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Pipeline performance chart saved to {save_path}")


def create_comparison_table(
    baseline: Dict[str, Any],
    with_verification: Dict[str, Any],
    save_path: str = "comparison_table.md"
) -> str:
    """
    Create markdown table comparing baseline vs verification results.
    
    Args:
        baseline: Baseline results dictionary
        with_verification: Results with verification dictionary
        save_path: Path to save the table
        
    Returns:
        Markdown table string
    """
    baseline_acc = baseline.get('accuracy', 0)
    verify_acc = with_verification.get('accuracy', 0)
    improvement = verify_acc - baseline_acc
    
    baseline_errors = baseline.get('total_errors', 0)
    verify_errors = with_verification.get('total_errors', 0)
    error_reduction = baseline_errors - verify_errors
    
    table = f"""# Comparison: Baseline vs With Verification

## Accuracy Comparison

| Metric | Baseline | With Verification | Improvement |
|--------|----------|-------------------|-------------|
| Accuracy | {baseline_acc:.2f}% | {verify_acc:.2f}% | {improvement:+.2f}% |
| Total Errors | {baseline_errors} | {verify_errors} | {error_reduction:+d} |

## Error Breakdown

### Baseline Errors
"""
    
    baseline_by_type = baseline.get('errors_by_type', {})
    for error_type, count in baseline_by_type.items():
        table += f"- {error_type}: {count}\n"
    
    table += "\n### With Verification Errors\n"
    verify_by_type = with_verification.get('errors_by_type', {})
    for error_type, count in verify_by_type.items():
        table += f"- {error_type}: {count}\n"
    
    table += f"\n## Summary\n\n"
    table += f"- Accuracy improved by {improvement:.2f} percentage points\n"
    table += f"- Total errors reduced by {error_reduction}\n"
    
    if improvement > 0:
        table += f"- ✅ Verification system shows {improvement:.1f}% improvement\n"
    else:
        table += f"- ⚠️ Verification system shows {abs(improvement):.1f}% decrease\n"
    
    # Save to file
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(table)
    
    print(f"Comparison table saved to {save_path}")
    return table


def generate_report(all_results: List[Dict], save_path: str = "results_report.md"):
    """
    Generate comprehensive analysis report.
    
    Args:
        all_results: List of all result dictionaries
        save_path: Path to save the report
    """
    if not all_results:
        report = "# Results Report\n\nNo results to analyze.\n"
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        return
    
    total_problems = len(all_results)
    total_errors = sum(r.get('errors', {}).get('total_errors', 0) for r in all_results)
    avg_confidence = np.mean([r.get('confidence', 0) for r in all_results])
    
    # Error breakdown
    error_counts = {}
    for result in all_results:
        by_type = result.get('errors', {}).get('by_type', {})
        for error_type, count in by_type.items():
            error_counts[error_type] = error_counts.get(error_type, 0) + count
    
    report = f"""# MathVerify-Integrated: Comprehensive Results Report

## Overview

- **Total Problems Processed**: {total_problems}
- **Total Errors Detected**: {total_errors}
- **Average Confidence Score**: {avg_confidence:.2f}
- **Error Rate**: {(total_errors / total_problems * 100):.2f}%

## Error Analysis

### Error Distribution

"""
    
    for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_errors * 100) if total_errors > 0 else 0
        report += f"- **{error_type}**: {count} ({percentage:.1f}%)\n"
    
    report += f"""
## Performance Metrics

### Confidence Distribution

"""
    
    confidences = [r.get('confidence', 0) for r in all_results]
    if confidences:
        report += f"- **Mean**: {np.mean(confidences):.2f}\n"
        report += f"- **Median**: {np.median(confidences):.2f}\n"
        report += f"- **Std Dev**: {np.std(confidences):.2f}\n"
        report += f"- **Min**: {np.min(confidences):.2f}\n"
        report += f"- **Max**: {np.max(confidences):.2f}\n"
    
    report += f"""
## Sample Results

"""
    
    # Show first 5 results
    for i, result in enumerate(all_results[:5], 1):
        report += f"### Problem {i}\n\n"
        report += f"**Problem**: {result.get('problem', 'N/A')[:100]}...\n\n"
        report += f"**Final Answer**: {result.get('final_answer', 'N/A')}\n\n"
        report += f"**Confidence**: {result.get('confidence', 0):.2f}\n\n"
        errors = result.get('errors', {})
        report += f"**Errors**: {errors.get('total_errors', 0)}\n\n"
        report += "---\n\n"
    
    # Save report
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Comprehensive report saved to {save_path}")

