"""Error location tracking and display for UI."""

import streamlit as st
from typing import Dict, List
from src.utils.error_location import ErrorLocationTracker


def display_error_location(
    steps: List[str],
    step_results: List[Dict],
    tracker: ErrorLocationTracker
) -> None:
    """Display error location tracking in UI.
    
    PRD Req 4.2.3: Error location tracking with step ranges.
    
    Args:
        steps: All steps in the solution
        step_results: Results for each step
        tracker: ErrorLocationTracker instance
    """
    st.markdown("### 📍 Error Location Analysis")
    
    # Get error summary
    summary = tracker.get_error_summary()
    
    if summary['total_errors'] == 0:
        st.success("✅ No errors detected in any step!")
        return
    
    # Show affected steps
    st.write(f"**Total Errors Found:** {summary['total_errors']}")
    st.write(f"**Affected Steps:** {', '.join([f'Step {idx+1}' for idx in summary['affected_steps']])}")
    
    # Show error ranges (PRD Req 4.2.3)
    error_ranges = summary.get('error_ranges', [])
    if error_ranges:
        st.write("**Error Ranges:**")
        for range_info in error_ranges:
            start = range_info['start'] + 1  # 1-indexed for display
            end = range_info['end'] + 1
            
            if start == end:
                st.write(f"  - Step {start}: {len(range_info['errors'])} error(s)")
            else:
                st.write(f"  - Steps {start}-{end}: {len(range_info['errors'])} error(s)")
    
    # Root cause analysis
    root_cause = tracker.trace_root_cause(steps)
    if root_cause:
        st.markdown("#### 🔍 Root Cause Analysis")
        st.write(f"**Root Cause Step:** Step {root_cause['root_cause_step'] + 1}")
        st.write(f"**Error Type:** {root_cause['root_cause_type']}")
        if root_cause['is_cascading']:
            st.warning("⚠️ **Cascading Error:** This error originated in an earlier step and affected subsequent steps.")
        st.write(f"**Explanation:** {root_cause['explanation']}")
    
    # Error type distribution
    if summary['error_types']:
        st.markdown("#### 📊 Error Type Distribution")
        for error_type, count in sorted(summary['error_types'].items(), key=lambda x: x[1], reverse=True):
            st.write(f"  - `{error_type}`: {count} occurrence(s)")

