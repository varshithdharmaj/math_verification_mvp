"""Enhanced consensus breakdown display matching PRD requirements."""

import streamlit as st
from typing import Dict


def display_consensus_breakdown(consensus: Dict, weights: Dict[str, float]):
    """Display consensus breakdown in PRD-specified format.
    
    PRD Req 4.4.6: Show weighted calculation breakdown.
    
    Args:
        consensus: Consensus result dict
        weights: Model weights dict
    """
    breakdown = consensus.get('breakdown', {})
    
    st.markdown("#### Consensus Calculation Breakdown")
    st.write("Weighted error score calculation:")
    
    # Create calculation display
    calculation_lines = []
    total_error_score = 0
    
    for name, info in breakdown.items():
        weight = info.get('weight', 0)
        confidence = info.get('confidence', 0)
        verdict = info.get('verdict', 'UNKNOWN')
        contribution = info.get('contribution', 0)
        
        if verdict == 'ERROR':
            contribution_display = f"{weight:.2f} × {confidence:.2f} = {contribution:.3f}"
            total_error_score += contribution
        else:
            contribution_display = f"-{weight:.2f} × {confidence:.2f} = {contribution:.3f}"
            total_error_score += contribution  # Already negative
        
        calculation_lines.append(f"  {name.capitalize()} ({weight*100:.0f}%): {contribution_display}")
    
    # Display calculation
    st.code("\n".join(calculation_lines))
    
    st.write(f"**Total Error Score:** {total_error_score:.3f}")
    
    # Show threshold comparison
    threshold = 0.0  # Updated threshold
    st.write(f"**Threshold:** {threshold:.2f}")
    
    final_verdict = consensus.get('final_verdict', 'UNKNOWN')
    if total_error_score > threshold:
        st.write(f"**Decision:** {total_error_score:.3f} > {threshold:.2f} → **ERROR**")
    else:
        st.write(f"**Decision:** {total_error_score:.3f} ≤ {threshold:.2f} → **VALID**")
    
    # Show agreement type impact
    agreement = consensus.get('agreement_type', 'MIXED')
    st.write(f"**Agreement Type:** {agreement}")
    
    if agreement == 'UNANIMOUS':
        st.info("All models agreed → Confidence boosted by 10%")
    elif agreement == 'MAJORITY':
        st.info("Majority of models agreed → Using average of agreeing models")
    else:
        st.warning("Mixed opinions → Confidence reduced by 20%")

