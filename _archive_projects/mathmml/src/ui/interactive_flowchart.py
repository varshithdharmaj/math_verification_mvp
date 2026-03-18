"""Interactive flowchart component for Streamlit."""

import streamlit as st
from typing import Dict, List, Optional
import json


def render_interactive_flowchart(verification_state: Dict):
    """Render an interactive flowchart showing the verification pipeline.
    
    Args:
        verification_state: Dict with pipeline state including:
            - steps: List of step results
            - current_step: Current step being processed
            - verifier_results: Dict of verifier results
            - consensus: Consensus result
    """
    st.markdown("### 🔄 Interactive Pipeline Flowchart")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Flow View", "🔍 Step Details", "📈 Model Comparison"])
    
    with tab1:
        render_flow_view(verification_state)
    
    with tab2:
        render_step_details(verification_state)
    
    with tab3:
        render_model_comparison(verification_state)


def render_flow_view(state: Dict):
    """Render the main flow view."""
    # Use columns to create a visual flow
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    # Step 1: Input
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 10px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #E8F5E9;'>
            <h4>📥 INPUT</h4>
            <p style='font-size: 12px;'>Problem + Steps</p>
        </div>
        """, unsafe_allow_html=True)
        if state.get('problem'):
            with st.expander("View Input"):
                st.write(f"**Problem:** {state['problem'][:100]}...")
                st.write(f"**Steps:** {len(state.get('steps', []))} steps")
    
    # Arrow
    st.markdown("<div style='text-align: center; font-size: 24px;'>⬇️</div>", unsafe_allow_html=True)
    
    # Step 2: Parsing
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 10px; border: 2px solid #2196F3; border-radius: 10px; background-color: #E3F2FD;'>
            <h4>🔍 PARSING</h4>
            <p style='font-size: 12px;'>Extract expressions</p>
        </div>
        """, unsafe_allow_html=True)
        if state.get('parsed_expressions'):
            with st.expander("View Parsed"):
                for expr in state['parsed_expressions'][:3]:
                    st.code(expr)
    
    # Arrow
    st.markdown("<div style='text-align: center; font-size: 24px;'>⬇️</div>", unsafe_allow_html=True)
    
    # Step 3: Parallel Models
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 10px; border: 2px solid #FF9800; border-radius: 10px; background-color: #FFF3E0;'>
            <h4>⚡ PARALLEL</h4>
            <p style='font-size: 12px;'>4 Models Running</p>
        </div>
        """, unsafe_allow_html=True)
        
        verifier_results = state.get('verifier_results', {})
        if verifier_results:
            with st.expander("View Models"):
                for name, result in verifier_results.items():
                    verdict = result.get('verdict', 'UNKNOWN')
                    color = "🟢" if verdict == 'VALID' else "🔴" if verdict == 'ERROR' else "🟡"
                    st.write(f"{color} **{name}**: {verdict}")
    
    # Arrow
    st.markdown("<div style='text-align: center; font-size: 24px;'>⬇️</div>", unsafe_allow_html=True)
    
    # Step 4: Consensus
    with col4:
        consensus = state.get('consensus', {})
        verdict = consensus.get('final_verdict', 'UNKNOWN')
        border_color = "#4CAF50" if verdict == 'VALID' else "#F44336" if verdict == 'ERROR' else "#9E9E9E"
        bg_color = "#E8F5E9" if verdict == 'VALID' else "#FFEBEE" if verdict == 'ERROR' else "#F5F5F5"
        
        st.markdown(f"""
        <div style='text-align: center; padding: 10px; border: 2px solid {border_color}; border-radius: 10px; background-color: {bg_color};'>
            <h4>⚖️ CONSENSUS</h4>
            <p style='font-size: 12px;'><strong>{verdict}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        if consensus:
            with st.expander("View Consensus"):
                st.write(f"**Confidence:** {consensus.get('overall_confidence', 0):.3f}")
                st.write(f"**Agreement:** {consensus.get('agreement_type', 'N/A')}")
                st.write(f"**Error Score:** {consensus.get('error_score', 0):.3f}")
    
    # Arrow
    st.markdown("<div style='text-align: center; font-size: 24px;'>⬇️</div>", unsafe_allow_html=True)
    
    # Step 5: Output
    with col5:
        st.markdown(f"""
        <div style='text-align: center; padding: 10px; border: 2px solid {border_color}; border-radius: 10px; background-color: {bg_color};'>
            <h4>📤 OUTPUT</h4>
            <p style='font-size: 12px;'><strong>{verdict}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        if consensus:
            with st.expander("View Output"):
                st.json(consensus)


def render_step_details(state: Dict):
    """Render detailed step-by-step view."""
    steps = state.get('steps', [])
    current_step_idx = state.get('current_step_idx', 0)
    
    if not steps:
        st.info("No steps processed yet. Run a verification to see details.")
        return
    
    for idx, step in enumerate(steps):
        is_current = idx == current_step_idx
        border_style = "3px solid #FF9800" if is_current else "1px solid #ccc"
        
        st.markdown(f"""
        <div style='padding: 15px; margin: 10px 0; border: {border_style}; border-radius: 8px; background-color: {'#FFF3E0' if is_current else '#FAFAFA'};'>
            <h4>Step {idx + 1}{' (Current)' if is_current else ''}</h4>
            <p><strong>Step Text:</strong> {step.get('step_text', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show verifier results for this step
        step_results = step.get('verifier_results', {})
        if step_results:
            cols = st.columns(4)
            for i, (name, result) in enumerate(step_results.items()):
                with cols[i % 4]:
                    verdict = result.get('verdict', 'UNKNOWN')
                    confidence = result.get('confidence', 0)
                    color = "green" if verdict == 'VALID' else "red" if verdict == 'ERROR' else "gray"
                    st.markdown(f"""
                    <div style='padding: 10px; border: 2px solid {color}; border-radius: 5px; text-align: center;'>
                        <strong>{name}</strong><br>
                        {verdict}<br>
                        <small>{confidence:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)


def render_model_comparison(state: Dict):
    """Render model comparison view."""
    verifier_results = state.get('verifier_results', {})
    consensus = state.get('consensus', {})
    
    if not verifier_results:
        st.info("No verification results yet.")
        return
    
    # Create comparison table
    import pandas as pd
    
    data = []
    for name, result in verifier_results.items():
        data.append({
            'Model': name,
            'Verdict': result.get('verdict', 'UNKNOWN'),
            'Confidence': f"{result.get('confidence', 0):.3f}",
            'Error Type': result.get('error_type', 'N/A'),
            'Weight': f"{consensus.get('breakdown', {}).get(name, {}).get('weight', 0):.2f}"
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
    
    # Visual comparison
    st.markdown("### Confidence Comparison")
    st.bar_chart(df.set_index('Model')['Confidence'].apply(lambda x: float(x)))
    
    # Agreement visualization
    if consensus:
        st.markdown("### Agreement Analysis")
        agreement = consensus.get('agreement_type', 'MIXED')
        verdict_counts = consensus.get('verdict_counts', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Agreement Type", agreement)
        with col2:
            st.metric("Error Votes", verdict_counts.get('error', 0))
        with col3:
            st.metric("Valid Votes", verdict_counts.get('valid', 0))

