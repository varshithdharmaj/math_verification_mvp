"""Visualizations for XAI explanations."""

import streamlit as st
import pandas as pd
from typing import Dict, List

# Try to import plotly, make it optional
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not available. Install with: pip install plotly")


def plot_confidence_breakdown(explanation: Dict):
    """Plot confidence breakdown for a verifier."""
    if not PLOTLY_AVAILABLE:
        # Fallback to simple text display
        confidence_factors = explanation.get('confidence_factors', {})
        if confidence_factors:
            st.write("**Confidence Factors:**")
            for factor, value in confidence_factors.items():
                st.write(f"- {factor}: {value:.3f}")
        return
    
    confidence_factors = explanation.get('confidence_factors', {})
    
    if not confidence_factors:
        return
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(confidence_factors.keys()),
            y=list(confidence_factors.values()),
            marker_color='lightblue',
            text=[f"{v:.2f}" for v in confidence_factors.values()],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Confidence Factors",
        xaxis_title="Factor",
        yaxis_title="Confidence Score",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"confidence_breakdown_{hash(str(explanation))}")


def plot_class_probabilities(explanation: Dict):
    """Plot class probabilities for ML classifier."""
    class_probs = explanation.get('class_probabilities', {})
    
    if not class_probs:
        return
    
    if not PLOTLY_AVAILABLE:
        # Fallback to table
        st.write("**Class Probabilities:**")
        df = pd.DataFrame([
            {'Class': k, 'Probability': f"{v:.3f}"} 
            for k, v in sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
        ])
        st.dataframe(df, use_container_width=True)
        return
    
    df = pd.DataFrame([
        {'Class': k, 'Probability': v} 
        for k, v in class_probs.items()
    ])
    df = df.sort_values('Probability', ascending=False)
    
    fig = px.bar(
        df,
        x='Class',
        y='Probability',
        color='Probability',
        color_continuous_scale='RdYlGn',
        title="Class Probabilities"
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True, key=f"class_probabilities_{hash(str(explanation))}")


def plot_verifier_contributions(consensus_explanation: Dict):
    """Plot weighted contributions of each verifier."""
    weighted_analysis = consensus_explanation.get('weighted_analysis', {})
    
    if not weighted_analysis:
        return
    
    if not PLOTLY_AVAILABLE:
        # Fallback to table
        st.write("**Verifier Contributions:**")
        df = pd.DataFrame([
            {
                'Verifier': name,
                'Contribution': f"{info['contribution']:.3f}",
                'Verdict': info['verdict'],
                'Impact': info['impact']
            }
            for name, info in weighted_analysis.items()
        ])
        st.dataframe(df, use_container_width=True)
        return
    
    verifiers = list(weighted_analysis.keys())
    contributions = [weighted_analysis[v]['contribution'] for v in verifiers]
    colors = ['red' if c > 0 else 'green' for c in contributions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=verifiers,
            y=contributions,
            marker_color=colors,
            text=[f"{c:.3f}" for c in contributions],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Verifier Contributions to Consensus",
        xaxis_title="Verifier",
        yaxis_title="Contribution",
        yaxis=dict(range=[-0.5, 0.5]),
        height=350
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig, use_container_width=True, key=f"verifier_contributions_{hash(str(consensus_explanation))}")


def plot_agreement_visualization(consensus: Dict):
    """Visualize agreement between verifiers."""
    verdict_counts = consensus.get('verdict_counts', {})
    agreement_type = consensus.get('agreement_type', 'MIXED')
    
    if not verdict_counts:
        return
    
    if not PLOTLY_AVAILABLE:
        # Fallback to text
        st.write(f"**Agreement Type:** {agreement_type}")
        st.write(f"- ERROR votes: {verdict_counts.get('error', 0)}")
        st.write(f"- VALID votes: {verdict_counts.get('valid', 0)}")
        return
    
    labels = ['ERROR', 'VALID']
    values = [
        verdict_counts.get('error', 0),
        verdict_counts.get('valid', 0)
    ]
    colors = ['#FF6B6B', '#51CF66']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title=f"Verifier Agreement ({agreement_type})",
        height=300
    )
    
    step_idx = consensus.get('step_idx', 0)
    st.plotly_chart(fig, use_container_width=True, key=f"agreement_viz_{step_idx}_{hash(str(consensus))}")


def plot_reasoning_chain(explanation: Dict):
    """Visualize the reasoning chain."""
    reasoning = explanation.get('reasoning', [])
    key_factors = explanation.get('key_factors', [])
    
    if not reasoning and not key_factors:
        return
    
    st.markdown("### Reasoning Chain")
    
    for i, reason in enumerate(reasoning, 1):
        st.markdown(f"**{i}.** {reason}")
    
    if key_factors:
        st.markdown("### Key Factors")
        for factor in key_factors:
            st.markdown(f"• {factor}")


def create_explanation_dashboard(
    verifier_explanations: Dict[str, Dict],
    consensus_explanation: Dict,
    step_idx: int = 0
):
    """Create a comprehensive explanation dashboard.
    
    Args:
        verifier_explanations: Explanations from each verifier
        consensus_explanation: Consensus explanation
        step_idx: Step index for unique chart keys
    """
    st.header("🔍 Explainable AI Dashboard")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "🔬 Per-Verifier", 
        "⚖️ Consensus", 
        "📈 Visualizations"
    ])
    
    with tab1:
        st.subheader("Consensus Summary")
        st.write(f"**Final Verdict:** {consensus_explanation.get('final_verdict', 'UNKNOWN')}")
        st.write(f"**Agreement Type:** {consensus_explanation.get('agreement_type', 'MIXED')}")
        
        st.subheader("Reasoning")
        for reason in consensus_explanation.get('reasoning', []):
            st.write(f"• {reason}")
        
        st.subheader("Contributing Factors")
        for factor in consensus_explanation.get('contributing_factors', []):
            st.write(f"• {factor}")
    
    with tab2:
        st.subheader("Individual Verifier Explanations")
        
        for verifier_name, explanation in verifier_explanations.items():
            with st.expander(f"🔍 {verifier_name.upper()}"):
                st.write(f"**Verdict:** {explanation.get('verdict', 'UNKNOWN')}")
                
                st.write("**Reasoning:**")
                for reason in explanation.get('reasoning', []):
                    st.write(f"• {reason}")
                
                if explanation.get('evidence'):
                    st.write("**Evidence:**")
                    for evidence in explanation.get('evidence', []):
                        st.write(f"  - {evidence}")
                
                if explanation.get('confidence_factors'):
                    st.write("**Confidence Factors:**")
                    for factor, value in explanation.get('confidence_factors', {}).items():
                        st.write(f"  - {factor}: {value:.3f}")
    
    with tab3:
        st.subheader("Consensus Analysis")
        
        agreement_data = {
            'verdict_counts': {
                'error': sum(1 for v in verifier_explanations.values() if v.get('verdict') == 'ERROR'),
                'valid': sum(1 for v in verifier_explanations.values() if v.get('verdict') == 'VALID')
            },
            'agreement_type': consensus_explanation.get('agreement_type', 'MIXED'),
            'step_idx': step_idx
        }
        plot_agreement_visualization(agreement_data)
        
        plot_verifier_contributions(consensus_explanation)
        
        st.subheader("Weighted Analysis")
        weighted_analysis = consensus_explanation.get('weighted_analysis', {})
        if weighted_analysis:
            df = pd.DataFrame([
                {
                    'Verifier': name,
                    'Weight': info['weight'],
                    'Contribution': info['contribution'],
                    'Verdict': info['verdict'],
                    'Impact': info['impact']
                }
                for name, info in weighted_analysis.items()
            ])
            st.dataframe(df, use_container_width=True)
    
    with tab4:
        st.subheader("Visualizations")
        
        # ML Classifier probabilities
        if 'ml_classifier' in verifier_explanations:
            ml_explanation = verifier_explanations['ml_classifier']
            if ml_explanation.get('class_probabilities'):
                plot_class_probabilities(ml_explanation)
        
        # Confidence breakdowns
        col1, col2 = st.columns(2)
        with col1:
            if 'symbolic' in verifier_explanations:
                plot_confidence_breakdown(verifier_explanations['symbolic'])
        with col2:
            if 'llm_logical' in verifier_explanations:
                plot_confidence_breakdown(verifier_explanations['llm_logical'])

