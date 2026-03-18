"""Streamlit dashboard with flowchart and live logs."""

import streamlit as st
import time
from typing import List, Dict
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.symbolic_verifier import SymbolicVerifier
from src.models.llm_logical_checker import LLMLogicalChecker
from src.models.ensemble_checker import EnsembleNeuralChecker
from src.models.ml_step_classifier import MLStepClassifierWrapper
from src.pipeline.consensus import ConsensusEngine
from src.pipeline.orchestrator import VerificationOrchestrator
from src.utils.logging_utils import SessionLogger, LogLevel
from src.utils.explanation import generate_explanation
from src.utils.correction import CorrectionEngine
from src.utils.llm_providers import get_available_providers
from src.ui.interactive_flowchart import render_interactive_flowchart
from src.xai.explainer import XAIExplainer
from src.xai.visualizations import create_explanation_dashboard
from src.utils.error_location import ErrorLocationTracker
from src.ui.error_location_display import display_error_location
from src.ui.consensus_breakdown_display import display_consensus_breakdown


def render_flowchart():
    """Render simple ASCII flowchart (fallback)."""
    st.markdown("""
    ```
    ┌─────────┐
    │  INPUT  │  Problem + Steps
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ PARSING │  Extract expressions, context
    └────┬────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │      PARALLEL MODELS               │
    │  ┌──────────┐  ┌──────────┐       │
    │  │Symbolic  │  │LLM Logic │       │
    │  │Verifier  │  │ Checker  │       │
    │  └──────────┘  └──────────┘       │
    │  ┌──────────┐  ┌──────────┐       │
    │  │Ensemble  │  │ML Class. │       │
    │  │Checker   │  │          │       │
    │  └──────────┘  └──────────┘       │
    └─────────────┬──────────────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │    CONSENSUS    │  Weighted voting
         └────────┬────────┘
                  │
                  ▼
            ┌─────────┐
            │ OUTPUT  │  Verdict + Confidence
            └─────────┘
    ```
    """)


def render_model_card(name: str, result: Dict, weight: float):
    """Render a model card.
    
    Args:
        name: Model name
        result: Result dict
        weight: Model weight
    """
    verdict = result.get('verdict', 'UNKNOWN')
    confidence = result.get('confidence', 0.0)
    error_type = result.get('error_type')
    
    color = "🟢" if verdict == 'VALID' else "🔴" if verdict == 'ERROR' else "🟡"
    
    st.markdown(f"### {color} {name} (weight: {weight:.2f})")
    st.write(f"**Verdict:** {verdict}")
    st.write(f"**Confidence:** {confidence:.3f}")
    if error_type:
        st.write(f"**Error Type:** {error_type}")
    if result.get('details'):
        with st.expander("Details"):
            st.write(result['details'])


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Math Verification System",
        page_icon="🔢",
        layout="wide"
    )
    
    st.title("🔢 Math Verification System")
    st.markdown("Four-model verification pipeline with weighted consensus")
    
    # Initialize session state
    if 'logger' not in st.session_state:
        st.session_state.logger = SessionLogger()
    if 'verifiers' not in st.session_state:
        st.session_state.verifiers = {}
        st.session_state.consensus_engine = ConsensusEngine()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LLM Logical Checker settings
        st.subheader("LLM Logical Checker")
        use_llm_api = st.checkbox("Use LLM API", value=False)
        if use_llm_api:
            available_providers = get_available_providers()
            if not available_providers:
                st.warning("⚠️ No LLM providers available. Set API keys or install Ollama.")
                available_providers = ["openai", "gemini", "llama", "anthropic"]  # Show all for selection
            
            llm_provider = st.selectbox(
                "Provider",
                options=["openai", "gemini", "llama", "anthropic"],
                index=0 if "openai" in available_providers else 1
            )
            
            # Model selection based on provider
            if llm_provider == "openai":
                llm_model = st.selectbox(
                    "Model",
                    options=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                    index=2
                )
            elif llm_provider == "gemini":
                llm_model = st.selectbox(
                    "Model",
                    options=["gemini-pro", "gemini-pro-vision"],
                    index=0
                )
            elif llm_provider == "llama":
                llm_model = st.text_input(
                    "Model Name",
                    value="llama2",
                    help="Ollama model name (e.g., llama2, mistral, codellama)"
                )
            else:  # anthropic
                llm_model = st.selectbox(
                    "Model",
                    options=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
                    index=1
                )
        else:
            llm_provider = "openai"
            llm_model = None
        
        # Ensemble Checker settings
        st.subheader("Ensemble Checker")
        use_ensemble_api = st.checkbox("Use Ensemble API", value=False)
        if use_ensemble_api:
            num_ensemble_models = st.slider("Number of Models", 1, 5, 3)
            
            st.write("Configure models:")
            ensemble_configs = []
            for i in range(num_ensemble_models):
                col1, col2 = st.columns(2)
                with col1:
                    provider = st.selectbox(
                        f"Provider {i+1}",
                        options=["openai", "gemini", "llama", "anthropic"],
                        key=f"ens_provider_{i}"
                    )
                with col2:
                    if provider == "openai":
                        model = st.selectbox(
                            f"Model {i+1}",
                            options=["gpt-4", "gpt-3.5-turbo"],
                            key=f"ens_model_{i}"
                        )
                    elif provider == "gemini":
                        model = st.selectbox(
                            f"Model {i+1}",
                            options=["gemini-pro"],
                            key=f"ens_model_{i}"
                        )
                    elif provider == "llama":
                        model = st.text_input(
                            f"Model {i+1}",
                            value="llama2",
                            key=f"ens_model_{i}"
                        )
                    else:
                        model = st.selectbox(
                            f"Model {i+1}",
                            options=["claude-3-sonnet-20240229"],
                            key=f"ens_model_{i}"
                        )
                ensemble_configs.append({"provider": provider, "model": model})
        else:
            num_ensemble_models = 3
            ensemble_configs = None
        
        model_path = st.text_input(
            "ML Classifier Model Path",
            value="models/checkpoints/",
            help="Path to trained classifier model"
        )
        
        st.header("📊 Session Stats")
        stats = st.session_state.logger.get_stats()
        st.write(f"Total Verifications: {stats['verifications']}")
        st.write(f"Valid Steps: {stats['valid_steps']}")
        st.write(f"Error Steps: {stats['error_steps']}")
        
        if st.button("Clear Logs"):
            st.session_state.logger.clear()
            st.rerun()
    
    # Main layout: Left input, Right results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        problem = st.text_area(
            "Problem Statement",
            height=100,
            placeholder="Enter the math problem here..."
        )
        
        steps_text = st.text_area(
            "Solution Steps (one per line)",
            height=200,
            placeholder="Step 1: ...\nStep 2: ...\n..."
        )
        
        if st.button("🔍 Verify", type="primary"):
            if not problem or not steps_text:
                st.error("Please provide both problem and steps")
            else:
                # Parse steps
                steps = [s.strip() for s in steps_text.split('\n') if s.strip()]
                
                # Initialize verifiers if needed (recreate if settings changed)
                st.session_state.verifiers = {
                    'symbolic': SymbolicVerifier(),
                    'llm_logical': LLMLogicalChecker(
                        use_api=use_llm_api,
                        api_provider=llm_provider if use_llm_api else "openai",
                        model=llm_model if use_llm_api else None
                    ),
                    'ensemble': EnsembleNeuralChecker(
                        use_apis=use_ensemble_api,
                        num_models=num_ensemble_models,
                        model_configs=ensemble_configs if use_ensemble_api else None
                    ),
                    'ml_classifier': MLStepClassifierWrapper(model_path=model_path)
                }
                
                # Initialize error location tracker (PRD Req 4.2.3)
                error_tracker = ErrorLocationTracker()
                
                # Verify each step
                results_container = st.container()
                
                with results_container:
                    st.header("📊 Results")
                    
                    step_results_list = []  # Store results for all steps
                    
                    for step_idx, step in enumerate(steps):
                        st.subheader(f"Step {step_idx + 1}")
                        st.write(f"**Step:** {step}")
                        
                        prev_steps = steps[:step_idx]
                        
                        # Log verification start
                        st.session_state.logger.log_verification_start(step, problem)
                        
                        # ============================================
                        # STEP 1: Show Individual Model Outputs
                        # ============================================
                        st.markdown("---")
                        st.markdown("### 📊 Individual Model Outputs")
                        st.write("Each model analyzes the step independently:")
                        
                        # Run verifiers in parallel using orchestrator
                        orchestrator = VerificationOrchestrator()
                        consensus_result = orchestrator.verify_step(
                            step, problem, prev_steps, st.session_state.verifiers
                        )
                        verifier_results = consensus_result['per_verifier_results']
                        
                        # Display results in columns
                        model_cols = st.columns(2)
                        
                        for idx, (name, result) in enumerate(verifier_results.items()):
                            with model_cols[idx % 2]:
                                with st.container():
                                    st.session_state.logger.log_model_result(
                                        name, result['verdict'], result['confidence']
                                    )
                                    
                                    # Display result immediately
                                    verdict = result.get('verdict', 'UNKNOWN')
                                    confidence = result.get('confidence', 0.0)
                                    error_type = result.get('error_type')
                                    
                                    # Color-coded display
                                    if verdict == 'VALID':
                                        st.success(f"✅ **{name.upper()}**")
                                        st.write(f"Verdict: **VALID**")
                                        st.write(f"Confidence: {confidence:.3f}")
                                    elif verdict == 'ERROR':
                                        st.error(f"❌ **{name.upper()}**")
                                        st.write(f"Verdict: **ERROR**")
                                        st.write(f"Confidence: {confidence:.3f}")
                                        if error_type:
                                            st.write(f"Error Type: `{error_type}`")
                                    else:
                                        st.warning(f"⚠️ **{name.upper()}**")
                                        st.write(f"Verdict: **{verdict}**")
                                        st.write(f"Confidence: {confidence:.3f}")
                                    
                                    # Show details in expander
                                    with st.expander(f"Details ({name})"):
                                        st.write(f"**Details:** {result.get('details', 'N/A')}")
                                        if result.get('error_type'):
                                            st.write(f"**Error Type:** {result.get('error_type')}")
                        
                        # ============================================
                        # STEP 2: Show Final Consensus
                        # ============================================
                        st.markdown("---")
                        st.markdown("### ⚖️ Final Consensus")
                        
                        # Use consensus from orchestrator (already computed)
                        consensus = consensus_result
                        st.session_state.logger.log_consensus(consensus)
                        
                        # Update verification state for interactive flowchart
                        parsed_exprs = []
                        try:
                            temp_verifier = SymbolicVerifier()
                            extracted = temp_verifier.extract_expression(step)
                            if extracted:
                                parsed_exprs.append(str(extracted[0]))
                        except Exception as e:
                            pass
                        
                        st.session_state.last_verification = {
                            'problem': problem,
                            'steps': [{'step_text': s, 'verifier_results': verifier_results if i == step_idx else {}} 
                                     for i, s in enumerate(steps)],
                            'verifier_results': verifier_results,
                            'consensus': consensus,
                            'current_step_idx': step_idx,
                            'parsed_expressions': parsed_exprs
                        }
                        
                        # Display final verdict prominently
                        verdict = consensus['final_verdict']
                        confidence = consensus['overall_confidence']
                        agreement = consensus['agreement_type']
                        error_score = consensus.get('error_score', 0)
                        
                        # Large verdict display
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            if verdict == 'VALID':
                                st.success(f"## ✅ FINAL VERDICT: VALID")
                            else:
                                st.error(f"## ❌ FINAL VERDICT: ERROR")
                        
                        with col2:
                            st.metric("Confidence", f"{confidence:.3f}")
                        
                        with col3:
                            st.metric("Agreement", agreement)
                        
                        # Show error details if ERROR (PRD Req 4.3.1)
                        if verdict == 'ERROR':
                            error_type = consensus.get('primary_error_type')
                            if error_type:
                                # Extract found/correct values for better explanation
                                found_value = ""
                                correct_value = ""
                                
                                # Try to extract from symbolic verifier result
                                if 'symbolic' in verifier_results:
                                    sym_details = verifier_results['symbolic'].get('details', '')
                                    # Extract numbers from details like "5 + 3 = 8.0, but step claims 9.0"
                                    import re
                                    match = re.search(r'=\s*([\d.]+).*claims\s*([\d.]+)', sym_details)
                                    if match:
                                        correct_value = match.group(1)
                                        found_value = match.group(2)
                                
                                explanation = generate_explanation(
                                    error_type, 
                                    step=step, 
                                    problem=problem,
                                    found_value=found_value,
                                    correct_value=correct_value
                                )
                                st.warning(f"**🔍 Error Type:** `{error_type}`")
                                st.info(f"**📝 Explanation:** {explanation}")
                                
                                # PRD Req 4.3.2: Show correction if available
                                correction_engine = CorrectionEngine()
                                correction = correction_engine.correct(error_type, step, problem)
                                
                                if correction.get('success'):
                                    st.success(f"**✅ Suggested Correction:** {correction['corrected_step']}")
                                    st.caption(f"Confidence: {correction['confidence']:.2f}")
                                elif correction.get('requires_review'):
                                    st.info("**💡 Hint:** This error may require manual review. Check the explanation above.")
                        
                        # Show consensus breakdown (PRD Req 4.4.6)
                        display_consensus_breakdown(consensus, st.session_state.consensus_engine.weights)
                        
                        # Also show table view
                        st.markdown("#### Detailed Breakdown Table")
                        breakdown = consensus.get('breakdown', {})
                        breakdown_df = pd.DataFrame([
                            {
                                'Model': name,
                                'Verdict': info['verdict'],
                                'Confidence': f"{info['confidence']:.3f}",
                                'Weight': f"{info['weight']:.3f}",
                                'Contribution': f"{info['contribution']:.3f}",
                                'Impact': '🔴 High' if abs(info['contribution']) > 0.2 else '🟡 Medium' if abs(info['contribution']) > 0.1 else '🟢 Low'
                            }
                            for name, info in breakdown.items()
                        ])
                        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                        
                        # ============================================
                        # STEP 3: Detailed Explanations from All Models
                        # ============================================
                        st.markdown("---")
                        st.markdown("### 🔍 Detailed Explanations - Where is the Error?")
                        st.write("Understanding why each model made its decision:")
                        
                        xai_explainer = XAIExplainer()
                        
                        # Generate explanations for each verifier
                        verifier_explanations = {}
                        for name, result in verifier_results.items():
                            verifier_explanations[name] = xai_explainer.explain_verifier_decision(
                                name, step, problem, result, prev_steps
                            )
                        
                        # Track error location (PRD Req 4.2.3)
                        if verdict == 'ERROR':
                            error_type = consensus.get('primary_error_type', 'logical_error')
                            error_tracker.track_step_error(
                                step_idx=step_idx,
                                step_text=step,
                                error_type=error_type,
                                confidence=confidence,
                                verifier_name='consensus'
                            )
                        
                        # Store step result
                        step_results_list.append({
                            'step_idx': step_idx,
                            'step_text': step,
                            'verdict': verdict,
                            'consensus': consensus,
                            'verifier_results': verifier_results
                        })
                        
                        # Generate consensus explanation
                        consensus_explanation = xai_explainer.explain_consensus(
                            consensus, verifier_results
                        )
                        
                        # Show explanations in tabs
                        exp_tabs = st.tabs([f"📊 {name.upper()}" for name in verifier_explanations.keys()] + ["⚖️ Consensus"])
                        
                        for idx, (name, explanation) in enumerate(verifier_explanations.items()):
                            with exp_tabs[idx]:
                                st.write(f"### {name.upper()} Explanation")
                                
                                # Verdict
                                verdict_exp = explanation.get('verdict', 'UNKNOWN')
                                if verdict_exp == 'ERROR':
                                    st.error(f"**Verdict:** ERROR")
                                elif verdict_exp == 'VALID':
                                    st.success(f"**Verdict:** VALID")
                                else:
                                    st.warning(f"**Verdict:** {verdict_exp}")
                                
                                # Reasoning
                                st.write("**Reasoning:**")
                                for reason in explanation.get('reasoning', []):
                                    st.write(f"• {reason}")
                                
                                # Evidence
                                if explanation.get('evidence'):
                                    st.write("**Evidence:**")
                                    for evidence in explanation.get('evidence', []):
                                        st.write(f"  - {evidence}")
                                
                                # Key Factors
                                if explanation.get('key_factors'):
                                    st.write("**Key Factors:**")
                                    for factor in explanation.get('key_factors', []):
                                        st.write(f"  • {factor}")
                                
                                # Confidence Factors
                                if explanation.get('confidence_factors'):
                                    st.write("**Confidence Breakdown:**")
                                    for factor, value in explanation.get('confidence_factors', {}).items():
                                        st.write(f"  - {factor}: {value:.3f}")
                                
                                # Special displays for specific verifiers
                                if name == 'ml_classifier' and explanation.get('top_predictions'):
                                    st.write("**Top Predictions:**")
                                    for pred in explanation.get('top_predictions', [])[:3]:
                                        st.write(f"  - {pred['class']}: {pred['probability']:.3f}")
                                
                                if name == 'ensemble' and explanation.get('voting_breakdown'):
                                    votes = explanation.get('voting_breakdown', {})
                                    st.write("**Voting Breakdown:**")
                                    st.write(f"  - ERROR votes: {votes.get('error', 0)}")
                                    st.write(f"  - VALID votes: {votes.get('valid', 0)}")
                        
                        # Consensus explanation tab
                        with exp_tabs[-1]:
                            st.write("### Consensus Explanation")
                            
                            st.write("**Final Verdict:**", consensus_explanation.get('final_verdict', 'UNKNOWN'))
                            st.write("**Agreement Type:**", consensus_explanation.get('agreement_type', 'MIXED'))
                            
                            st.write("**Reasoning:**")
                            for reason in consensus_explanation.get('reasoning', []):
                                st.write(f"• {reason}")
                            
                            st.write("**Contributing Factors:**")
                            for factor in consensus_explanation.get('contributing_factors', []):
                                st.write(f"• {factor}")
                            
                            # Show full explanation dashboard
                            st.markdown("---")
                            create_explanation_dashboard(verifier_explanations, consensus_explanation, step_idx=step_idx)
                        
                        st.divider()
                    
                    # Show error location analysis for all steps (PRD Req 4.2.3)
                    if error_tracker.step_errors:
                        st.markdown("---")
                        display_error_location(steps, step_results_list, error_tracker)
    
    with col2:
        st.header("🔄 Interactive Pipeline")
        
        # Track verification state for interactive flowchart
        if 'verification_state' not in st.session_state:
            st.session_state.verification_state = {
                'problem': '',
                'steps': [],
                'verifier_results': {},
                'consensus': {},
                'current_step_idx': 0,
                'parsed_expressions': []
            }
        
        # Update state if verification was just run
        if 'last_verification' in st.session_state:
            state = st.session_state.verification_state
            state.update(st.session_state.last_verification)
            render_interactive_flowchart(state)
        else:
            render_flowchart()
        
        st.header("📋 Live Logs")
        
        # Display recent logs
        recent_logs = st.session_state.logger.get_recent_logs(20)
        
        log_container = st.container()
        with log_container:
            for log in recent_logs:
                level = log['level']
                message = log['message']
                timestamp = log['timestamp']
                
                st.markdown(f"{level} **{timestamp}:** {message}")
                
                if log.get('details'):
                    with st.expander("Details"):
                        st.json(log['details'])


if __name__ == "__main__":
    main()

