"""
Streamlit Dashboard for Mathematical Reasoning Verification System
Interactive UI with real-time processing logs and results display
"""

import streamlit as st
import time
from typing import List, Dict, Any

from core import run_verification_parallel


# Initialize session state
if 'steps_log' not in st.session_state:
    st.session_state.steps_log = []
if 'results' not in st.session_state:
    st.session_state.results = None


def add_log(step: str, model: str, status: str, details: str):
    """Add entry to processing log."""
    log_entry = {
        "step": step,
        "model": model,
        "status": status,
        "details": details,
        "timestamp": time.time()
    }
    st.session_state.steps_log.append(log_entry)


def display_flowchart(problem="", steps_input=""):
    """Display interactive flowchart with expandable explanations."""
    
    # Check if we have results to show status
    has_results = st.session_state.results is not None
    results = st.session_state.results if has_results else None
    
    # Parse steps if provided
    steps = []
    if steps_input:
        steps = [s.strip() for s in steps_input.split('\n') if s.strip()]
    elif has_results:
        steps = results.get('steps', [])
    
    # Problem-Specific Flowchart
    if problem or steps:
        st.markdown("### 📊 Problem Flowchart")
        st.markdown("**Problem:**")
        st.info(problem if problem else "No problem entered yet")
        
        if steps:
            st.markdown("**Solution Flow:**")
            # Create flowchart for the actual problem
            flowchart_lines = []
            flowchart_lines.append("```")
            # Format problem text to fit in box
            problem_display = problem[:45] + "..." if len(problem) > 45 else problem
            problem_display = problem_display.ljust(50)
            flowchart_lines.append("┌─────────────────────────────────────────────────────────┐")
            flowchart_lines.append(f"│  📥 PROBLEM: {problem_display} │")
            flowchart_lines.append("└────────────────────┬────────────────────────────────────┘")
            flowchart_lines.append("                     │")
            flowchart_lines.append("                     ▼")
            
            for i, step in enumerate(steps, 1):
                # Extract key info from step
                step_short = step[:45] + "..." if len(step) > 45 else step
                
                # Check if this step has an error (if results available)
                has_error = False
                if has_results:
                    classified_errors = results.get('classified_errors', [])
                    for error in classified_errors:
                        if error.get('step_number') == i:
                            has_error = True
                            break
                
                # Determine status icon
                status_icon = "❌" if has_error else "✅"
                
                # Format step text to fit in box (max 45 chars)
                step_display = step_short.ljust(45)
                
                if i < len(steps):
                    flowchart_lines.append(f"┌─────────────────────────────────────────────────────────┐")
                    flowchart_lines.append(f"│  {status_icon} STEP {i}: {step_display} │")
                    flowchart_lines.append("└────────────────────┬────────────────────────────────────┘")
                    flowchart_lines.append("                     │")
                    flowchart_lines.append("                     ▼")
                else:
                    flowchart_lines.append(f"┌─────────────────────────────────────────────────────────┐")
                    flowchart_lines.append(f"│  {status_icon} STEP {i}: {step_display} │")
                    flowchart_lines.append("└────────────────────┬────────────────────────────────────┘")
                    flowchart_lines.append("                     │")
                    flowchart_lines.append("                     ▼")
                    flowchart_lines.append("┌─────────────────────────────────────────────────────────┐")
                    flowchart_lines.append("│  📤 FINAL ANSWER                                        │")
                    if has_results:
                        consensus = results.get('consensus', {})
                        final_verdict = consensus.get('final_verdict', 'UNKNOWN')
                        verdict_icon = "❌ ERROR" if final_verdict == "ERROR" else "✅ VALID"
                        verdict_display = verdict_icon.ljust(55)
                        flowchart_lines.append(f"│  {verdict_display} │")
                    flowchart_lines.append("└─────────────────────────────────────────────────────────┘")
            
            flowchart_lines.append("```")
            flowchart_text = "\n".join(flowchart_lines)
            st.code(flowchart_text, language=None)
            
            # Show step details
            st.markdown("**Step Details:**")
            for i, step in enumerate(steps, 1):
                # Check for errors in this step
                step_errors = []
                if has_results:
                    classified_errors = results.get('classified_errors', [])
                    step_errors = [e for e in classified_errors if e.get('step_number') == i]
                
                if step_errors:
                    with st.expander(f"Step {i}: {step[:60]}{'...' if len(step) > 60 else ''} ❌", expanded=False):
                        st.write(f"**Full Step:** {step}")
                        for error in step_errors:
                            st.error(f"**Error Found:** {error.get('category', 'Unknown')}")
                            st.write(f"- Found: `{error.get('found', 'N/A')}`")
                            st.write(f"- Correct: `{error.get('correct', 'N/A')}`")
                            explanations = results.get('explanations', {})
                            if i in explanations:
                                st.info(f"**Explanation:** {explanations[i]}")
                else:
                    with st.expander(f"Step {i}: {step[:60]}{'...' if len(step) > 60 else ''} ✅", expanded=False):
                        st.write(f"**Full Step:** {step}")
                        st.success("No errors detected in this step")
        else:
            st.info("Enter solution steps to see the problem flowchart")
    
    st.markdown("---")
    
    # Step 1: INPUT
    with st.expander("📥 **STEP 1: INPUT** - Problem & Solution Steps", expanded=True):
        st.markdown("""
        **What happens here?**
        - The system receives the math problem and step-by-step solution
        - Input is validated and prepared for processing
        - Steps are parsed and segmented for analysis
        """)
        if problem:
            st.success(f"✅ Received problem: {problem}")
        if steps:
            st.success(f"✅ Received {len(steps)} solution steps")
        if has_results:
            st.code(f"Problem: {results.get('problem', '')[:100]}...")
    
    # Step 2: PARSING
    with st.expander("🔍 **STEP 2: PARSING** - Extract Mathematical Expressions", expanded=has_results):
        st.markdown("""
        **What happens here?**
        - Mathematical expressions are extracted using regex patterns
        - Operations (+, -, *, /) are identified
        - Numbers and variables are recognized
        - Each step is prepared for verification
        """)
        if has_results:
            steps = results.get('steps', [])
            st.success(f"✅ Parsed {len(steps)} steps")
            for i, step in enumerate(steps[:3], 1):
                st.write(f"  Step {i}: {step[:60]}...")
            if len(steps) > 3:
                st.write(f"  ... and {len(steps) - 3} more steps")
    
    # Step 3: PARALLEL EXECUTION
    with st.expander("🔄 **STEP 3: PARALLEL EXECUTION** - 3 Models Running Simultaneously", expanded=has_results):
        st.markdown("""
        **What happens here?**
        - **Model 1 (Symbolic) 🔢**: Uses SymPy to verify all arithmetic calculations
          - Weight: 40% (most reliable for math)
          - Not affected by sidebar selection
        
        - **Model 2 (LLM Logical) 🧠**: Checks for logical consistency and contradictions
          - Weight: 35%
          - Uses first selected model from sidebar (e.g., GPT-4)
          - Currently: Pattern-based simulation
        
        - **Model 3 (Ensemble) 🤖**: Simulates multiple LLMs voting on solution validity
          - Weight: 25%
          - Uses ALL selected models from sidebar (GPT-4, Llama 2, Gemini)
          - Each model votes, majority wins
          - Currently: Pattern-based simulation
        
        All three models run **in parallel** using ThreadPoolExecutor for speed!
        """)
        if has_results:
            model_results = results.get('model_results', {})
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'symbolic' in model_results:
                    verdict = model_results['symbolic']['verdict']
                    conf = model_results['symbolic']['confidence'] * 100
                    errors = len(model_results['symbolic'].get('errors', []))
                    if verdict == "ERROR":
                        st.error(f"**🔢 Symbolic**\n\n❌ {verdict}\n\nConfidence: {conf:.1f}%\n\nErrors: {errors}")
                    else:
                        st.success(f"**🔢 Symbolic**\n\n✅ {verdict}\n\nConfidence: {conf:.1f}%\n\nErrors: {errors}")
            
            with col2:
                if 'llm_logical' in model_results:
                    verdict = model_results['llm_logical']['verdict']
                    conf = model_results['llm_logical']['confidence'] * 100
                    errors = len(model_results['llm_logical'].get('errors', []))
                    if verdict == "ERROR":
                        st.error(f"**🧠 LLM Logical**\n\n❌ {verdict}\n\nConfidence: {conf:.1f}%\n\nErrors: {errors}")
                    else:
                        st.success(f"**🧠 LLM Logical**\n\n✅ {verdict}\n\nConfidence: {conf:.1f}%\n\nErrors: {errors}")
            
            with col3:
                if 'ensemble' in model_results:
                    verdict = model_results['ensemble']['verdict']
                    conf = model_results['ensemble']['confidence'] * 100
                    agreement = model_results['ensemble'].get('agreement', 'N/A')
                    if verdict == "ERROR":
                        st.error(f"**🤖 Ensemble**\n\n❌ {verdict}\n\nConfidence: {conf:.1f}%\n\nAgreement: {agreement}")
                    else:
                        st.success(f"**🤖 Ensemble**\n\n✅ {verdict}\n\nConfidence: {conf:.1f}%\n\nAgreement: {agreement}")
        else:
            st.info("⏳ Models will execute in parallel when you click 'Verify Solution'")
    
    # Step 4: CONSENSUS
    with st.expander("⚖️ **STEP 4: CONSENSUS** - Weighted Voting Mechanism", expanded=has_results):
        st.markdown("""
        **What happens here?**
        - The system combines results from all 3 models using **weighted voting**:
          - **Symbolic Model**: 40% weight (most reliable for arithmetic)
          - **LLM Logical Model**: 35% weight (good for reasoning)
          - **Ensemble Model**: 25% weight (provides diversity)
        
        - An **error score** is calculated: if > 0.50, verdict = ERROR
        - **Confidence** is adjusted based on agreement:
          - All 3 agree: confidence boosted by 10%
          - 2/3 agree: uses average of agreeing models
          - Mixed: confidence penalized by 20%
        """)
        if has_results:
            consensus = results.get('consensus', {})
            final_verdict = consensus.get('final_verdict', 'UNKNOWN')
            overall_conf = consensus.get('overall_confidence', 0) * 100
            error_score = consensus.get('error_score', 0)
            agreement = consensus.get('agreement_type', 'UNKNOWN')
            
            st.markdown(f"""
            **Consensus Results:**
            - **Final Verdict**: {'❌ ERROR' if final_verdict == 'ERROR' else '✅ VALID'}
            - **Overall Confidence**: {overall_conf:.1f}%
            - **Error Score**: {error_score:.3f} (threshold: 0.50)
            - **Agreement Type**: {agreement}
            """)
            
            # Show individual model contributions
            st.markdown("**Model Contributions:**")
            individual_verdicts = consensus.get('individual_verdicts', {})
            individual_confidences = consensus.get('individual_confidences', {})
            weights = {"symbolic": 0.40, "llm_logical": 0.35, "ensemble": 0.25}
            
            for model_name, verdict in individual_verdicts.items():
                weight = weights.get(model_name, 0)
                confidence = individual_confidences.get(model_name, 0) * 100
                contribution = weight * individual_confidences.get(model_name, 0)
                st.write(f"  - **{model_name.title()}**: {verdict} ({confidence:.1f}% confidence) → {weight*100:.0f}% weight → contributes {contribution:.3f}")
    
    # Step 5: ERROR CLASSIFICATION
    with st.expander("🏷️ **STEP 5: ERROR CLASSIFICATION** - Categorize & Analyze Errors", expanded=has_results and len(results.get('classified_errors', [])) > 0):
        st.markdown("""
        **What happens here?**
        - Each detected error is classified into one of 10+ error types:
          - Arithmetic Error (calculation mistakes)
          - Logical Error (contradictions)
          - Operation Mismatch (says one thing, does another)
          - Semantic Error (meaning doesn't match)
          - And more...
        
        - **Severity** is assigned: HIGH, MEDIUM, or LOW
        - **Fixability** is assessed: can the error be auto-corrected?
        """)
        if has_results:
            classified_errors = results.get('classified_errors', [])
            if classified_errors:
                st.success(f"✅ Classified {len(classified_errors)} error(s)")
                for error in classified_errors[:3]:
                    st.markdown(f"""
                    **Error in Step {error.get('step_number', '?')}:**
                    - **Category**: {error.get('category', 'Unknown')}
                    - **Severity**: {error.get('severity', 'Unknown')}
                    - **Fixable**: {'Yes' if error.get('fixable', False) else 'No'}
                    - **Fixability Score**: {error.get('fixability_score', 0)*100:.0f}%
                    """)
            else:
                st.info("✅ No errors found - solution is valid!")
    
    # Step 6: EXPLANATION GENERATION
    with st.expander("💬 **STEP 6: EXPLANATION GENERATION** - Create Human-Readable Explanations", expanded=has_results and len(results.get('explanations', {})) > 0):
        st.markdown("""
        **What happens here?**
        - For each error, a natural language explanation is generated
        - Explains **why** the error occurred
        - Provides educational context
        - Suggests how to avoid similar mistakes
        - Includes learning tips
        """)
        if has_results:
            explanations = results.get('explanations', {})
            if explanations:
                st.success(f"✅ Generated {len(explanations)} explanation(s)")
                for step_num, explanation in list(explanations.items())[:2]:
                    with st.container():
                        st.markdown(f"**Step {step_num} Explanation:**")
                        st.info(explanation)
            else:
                st.info("✅ No explanations needed - solution is correct!")
    
    # Step 7: ERROR CORRECTION
    with st.expander("🔧 **STEP 7: ERROR CORRECTION** - Automatic Fixes", expanded=has_results and results.get('correction', {}).get('fixed_count', 0) > 0):
        st.markdown("""
        **What happens here?**
        - Fixable errors are automatically corrected
        - Arithmetic errors: correct values are calculated and replaced
        - Operation mismatches: operations are corrected
        - Success rate is tracked for each error type
        - Errors requiring manual review are flagged
        """)
        if has_results:
            correction = results.get('correction', {})
            fixed_count = correction.get('fixed_count', 0)
            if fixed_count > 0:
                st.success(f"✅ Fixed {fixed_count} error(s)")
                st.write(f"**Success Rate**: {correction.get('success_rate', 0)*100:.1f}%")
                correction_log = correction.get('correction_log', [])
                if correction_log:
                    for log_entry in correction_log[:2]:
                        st.markdown(f"""
                        **Step {log_entry['step']} ({log_entry['type']}):**
                        - Original: `{log_entry['original']}`
                        - Corrected: `{log_entry['corrected']}`
                        """)
            else:
                st.info("✅ No corrections needed")
    
    # Step 8: OUTPUT
    with st.expander("📤 **STEP 8: OUTPUT** - Final Results", expanded=has_results):
        st.markdown("""
        **What happens here?**
        - Final verdict is displayed (VALID or ERROR)
        - Overall confidence score is shown
        - All errors with explanations are presented
        - Processing time is reported
        - Results are ready for review
        """)
        if has_results:
            consensus = results.get('consensus', {})
            final_verdict = consensus.get('final_verdict', 'UNKNOWN')
            overall_conf = consensus.get('overall_confidence', 0) * 100
            processing_time = results.get('processing_time', 0)
            total_errors = len(results.get('classified_errors', []))
            
            if final_verdict == "ERROR":
                st.error(f"**Final Verdict**: ❌ {final_verdict}")
            else:
                st.success(f"**Final Verdict**: ✅ {final_verdict}")
            
            st.metric("Overall Confidence", f"{overall_conf:.1f}%")
            st.metric("Processing Time", f"{processing_time:.3f}s")
            st.metric("Total Errors Found", total_errors)
            
            st.success("✅ Verification complete! Results displayed above.")
        else:
            st.info("⏳ Results will appear here after verification")
    
    # Visual flowchart diagram
    st.markdown("---")
    st.markdown("### 📊 Processing Flow Diagram")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────┐
    │                    📥 INPUT                              │
    │         Problem + Solution Steps                        │
    └────────────────────┬────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────┐
    │                  🔍 PARSING                              │
    │    Extract expressions, identify operations             │
    └────────────────────┬────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────┐
    │            🔄 PARALLEL EXECUTION                         │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
    │  │ Symbolic │  │ LLM      │  │ Ensemble │                │
    │  │ (SymPy)  │  │ Logical  │  │ (Voting)│                │
    │  │   40%    │  │   35%    │  │   25%   │                │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘                │
    └───────┼─────────────┼──────────────┼──────────────────────┘
            │             │              │
            └─────────────┴──────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────┐
    │                ⚖️ CONSENSUS                             │
    │    Weighted Voting → Final Verdict & Confidence         │
    └────────────────────┬────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────┐
    │            🏷️ ERROR CLASSIFICATION                      │
    │    Categorize → Severity → Fixability                   │
    └────────────────────┬────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────┐
    │        💬 EXPLANATION GENERATION                          │
    │    Natural language explanations for each error          │
    └────────────────────┬────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────┐
    │            🔧 ERROR CORRECTION                           │
    │    Auto-fix fixable errors → Track success rate          │
    └────────────────────┬────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────┐
    │                    📤 OUTPUT                             │
    │    Final Verdict + Confidence + All Details             │
    └─────────────────────────────────────────────────────────┘
    ```""")


def display_logs():
    """Display processing logs with color coding."""
    if not st.session_state.steps_log:
        return
    
    st.subheader("📊 Processing Flow")
    
    for log_entry in st.session_state.steps_log:
        status = log_entry["status"]
        step = log_entry["step"]
        model = log_entry["model"]
        details = log_entry["details"]
        
        # Color coding based on status
        if status.startswith("✓"):
            st.success(f"**{status}** [{model}] {step}: {details}")
        elif status.startswith("❌"):
            st.error(f"**{status}** [{model}] {step}: {details}")
        elif status.startswith("⚠️"):
            st.warning(f"**{status}** [{model}] {step}: {details}")
        else:
            st.info(f"**{status}** [{model}] {step}: {details}")


def display_results():
    """Display verification results."""
    if not st.session_state.results:
        return
    
    results = st.session_state.results
    
    st.header("🎯 Results")
    
    # Final verdict, confidence, processing time
    col1, col2, col3 = st.columns(3)
    
    consensus = results.get("consensus", {})
    final_verdict = consensus.get("final_verdict", "UNKNOWN")
    overall_confidence = consensus.get("overall_confidence", 0.0)
    processing_time = results.get("processing_time", 0.0)
    
    with col1:
        if final_verdict == "ERROR":
            st.error(f"**Final Verdict:** ❌ {final_verdict}")
        else:
            st.success(f"**Final Verdict:** ✅ {final_verdict}")
    
    with col2:
        st.metric("**Confidence**", f"{overall_confidence * 100:.1f}%")
    
    with col3:
        st.metric("**Processing Time**", f"{processing_time:.2f}s")
    
    # Model verdicts
    st.subheader("🤖 Model Verdicts")
    model_results = results.get("model_results", {})
    
    cols = st.columns(3)
    model_names = ["symbolic", "llm_logical", "ensemble"]
    model_display_names = {
        "symbolic": "🔢 Symbolic",
        "llm_logical": "🧠 LLM Logical",
        "ensemble": "🤖 Ensemble"
    }
    
    for idx, model_name in enumerate(model_names):
        with cols[idx]:
            if model_name in model_results:
                model_result = model_results[model_name]
                verdict = model_result.get("verdict", "UNKNOWN")
                confidence = model_result.get("confidence", 0.0)
                errors_count = len(model_result.get("errors", []))
                
                if verdict == "ERROR":
                    st.error(f"**{model_display_names[model_name]}**\n\nVerdict: ❌ {verdict}\n\nConfidence: {confidence * 100:.1f}%\n\nErrors: {errors_count}")
                else:
                    st.success(f"**{model_display_names[model_name]}**\n\nVerdict: ✅ {verdict}\n\nConfidence: {confidence * 100:.1f}%\n\nErrors: {errors_count}")
    
    # Consensus mechanism breakdown
    st.subheader("⚖️ Consensus Mechanism")
    agreement_type = consensus.get("agreement_type", "UNKNOWN")
    error_score = consensus.get("error_score", 0.0)
    individual_verdicts = consensus.get("individual_verdicts", {})
    individual_confidences = consensus.get("individual_confidences", {})
    
    st.info(f"**Agreement:** {agreement_type}")
    st.info(f"**Error Score:** {error_score:.3f} (threshold: 0.50)")
    
    st.write("**Individual Model Results:**")
    for model_name, verdict in individual_verdicts.items():
        confidence = individual_confidences.get(model_name, 0.0)
        st.write(f"- {model_display_names.get(model_name, model_name)}: {verdict} (confidence: {confidence * 100:.1f}%)")
    
    # Error details
    classified_errors = results.get("classified_errors", [])
    if classified_errors:
        st.subheader("🔴 Error Details")
        for error in classified_errors:
            with st.expander(f"Error in Step {error.get('step_number', 0)}: {error.get('category', 'Unknown')}"):
                st.write(f"**Type:** {error.get('type', 'unknown')}")
                st.write(f"**Found:** {error.get('found', 'N/A')}")
                st.write(f"**Correct:** {error.get('correct', 'N/A')}")
                st.write(f"**Severity:** {error.get('severity', 'UNKNOWN')}")
                st.write(f"**Fixable:** {'Yes' if error.get('fixable', False) else 'No'}")
                
                # Show explanation
                explanations = results.get("explanations", {})
                step_num = error.get("step_number", 0)
                if step_num in explanations:
                    st.write(f"**Explanation:** {explanations[step_num]}")
    
    # Correction results
    correction = results.get("correction", {})
    if correction and correction.get("fixed_count", 0) > 0:
        st.subheader("🔧 Corrections Applied")
        st.success(f"**Fixed:** {correction.get('fixed_count', 0)} / {correction.get('total_fixable', 0)} errors")
        st.write(f"**Success Rate:** {correction.get('success_rate', 0.0) * 100:.1f}%")
        
        correction_log = correction.get("correction_log", [])
        if correction_log:
            with st.expander("View Correction Log"):
                for log_entry in correction_log:
                    st.write(f"**Step {log_entry['step']}:** {log_entry['type']}")
                    st.write(f"Original: {log_entry['original']}")
                    st.write(f"Corrected: {log_entry['corrected']}")


# Main UI
st.set_page_config(page_title="Math Verification System", page_icon="🔢", layout="wide")

st.title("🔢 Mathematical Reasoning Verification System")
st.markdown("3-Model Parallel Verification with Weighted Consensus")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    gpt4_enabled = st.checkbox("GPT-4", value=True)
    llama_enabled = st.checkbox("Llama 2", value=True)
    gemini_enabled = st.checkbox("Gemini", value=True)
    
    selected_models = []
    if gpt4_enabled:
        selected_models.append("GPT-4")
    if llama_enabled:
        selected_models.append("Llama 2")
    if gemini_enabled:
        selected_models.append("Gemini")
    
    if not selected_models:
        selected_models = ["GPT-4", "Llama 2", "Gemini"]
    
    st.info(f"Selected models: {', '.join(selected_models)}")
    
    st.markdown("---")
    st.markdown("### 📖 How Models Are Used")
    with st.expander("ℹ️ Click to see how sidebar models work"):
        st.markdown("""
        **Model 1 (Symbolic) 🔢:**
        - Uses SymPy library (not affected by sidebar)
        - Verifies arithmetic calculations
        - Weight: 40%
        
        **Model 2 (LLM Logical) 🧠:**
        - Uses first selected model from sidebar
        - Checks logical consistency
        - Weight: 35%
        
        **Model 3 (Ensemble) 🤖:**
        - Uses ALL selected models from sidebar
        - Each model votes on solution validity
        - Majority voting determines verdict
        - Weight: 25%
        
        **Note:** Currently using pattern-based simulation.
        For production, integrate real LLM APIs (OpenAI, Anthropic, Google).
        """)
        
        st.markdown("### 🔄 Model Flow Diagram")
        st.code("""
        Sidebar Selection
        ┌─────────────────┐
        │ GPT-4    ✓     │
        │ Llama 2  ✓     │──┐
        │ Gemini   ✓     │  │
        └─────────────────┘  │
                             │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
        Model 2 (LLM Logical)              Model 3 (Ensemble)
        Uses: GPT-4 (first selected)       Uses: All selected
        Weight: 35%                         Weight: 25%
        └───────────────────────────────────────────┘
                             │
                             ▼
                    Consensus Mechanism
                    (Weighted Voting)
        """, language=None)

# Main layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.header("📝 Input")
    
    problem = st.text_area(
        "Problem:",
        height=80,
        placeholder="Enter the math problem here...",
        value="Janet has 3 apples. She buys 2 more. She gives 1 away. How many?"
    )
    
    steps_input = st.text_area(
        "Solution Steps (one per line):",
        height=120,
        placeholder="Enter solution steps, one per line...",
        value="Janet starts with 3 apples\nShe buys 2 more: 3 + 2 = 5 apples\nShe gives 1 away: 5 - 1 = 6 apples"
    )
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        verify_button = st.button("🚀 Verify Solution", type="primary", use_container_width=True)
    
    with col_btn2:
        clear_button = st.button("🔄 Clear", use_container_width=True)

with col_right:
    st.header("🎯 Live Flowchart")
    display_flowchart(problem=problem, steps_input=steps_input)

# Handle button clicks
if verify_button:
    # Clear previous logs
    st.session_state.steps_log = []
    st.session_state.results = None
    
    # Parse steps
    steps = [s.strip() for s in steps_input.split('\n') if s.strip()]
    
    if not problem or not steps:
        st.warning("Please enter both problem and solution steps.")
    else:
        # Add initial log
        add_log("Verification Started", "System", "⏳", "Initializing models...")
        
        # Run verification
        with st.spinner("Running verification..."):
            try:
                # Add logs for each model starting
                add_log("Model 1 Started", "Symbolic", "🔢", "Checking arithmetic...")
                add_log("Model 2 Started", "LLM Logical", "🧠", "Checking logical consistency...")
                add_log("Model 3 Started", "Ensemble", "🤖", "Running ensemble voting...")
                
                # Use first selected model for LLM Logical, or default to GPT-4
                llm_model_name = selected_models[0] if selected_models else "GPT-4"
                
                results = run_verification_parallel(
                    problem=problem,
                    steps=steps,
                    model_name=llm_model_name,
                    model_list=selected_models
                )
                
                # Add completion logs
                for model_name, model_result in results["model_results"].items():
                    verdict = model_result.get("verdict", "UNKNOWN")
                    errors_count = len(model_result.get("errors", []))
                    status = "✓ ERROR" if verdict == "ERROR" else "✓ VALID"
                    add_log(
                        f"Model {model_name} Completed",
                        model_name.title(),
                        status,
                        f"Found {errors_count} error(s)"
                    )
                
                # Add consensus log
                consensus_verdict = results["consensus"].get("final_verdict", "UNKNOWN")
                add_log(
                    "Consensus Computed",
                    "Consensus",
                    "⚖️",
                    f"Final verdict: {consensus_verdict}"
                )
                
                st.session_state.results = results
                
            except Exception as e:
                st.error(f"Error during verification: {str(e)}")
                add_log("Error", "System", "❌", str(e))

if clear_button:
    st.session_state.steps_log = []
    st.session_state.results = None
    st.rerun()

# Display logs and results
display_logs()
display_results()

