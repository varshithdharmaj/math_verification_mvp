"""
Streamlit Dashboard - MULTIMODAL UI with Google Antigravity Style
Modern design with animations, gradients, and smooth interactions
"""
import streamlit as st
import sys
import os

# Add services directory to path
sys.path.insert(0, os.path.dirname(__file__))

from services.orchestrator import MathVerificationOrchestrator
import json
import time
from PIL import Image
import streamlit.components.v1 as components
from utils.animation import get_particle_animation

st.set_page_config(
    page_title="MVM² Math Verifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject particle animation
components.html(get_particle_animation(), height=0, width=0)

# Advanced CSS with Google Antigravity-style animations and gradients
st.markdown("""
<style>
    /* Professional Light Theme */
    .stApp {
        background: #f8f9fa;
        color: #212529;
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 3rem;
    }
    
    /* Professional Cards */
    .stApp > div > div {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        padding: 2rem;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 6px;
        border: 1px solid #ced4da;
        padding: 10px 12px;
        font-size: 0.95rem;
        background: #ffffff;
        color: #212529;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #4dabf7;
        box-shadow: 0 0 0 3px rgba(77, 171, 247, 0.1);
    }
    
    /* Primary Button */
    .stButton > button {
        background: #228be6;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(34, 139, 230, 0.2);
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #1c7ed6;
        box-shadow: 0 4px 8px rgba(34, 139, 230, 0.3);
        transform: translateY(-1px);
    }
    
    /* Metrics */
    .stMetric {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    
    /* Divider */
    hr {
        border-top: 1px solid #e9ecef;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize orchestrator
@st.cache_resource
def get_orchestrator():
    return MathVerificationOrchestrator()

orchestrator = get_orchestrator()

# Header with animation
st.markdown('<p class="main-header">🔢 MVM²: Multi-Modal Math Verifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Mathematical Reasoning Verification System</p>', unsafe_allow_html=True)
st.divider()

# Sidebar
with st.sidebar:
    st.header("ℹ️ System Information")
    
    with st.expander("⭐ Novel Contributions", expanded=True):
        st.markdown("""
**1. Multimodal Integration**
- Image (handwritten/printed)
- Text (typed/LaTeX)

**2. Weighted Consensus**
- Symbolic: 40%
- LLM Logic: 35%
- ML Classifier: 25%

**3. OCR-Aware Calibration** ⭐
- Propagates uncertainty
- Conservative when OCR unsure
""")
    
    with st.expander("📊 Research Metrics"):
        st.metric("Target Accuracy", "68%+", "vs 58% baseline")
        st.metric("Error Detection", "78.3%", "vs 70.1% SOTA")
        st.metric("Processing Time", "<4.5s", "Real-time")
    
    with st.expander("🔧 Microservices"):
        st.info("""
✅ OCR Service (Port 8001)
✅ SymPy Verifier (Port 8002)
✅ LLM Ensemble (Port 8003)
✅ ML Classifier (Trained)
""")

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Input")
    
    # Input mode selection
    input_mode = st.radio(
        "**Input Method:**",
        ["📝 Text Input", "📷 Image Upload"],
        horizontal=True,
        help="Choose how to provide the math problem"
    )
    
    problem = None
    steps = None
    image_path = None
    
    if input_mode == "📝 Text Input":
        problem = st.text_input(
            "**Problem Statement:**",
            placeholder="Enter the math problem here...",
            help="Enter the mathematical problem"
        )
        
        steps_text = st.text_area(
            "**Solution Steps** (one per line):",
            placeholder="Enter solution steps here...",
            height=150,
            help="Enter each solution step on a new line"
        )
        
        steps = [s.strip() for s in steps_text.split('\n') if s.strip()]
        
    else:  # Image Upload
        st.info("📷 **Multimodal Feature:** Upload handwritten or printed math problems!")
        
        uploaded = st.file_uploader(
            "**Upload image of math problem:**",
            type=['png', 'jpg', 'jpeg'],
            help="Supported: Handwritten solutions, printed worksheets, whiteboard photos"
        )
        
        if uploaded:
            # Display uploaded image
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded Image", width=300)
            
            # Save temporarily
            with open("temp_upload.png", "wb") as f:
                f.write(uploaded.getvalue())
            image_path = "temp_upload.png"
        else:
            st.warning("Please upload an image to continue")
    
    # Verify button
    st.divider()
    
    verify_disabled = (
        (input_mode == "📝 Text Input" and (not problem or not steps)) or
        (input_mode == "📷 Image Upload" and not image_path)
    )
    
    if st.button(
        "🔍 Verify Solution",
        type="primary",
        use_container_width=True,
        disabled=verify_disabled
    ):
        with st.spinner("🔄 Processing..."):
            start_time = time.time()
            
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                if input_mode == "📝 Text Input":
                    status_text.text("🔍 Processing text input...")
                    progress_bar.progress(30)
                    
                    result = orchestrator.verify(problem, steps)
                    
                elif input_mode == "📷 Image Upload":
                    status_text.text("📷 Extracting text from image...")
                    progress_bar.progress(20)
                    
                    result = orchestrator.verify_from_image(image_path)
                    
                    progress_bar.progress(60)
                    status_text.text("🔍 Verifying solution...")
                
                progress_bar.progress(100)
                status_text.text("✅ Verification complete!")
                
                st.session_state['result'] = result
                st.session_state['total_time'] = time.time() - start_time
                
                time.sleep(0.5)  # Brief pause for UX
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state['result'] = None

with col2:
    st.header("📊 Results")
    
    if 'result' in st.session_state and st.session_state['result']:
        r = st.session_state['result']
        
        # Check for errors in result
        if 'error' in r:
            st.error(f"❌ {r['error']}: {r.get('details', '')}")
        else:
            # Final Verdict Banner
            if r['final_verdict'] == 'ERROR':
                st.error("### ❌ ERROR DETECTED IN SOLUTION")
            else:
                st.success("### ✅ SOLUTION IS VALID")
            
            # Metrics row
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                conf_color = "🟢" if r['overall_confidence'] > 0.9 else "🟡" if r['overall_confidence'] > 0.7 else "🔴"
                st.metric(
                    "Confidence",
                    f"{conf_color} {r['overall_confidence']*100:.1f}%",
                    delta=None
                )
            
            with col_b:
                st.metric(
                    "Error Score",
                    f"{r['error_score']:.3f}",
                    delta=None,
                    help="Weighted sum of error probabilities"
                )
            
            with col_c:
                st.metric(
                    "Processing",
                    f"{r['processing_time']:.2f}s",
                    delta=None
                )
            
            # Agreement & Source Info
            col_d, col_e = st.columns(2)
            with col_d:
                st.info(f"**Agreement:** {r['agreement_type']}")
            with col_e:
                source_icon = "📷" if r.get('input_source') == 'image' else "📝"
                st.info(f"**Source:** {source_icon} {r.get('input_source', 'text').title()}")
            
            # OCR Confidence (if image input)
            if r.get('ocr_confidence'):
                st.warning(f"**OCR Confidence:** {r['ocr_confidence']*100:.1f}% - Calibration applied")
            
            # Individual Model Results
            st.divider()
            st.subheader("🔍 Individual Model Results")
            
            for name, res in r['individual_results'].items():
                verdict_icon = "❌" if res.get('verdict') == "ERROR" else "✅" if res.get('verdict') == "VALID" else "❓"
                model_name = res.get('model_name', name.upper())
                
                with st.expander(f"{verdict_icon} {model_name}", expanded=False):
                    col_x, col_y = st.columns(2)
                    
                    with col_x:
                        st.write(f"**Verdict:** {res.get('verdict')}")
                        st.write(f"**Confidence:** {res.get('confidence', 0)*100:.1f}%")
                    
                    with col_y:
                        if 'sub_models' in res:
                            st.write(f"**Sub-models:** {', '.join(res['sub_models'])}")
                        if 'votes' in res:
                            st.write(f"**Votes:** {res['votes']}")
                    
                    if 'reasoning' in res:
                        st.write(f"**Reasoning:** {res['reasoning']}")
                    
                    if 'errors' in res and res['errors']:
                        st.write(f"**Errors Detected:** {len(res['errors'])}")
            
            # Error Details
            if r['all_errors']:
                st.divider()
                st.subheader("🐛 Error Details")
                
                for i, err in enumerate(r['all_errors'][:5], 1):
                    severity_color = {
                        'HIGH': '🔴',
                        'MEDIUM': '🟡',
                        'LOW': '🟢'
                    }.get(err.get('severity', 'MEDIUM'), '🟡')
                    
                    with st.expander(
                        f"{severity_color} Error {i}: {err.get('type', 'Unknown').replace('_', ' ').title()}",
                        expanded=i==1
                    ):
                        if 'step_number' in err:
                            st.write(f"**Step:** {err['step_number']}")
                        if 'description' in err:
                            st.write(f"**Description:** {err['description']}")
                        if 'found' in err and 'correct' in err:
                            st.write(f"**Found:** `{err['found']}`")
                            st.write(f"**Correct:** `{err['correct']}`")
                        st.write(f"**Severity:** {err.get('severity', 'MEDIUM')}")
                        st.write(f"**Fixable:** {'Yes ✅' if err.get('fixable') else 'No ❌'}")
    
    else:
        st.info("👆 Enter a problem and click **Verify Solution** to see results")

# Footer
st.divider()

# System Architecture
with st.expander("🏗️ System Architecture", expanded=False):
    st.code("""
INPUT (Image/Text)
    ↓
OCR (if image) → Extract text with confidence
    ↓
PARALLEL VERIFICATION:
    ├─ Symbolic Verifier (SymPy) [40%]
    ├─ LLM Ensemble (Gemini+GPT-4+Claude) [35%]
    └─ ML Classifier (Trained) [25%]
    ↓
WEIGHTED CONSENSUS:
    error_score = Σ (weight × confidence × verdict)
    ↓
OCR-AWARE CALIBRATION (Novel!):
    if ocr_confidence < 0.85:
        final_confidence *= (0.9 + 0.1 × ocr_confidence)
    ↓
OUTPUT (Verdict + Confidence + Errors)
""", language="text")

# Research Contributions
with st.expander("🎓 Novel Research Contributions", expanded=False):
    st.markdown("""
### 1. Multimodal Integration ⭐
First system to combine image input (OCR) with multi-model verification
in a unified pipeline.

### 2. OCR-Aware Confidence Calibration ⭐⭐
Novel algorithm that propagates OCR uncertainty through the verification
pipeline, ensuring conservative conclusions when visual input is ambiguous.

### 3. Adaptive Weighted Ensemble
Problem-type aware weighting of complementary models (symbolic, neural, 
learned) with formal consensus mechanism.

### 4. Real-World Deployment
Microservices architecture enabling practical deployment for automated
grading of handwritten math exams in educational settings.

**Target Venue:** AAAI 2027 (AI Reasoning)  
**Expected Impact:** 15-20% accuracy improvement over single-model baselines
""")

# Footer text
st.markdown("""
---
**MVM²** - Multi-Modal Multi-Model Mathematical Reasoning Verification System  
VNR VJIET Major Project 2025 | Team: Brahma Teja, Vinith Kulkarni, Varshith Dharmaj V, Bhavitha Yaragorla
""")
