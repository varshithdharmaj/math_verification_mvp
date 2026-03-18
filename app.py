import gradio as gr
import os
import time
import cv2
import numpy as np
from PIL import Image
import tempfile
import json

# Import consolidated modules
from ocr_module import MVM2OCREngine
from reasoning_engine import run_agent_orchestrator
from verification_service import calculate_symbolic_score
from consensus_fusion import evaluate_consensus
from report_module import generate_mvm2_report, export_to_pdf
from image_enhancing import ImageEnhancer
from evaluation_module import evaluate_from_labels

# Initialize Engines (Pix2Text OCR + handwritten enhancer)
ocr_engine = MVM2OCREngine()
enhancer = ImageEnhancer(sigma=1.2)


def process_mvm2_pipeline(image, auto_enhance):
    """
    Full MVM² pipeline used by the primary Gradio button:
    1) Optional handwritten-math enhancement (CLAHE + Gaussian blur).
    2) OCR via Pix2Text (or simulated backend).
    3) Multi-agent reasoning.
    4) Adaptive consensus fusion.
    5) Markdown + PDF report generation.
    """
    if image is None:
        return "Please upload an image.", None, None

    # 1. Preprocessing: notebook-friendly enhancement before OCR.
    if auto_enhance:
        enhanced_img_np, meta = enhancer.enhance(image)
        # Save temp enhanced image for OCR
        temp_img_path = os.path.join(tempfile.gettempdir(), "enhanced_input.png")
        cv2.imwrite(temp_img_path, enhanced_img_np)
    else:
        # Save original PIL image directly for OCR
        temp_img_path = os.path.join(tempfile.gettempdir(), "original_input.png")
        image.save(temp_img_path)
        meta = {"metrics": {"initial_contrast": 0}}

    # 2. OCR Extraction
    ocr_results = ocr_engine.process_image(temp_img_path)
    latex_text = ocr_results["latex_output"]
    ocr_conf = ocr_results["weighted_confidence"]

    if "No math detected" in latex_text:
        return f"OCR Failure: {latex_text}", None, None

    # 3. Multi-Agent Reasoning
    agent_responses = run_agent_orchestrator(latex_text)

    # 4. Consensus Fusion
    consensus_result = evaluate_consensus(agent_responses, ocr_confidence=ocr_conf)

    # 5. Report Generation
    reports = generate_mvm2_report(consensus_result, latex_text, ocr_conf)
    md_report = reports["markdown"]
    json_report = json.loads(reports["json"])

    # 6. Export to PDF
    pdf_path = os.path.join(tempfile.gettempdir(), f"MVM2_Report_{reports['report_id']}.pdf")
    export_to_pdf(json_report, pdf_path)

    return md_report, pdf_path, latex_text


def fresnel_integral_self_test():
    """
    Deployment-time self-test using a canonical Fresnel-style integral.

    This bypasses OCR and directly feeds a LaTeX integral into the
    reasoning + consensus stack so that the Space can be verified
    even without uploading an image.
    """
    # Canonical Fresnel-style test integral (consistent with LLMAgent simulator).
    latex_text = "\\int_{0}^{\\pi} \\sin(x^{2}) \\, dx"
    ocr_conf = 0.95  # Synthetic high OCR confidence for the self-test path.

    # 3. Multi-Agent Reasoning
    agent_responses = run_agent_orchestrator(latex_text)

    # 4. Consensus Fusion
    consensus_result = evaluate_consensus(agent_responses, ocr_confidence=ocr_conf)

    # 5. Report Generation
    reports = generate_mvm2_report(consensus_result, latex_text, ocr_conf)
    md_report = reports["markdown"]
    json_report = json.loads(reports["json"])

    # 6. Export to PDF
    pdf_path = os.path.join(tempfile.gettempdir(), f"MVM2_Report_{reports['report_id']}_fresnel.pdf")
    export_to_pdf(json_report, pdf_path)

    return md_report, pdf_path, latex_text


def run_offline_evaluation(y_true_str: str, y_pred_str: str) -> str:
    """
    Lightweight bridge into `evaluation_module` for HF Spaces.

    Users supply comma-separated labels for:
    - y_true: ground-truth final answers.
    - y_pred: MVM² consensus outputs.

    This function parses the lists, calls `evaluate_from_labels`, and
    returns a Markdown summary that can be rendered directly in Gradio.
    """
    try:
        # Parse comma-separated labels; keep them as strings to support
        # symbolic/numeric answers (e.g., LaTeX, "42", "0.779", etc.).
        y_true = [s.strip() for s in y_true_str.split(",") if s.strip()]
        y_pred = [s.strip() for s in y_pred_str.split(",") if s.strip()]

        if not y_true or not y_pred:
            return "Please provide non-empty comma-separated lists for both ground truth and predictions."

        metrics, md_report = evaluate_from_labels(
            y_true=y_true,
            y_pred=y_pred,
            save_json_path=None,
        )
        return md_report
    except Exception as e:
        return f"Evaluation error: {e}"

# Custom CSS for Professional Educational Styling
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}
.mvm2-header {
    text-align: center;
    background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.report-area {
    background-color: #f9f9f9;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #ddd;
}
"""

with gr.Blocks(css=custom_css, title="MVM²: Math Verification & Multi-Signal Consensus") as demo:
    gr.Markdown(
        """
        <div class="mvm2-header">
            <h1>🧠 MVM²: Neuro-Symbolic Math Verification</h1>
            <p>Adaptive Multi-Signal Consensus for Handwritten Mathematical Equation Verification</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload Handwritten Math (Student Notebook)")
            enhance_toggle = gr.Checkbox(
                label="Auto-Enhance for Handwritten Math (CLAHE + Gaussian Blur)", value=True
            )
            run_btn = gr.Button("🚀 Run Multimodal Verification", variant="primary")
            fresnel_btn = gr.Button("🧪 Run Fresnel Integral Self-Test", variant="secondary")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📄 Explainable Diagnostic Report"):
                    report_output = gr.Markdown(label="Verification Report", elem_classes="report-area")
                    download_btn = gr.File(label="Download PDF Report")
                with gr.TabItem("🔍 Raw OCR Extraction"):
                    ocr_output = gr.Textbox(label="Transcribed LaTeX", interactive=False)
                with gr.TabItem("📊 Offline Evaluation (HF `evaluate`)"):
                    eval_y_true = gr.Textbox(
                        label="Ground-Truth Labels (comma-separated)",
                        placeholder="e.g., 1, 0, 1, 1, 0",
                    )
                    eval_y_pred = gr.Textbox(
                        label="Predicted Labels (comma-separated)",
                        placeholder="e.g., 1, 0, 0, 1, 0",
                    )
                    eval_button = gr.Button("Run Evaluation")
                    eval_report = gr.Markdown(label="Evaluation Report")

    gr.Markdown(
        """
        ### 🧪 Project MVM² Capabilities:
        - **Robust OCR**: Pix2Text handles complex LaTeX commands and handwritten strokes.
        - **Neuro-Symbolic Fusion**: Weighted $Score_j$ formula combines LLM logic with SymPy validation.
        - **Hallucination Detection**: Automatically flags agents with low consistency scores (< 0.7).
        """
    )

    run_btn.click(
        fn=process_mvm2_pipeline,
        inputs=[input_img, enhance_toggle],
        outputs=[report_output, download_btn, ocr_output],
    )

    fresnel_btn.click(
        fn=fresnel_integral_self_test,
        inputs=[],
        outputs=[report_output, download_btn, ocr_output],
    )

    eval_button.click(
        fn=run_offline_evaluation,
        inputs=[eval_y_true, eval_y_pred],
        outputs=[eval_report],
    )

if __name__ == "__main__":
    demo.launch()
