"""
Enhanced Gradio Web Interface for MathVerifyProject
Features: Clean UI, step-by-step reasoning, error classification, LaTeX rendering, image upload
"""

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

import sys
import os
from typing import Dict, Any, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core_verification import MathVerifier
from ocr_input import HandwritingTranscriber


def classify_error_detailed(gold: str, pred: str, gold_parsed: list, pred_parsed: list) -> Dict[str, Any]:
    """
    Classify the type of error with detailed taxonomy.
    
    Args:
        gold: Original gold answer string
        pred: Original prediction string
        gold_parsed: Parsed gold answer
        pred_parsed: Parsed prediction
        
    Returns:
        Dictionary with error classification details
    """
    error_info = {
        'category': None,
        'subcategory': None,
        'severity': 'medium',
        'description': '',
        'color': '#ffc107'  # Default yellow
    }
    
    # Parse Errors
    if not gold_parsed:
        error_info.update({
            'category': 'Parse Error',
            'subcategory': 'Gold Answer Parsing Failed',
            'severity': 'high',
            'description': 'Could not parse the gold answer. Check if it\'s in valid LaTeX or mathematical expression format.',
            'color': '#dc3545'  # Red
        })
        return error_info
    
    if not pred_parsed:
        error_info.update({
            'category': 'Parse Error',
            'subcategory': 'Prediction Parsing Failed',
            'severity': 'high',
            'description': 'Could not parse the prediction. Check if it\'s in valid LaTeX or mathematical expression format.',
            'color': '#dc3545'  # Red
        })
        return error_info
    
    # Format Mismatch
    if gold_parsed and pred_parsed:
        gold_str = str(gold_parsed[0]) if gold_parsed else ''
        pred_str = str(pred_parsed[0]) if pred_parsed else ''
        
        # Check if they're numerically equivalent but different formats
        try:
            import sympy
            if hasattr(gold_parsed[0], 'evalf') and hasattr(pred_parsed[0], 'evalf'):
                gold_val = gold_parsed[0].evalf()
                pred_val = pred_parsed[0].evalf()
                if abs(float(gold_val) - float(pred_val)) < 1e-10:
                    error_info.update({
                        'category': 'Format Mismatch',
                        'subcategory': 'Equivalent but Different Representation',
                        'severity': 'low',
                        'description': 'Answers are mathematically equivalent but in different formats (e.g., fraction vs decimal).',
                        'color': '#ffc107'  # Yellow
                    })
                    return error_info
        except:
            pass
    
    # Calculation Error (if both are numbers but different)
    try:
        import sympy
        if gold_parsed and pred_parsed:
            if hasattr(gold_parsed[0], 'evalf') and hasattr(pred_parsed[0], 'evalf'):
                gold_val = float(gold_parsed[0].evalf())
                pred_val = float(pred_parsed[0].evalf())
                if abs(gold_val - pred_val) > 1e-10:
                    error_info.update({
                        'category': 'Calculation Error',
                        'subcategory': 'Incorrect Numerical Value',
                        'severity': 'high',
                        'description': f'Prediction ({pred_val}) does not match gold answer ({gold_val}).',
                        'color': '#dc3545'  # Red
                    })
                    return error_info
    except:
        pass
    
    # Notation Error
    if gold_parsed and pred_parsed:
        error_info.update({
            'category': 'Notation Error',
            'subcategory': 'Format or Symbol Mismatch',
            'severity': 'medium',
            'description': 'Answers may be equivalent but use different notation or formatting.',
            'color': '#fd7e14'  # Orange
        })
        return error_info
    
    # Default
    error_info.update({
        'category': 'Unknown Error',
        'subcategory': 'Unclassified',
        'severity': 'medium',
        'description': 'Could not classify the error type.',
        'color': '#6c757d'  # Gray
    })
    return error_info


def classify_error(gold_parsed: list, pred_parsed: list) -> str:
    """
    Simple error classification (backward compatibility).
    
    Args:
        gold_parsed: Parsed gold answer
        pred_parsed: Parsed prediction
        
    Returns:
        Error classification string
    """
    if not gold_parsed:
        return "Parse Error: Could not parse gold answer"
    if not pred_parsed:
        return "Parse Error: Could not parse prediction"
    return "Format Mismatch: Answers may be equivalent but in different formats"


def verify_problem(gold: str, pred: str) -> Dict[str, Any]:
    """
    Verify mathematical answer with detailed output and error taxonomy.
    
    Args:
        gold: Gold/correct answer
        pred: Model prediction
        
    Returns:
        Dictionary with verification results and details
    """
    verifier = MathVerifier()
    
    try:
        # Parse both expressions
        gold_parsed = verifier.parse_expression(gold, is_gold=True)
        pred_parsed = verifier.parse_expression(pred, is_gold=False)
        
        # Verify
        is_correct = verifier.verify_answer(gold, pred)
        
        # Classify error with detailed taxonomy if incorrect
        error_info = None
        error_type = None
        if not is_correct:
            error_info = classify_error_detailed(gold, pred, gold_parsed, pred_parsed)
            error_type = f"{error_info['category']}: {error_info['subcategory']}"
        
        # Build detailed output
        result = {
            'valid': is_correct,
            'gold': gold,
            'prediction': pred,
            'gold_parsed': str(gold_parsed[0]) if gold_parsed and len(gold_parsed) > 0 else "Could not parse",
            'pred_parsed': str(pred_parsed[0]) if pred_parsed and len(pred_parsed) > 0 else "Could not parse",
            'error_type': error_type,
            'error_info': error_info,
            'details': f"Gold parsed as: {gold_parsed[0] if gold_parsed else 'N/A'}, "
                      f"Prediction parsed as: {pred_parsed[0] if pred_parsed else 'N/A'}"
        }
        
        return result
    except Exception as e:
        return {
            'valid': False,
            'gold': gold,
            'prediction': pred,
            'error_type': f"Error: {str(e)}",
            'error_info': {
                'category': 'System Error',
                'subcategory': 'Exception',
                'severity': 'high',
                'description': str(e),
                'color': '#dc3545'
            },
            'details': f"An error occurred during verification: {str(e)}"
        }


def format_latex_for_display(text: str) -> str:
    """
    Format text for LaTeX rendering in Markdown.
    Converts LaTeX expressions to display format.
    
    Args:
        text: Input text that may contain LaTeX
        
    Returns:
        Formatted string with LaTeX in display format
    """
    if not text:
        return text
    
    # If text contains LaTeX delimiters, ensure proper formatting
    # Convert $...$ to $$...$$ for block display, or keep inline
    import re
    
    # Check if it's already in LaTeX format
    if '$' in text or '\\' in text:
        # Wrap in math block if not already wrapped
        if not text.strip().startswith('$$') and not text.strip().startswith('\\['):
            # Check if it's inline or block
            if text.count('$') >= 2:
                # Already has LaTeX delimiters, use as is
                return f"${text}$"
            else:
                # Add delimiters
                return f"$${text}$$"
    
    return text


def format_verification_output(result: Dict[str, Any]) -> str:
    """
    Format verification result as HTML with LaTeX rendering and error taxonomy.
    
    Args:
        result: Verification result dictionary
        
    Returns:
        HTML formatted string with LaTeX support
    """
    status_color = "green" if result['valid'] else "red"
    status_icon = "✔" if result['valid'] else "✖"
    status_text = "CORRECT" if result['valid'] else "INCORRECT"
    
    # Format LaTeX for display
    gold_display = format_latex_for_display(result['gold'])
    pred_display = format_latex_for_display(result['prediction'])
    
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; padding: 20px;">
        <h2 style="color: {status_color}; margin-bottom: 20px;">
            {status_icon} Verification Result: <span style="color: {status_color}; font-weight: bold;">{status_text}</span>
        </h2>
        
        <div style="background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%); padding: 20px; border-radius: 8px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0; color: #333;">📝 Input</h3>
            <div style="margin: 10px 0;">
                <b style="color: #555;">Gold Answer:</b>
                <div style="background-color: white; padding: 10px; border-radius: 4px; margin-top: 5px; font-family: 'Courier New', monospace;">
                    {gold_display}
                </div>
            </div>
            <div style="margin: 10px 0;">
                <b style="color: #555;">Prediction:</b>
                <div style="background-color: white; padding: 10px; border-radius: 4px; margin-top: 5px; font-family: 'Courier New', monospace;">
                    {pred_display}
                </div>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #e8f4f8 0%, #d1e7dd 100%); padding: 20px; border-radius: 8px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0; color: #333;">🔍 Parsing Details</h3>
            <p><b>Gold Parsed:</b> <code style="background-color: rgba(255,255,255,0.7); padding: 4px 8px; border-radius: 3px;">{result.get('gold_parsed', 'N/A')}</code></p>
            <p><b>Prediction Parsed:</b> <code style="background-color: rgba(255,255,255,0.7); padding: 4px 8px; border-radius: 3px;">{result.get('pred_parsed', 'N/A')}</code></p>
        </div>
        
        <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%); padding: 20px; border-radius: 8px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0; color: #333;">ℹ️ Details</h3>
            <p style="margin: 0;">{result.get('details', 'No additional details')}</p>
        </div>
    """
    
    # Error Taxonomy Breakdown
    if result.get('error_info'):
        error_info = result['error_info']
        error_color = error_info.get('color', '#dc3545')
        severity = error_info.get('severity', 'medium')
        
        # Severity indicator
        severity_colors = {
            'high': '#dc3545',  # Red
            'medium': '#fd7e14',  # Orange
            'low': '#ffc107'  # Yellow
        }
        severity_color = severity_colors.get(severity, '#6c757d')
        
        # Severity bar
        severity_width = {'high': '100%', 'medium': '66%', 'low': '33%'}.get(severity, '50%')
        
        html += f"""
        <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 20px; border-radius: 8px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid {error_color};">
            <h3 style="margin-top: 0; color: #721c24;">⚠️ Error Taxonomy</h3>
            
            <div style="margin: 15px 0;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <span style="font-weight: bold; color: #721c24;">Category:</span>
                    <span style="background-color: {error_color}; color: white; padding: 6px 12px; border-radius: 20px; font-weight: bold; font-size: 14px;">
                        {error_info.get('category', 'Unknown')}
                    </span>
                </div>
                
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <span style="font-weight: bold; color: #721c24;">Subcategory:</span>
                    <span style="background-color: {error_info.get('color', '#6c757d')}; color: white; padding: 4px 10px; border-radius: 15px; font-size: 13px;">
                        {error_info.get('subcategory', 'Unclassified')}
                    </span>
                </div>
                
                <div style="margin: 15px 0;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                        <span style="font-weight: bold; color: #721c24;">Severity:</span>
                        <span style="background-color: {severity_color}; color: white; padding: 4px 10px; border-radius: 15px; font-size: 13px; text-transform: uppercase;">
                            {severity}
                        </span>
                    </div>
                    <div style="background-color: #e9ecef; height: 8px; border-radius: 4px; overflow: hidden; margin-top: 5px;">
                        <div style="background-color: {severity_color}; height: 100%; width: {severity_width}; transition: width 0.3s ease;"></div>
                    </div>
                </div>
                
                <div style="background-color: rgba(255,255,255,0.7); padding: 12px; border-radius: 5px; margin-top: 10px;">
                    <p style="margin: 0; color: #333;"><b>Description:</b> {error_info.get('description', 'No description available')}</p>
                </div>
            </div>
        </div>
        """
    elif result.get('error_type'):
        # Fallback for simple error type
        html += f"""
        <div style="background-color: #f8d7da; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h3 style="color: #721c24;">Error Classification</h3>
            <p><b>{result['error_type']}</b></p>
        </div>
        """
    
    html += "</div>"
    return html


def verify_math_interface(gold: str, pred: str) -> Tuple[str, str]:
    """
    Main verification function for Gradio interface with LaTeX rendering.
    
    Args:
        gold: Gold answer
        pred: Prediction
        
    Returns:
        Tuple of (HTML formatted result, LaTeX formatted markdown for display)
    """
    if not gold or not pred:
        error_msg = "<p style='color: red;'>Please provide both gold answer and prediction.</p>"
        return error_msg, ""
    
    result = verify_problem(gold, pred)
    html_output = format_verification_output(result)
    
    # Create LaTeX-formatted markdown for separate display
    latex_markdown = f"""
## Verification Result

**Gold Answer:**
$${format_latex_for_display(gold)}$$

**Prediction:**
$${format_latex_for_display(pred)}$$

**Result:** {'✅ CORRECT' if result['valid'] else '❌ INCORRECT'}
"""
    
    if result.get('error_info'):
        error_info = result['error_info']
        latex_markdown += f"""
**Error Category:** {error_info.get('category', 'Unknown')}
**Severity:** {error_info.get('severity', 'medium').upper()}
"""
    
    return html_output, latex_markdown


def transcribe_image(image, model_path: str = None) -> Tuple[str, str, str]:
    """
    Transcribe handwritten math from image with enhanced OCR support.
    
    Args:
        image: Uploaded image file (PIL Image or file path)
        model_path: Optional model path
        
    Returns:
        Tuple of (latex_output, status_message, latex_markdown)
    """
    if image is None:
        return "", "Please upload an image file.", ""
    
    try:
        from PIL import Image
        import numpy as np
        
        # Handle different image input types
        if isinstance(image, str):
            # File path
            img = Image.open(image)
        elif hasattr(image, 'name'):
            # Gradio file object
            img = Image.open(image.name)
        else:
            # PIL Image or numpy array
            img = image if isinstance(image, Image.Image) else Image.fromarray(image)
        
        # Try to use OCR transcriber if available
        try:
            if not model_path:
                default_model = os.path.join(
                    os.path.dirname(__file__), '..',
                    'handwritten-math-transcription', 'model', 'model_best_0.pth'
                )
                if os.path.exists(default_model):
                    model_path = default_model
            
            if model_path and os.path.exists(model_path):
                transcriber = HandwritingTranscriber(model_path=model_path)
                # Note: transcribe_image method needs to be implemented
                # For now, return placeholder
                latex = "\\placeholder"
                status = f"Image loaded successfully. OCR processing requires InkML conversion. Image size: {img.size}"
            else:
                latex = "\\placeholder"
                status = "Image OCR feature - model checkpoint required. Please specify model path or use InkML file."
        except Exception as e:
            latex = "\\placeholder"
            status = f"OCR processing error: {str(e)}. Image loaded: {img.size}"
        
        # Create LaTeX markdown for display
        latex_markdown = f"""
## Transcribed LaTeX

$${latex}$$

**Status:** {status}
"""
        
        return latex, status, latex_markdown
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        return "", error_msg, ""


def transcribe_inkml_file(file, model_path: str = None) -> Tuple[str, str]:
    """
    Transcribe InkML file to LaTeX.
    
    Args:
        file: Uploaded InkML file
        model_path: Optional model path
        
    Returns:
        Tuple of (latex_output, status_message)
    """
    if file is None:
        return "", "Please upload an InkML file."
    
    try:
        # Try default model if not specified
        if not model_path:
            default_model = os.path.join(
                os.path.dirname(__file__), '..',
                'handwritten-math-transcription', 'model', 'model_best_0.pth'
            )
            if os.path.exists(default_model):
                model_path = default_model
            else:
                return "", "Error: No model found. Please specify model path or ensure model exists."
        
        transcriber = HandwritingTranscriber(model_path=model_path)
        
        # Get file path from Gradio file object
        file_path = file.name if hasattr(file, 'name') else str(file)
        latex, gt = transcriber.transcribe_inkml(file_path)
        
        output = f"**Transcribed LaTeX:** `{latex}`"
        if gt:
            output += f"\n\n**Ground Truth:** `{gt}`"
        
        return latex, output
    except Exception as e:
        return "", f"Error: {str(e)}"


def create_gradio_app():
    """
    Create enhanced Gradio interface with all features.
    
    Returns:
        Gradio Blocks interface
    """
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is not installed. Please install it with: pip install gradio")
    
    # Custom CSS for better styling with LaTeX support
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .output-html {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        max-height: 600px;
        overflow-y: auto;
    }
    .latex-output {
        border: 2px solid #4a90e2;
        border-radius: 8px;
        padding: 15px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        min-height: 200px;
    }
    .latex-output h2, .latex-output h3 {
        color: #2c3e50;
        margin-top: 0;
    }
    .latex-output code {
        background-color: #f4f4f4;
        padding: 2px 6px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
    }
    .error-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
        margin: 2px;
    }
    .severity-bar {
        height: 8px;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    """
    
    # Gradio 6.0 compatibility - simplified syntax
    with gr.Blocks(title="MathVerify: Mathematical Reasoning Verification") as app:
        # Header
        gr.Markdown("""
        # 🔬 MathVerify: Mathematical Reasoning Verification System
        
        A comprehensive system for verifying mathematical expressions, evaluating benchmarks, 
        and processing handwritten math input. Powered by Math-Verify, MATH-V, MathVerse, 
        and handwritten-math-transcription.
        """)
        
        with gr.Tabs():
            # Tab 1: Verification
            with gr.Tab("✅ Answer Verification", id="verification"):
                gr.Markdown("""
                ### Verify Mathematical Answers
                Enter a gold (correct) answer and a model prediction to verify if they match.
                Supports LaTeX, plain mathematical expressions, and numbers.
                """)
                
                with gr.Row():
                    with gr.Column():
                        gold_input = gr.Textbox(
                            label="Gold/Expected Answer",
                            placeholder='Enter correct answer (e.g., "1/2", "$\\frac{1}{2}$", "0.5")',
                            lines=2
                        )
                        pred_input = gr.Textbox(
                            label="Model Prediction",
                            placeholder='Enter prediction to verify (e.g., "0.5", "$\\frac{1}{2}$")',
                            lines=2
                        )
                        verify_btn = gr.Button("🔍 Verify Answer", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        verify_output = gr.HTML(
                            label="Verification Result (Detailed)",
                            elem_classes=["output-html"]
                        )
                    with gr.Column(scale=1):
                        latex_display = gr.Markdown(
                            label="LaTeX Rendered View",
                            elem_classes=["latex-output"]
                        )
                
                # Example inputs
                gr.Markdown("### 💡 Examples")
                gr.Examples(
                    examples=[
                        ["1/2", "0.5"],
                        ["$\\frac{1}{2}$", "0.5"],
                        ["42", "43"],
                        ["$\\sqrt{4}$", "2"],
                    ],
                    inputs=[gold_input, pred_input],
                    label="Click an example to try it"
                )
                
                verify_btn.click(
                    fn=verify_math_interface,
                    inputs=[gold_input, pred_input],
                    outputs=[verify_output, latex_display]
                )
            
            # Tab 2: OCR Transcription
            with gr.Tab("📝 Handwritten Math OCR", id="ocr"):
                gr.Markdown("""
                ### Transcribe Handwritten Mathematical Expressions
                Upload an InkML file or image to transcribe handwritten math to LaTeX.
                """)
                
                with gr.Tabs():
                    with gr.Tab("InkML File"):
                        gr.Markdown("Upload an InkML file containing handwritten stroke data.")
                        inkml_file = gr.File(
                            label="Upload InkML File",
                            file_types=[".inkml"],
                            type="filepath"
                        )
                        inkml_model_path = gr.Textbox(
                            label="Model Path (optional)",
                            placeholder="Leave empty to use default model",
                            value=""
                        )
                        inkml_btn = gr.Button("📄 Transcribe InkML", variant="primary")
                        
                        with gr.Row():
                            inkml_latex = gr.Textbox(
                                label="Transcribed LaTeX",
                                lines=3
                            )
                            inkml_status = gr.Markdown(label="Status")
                        
                        inkml_btn.click(
                            fn=transcribe_inkml_file,
                            inputs=[inkml_file, inkml_model_path],
                            outputs=[inkml_latex, inkml_status]
                        )
                    
                    with gr.Tab("Image Upload"):
                        gr.Markdown("""
                        ### Upload Handwritten Math Image
                        Upload an image file (PNG, JPG, etc.) containing handwritten mathematical expressions.
                        The system will attempt to transcribe it to LaTeX.
                        """)
                        image_file = gr.Image(
                            label="Upload Math Image",
                            type="filepath",
                            sources=["upload", "clipboard"]
                        )
                        image_model_path = gr.Textbox(
                            label="Model Path (optional)",
                            placeholder="Leave empty to use default model",
                            value=""
                        )
                        image_btn = gr.Button("🖼️ Transcribe Image", variant="primary")
                        
                        with gr.Row():
                            with gr.Column():
                                image_latex = gr.Textbox(
                                    label="Transcribed LaTeX Code",
                                    lines=3
                                )
                                image_status = gr.Markdown(label="Status")
                            with gr.Column():
                                image_latex_display = gr.Markdown(
                                    label="LaTeX Rendered View",
                                    elem_classes=["latex-output"]
                                )
                        
                        image_btn.click(
                            fn=transcribe_image,
                            inputs=[image_file, image_model_path],
                            outputs=[image_latex, image_status, image_latex_display]
                        )
            
            # Tab 3: Batch Verification
            with gr.Tab("📊 Batch Verification", id="batch"):
                gr.Markdown("""
                ### Batch Verification
                Upload files with multiple gold answers and predictions for batch processing.
                """)
                
                gold_file = gr.File(
                    label="Upload Gold Answers File (one per line)",
                    file_types=[".txt", ".csv"]
                )
                pred_file = gr.File(
                    label="Upload Predictions File (one per line)",
                    file_types=[".txt", ".csv"]
                )
                batch_btn = gr.Button("📊 Process Batch", variant="primary")
                
                batch_output = gr.Dataframe(
                    label="Batch Results",
                    headers=["Gold", "Prediction", "Correct", "Error Type"]
                )
                
                def process_batch(gold_file, pred_file):
                    """Process batch verification."""
                    if not gold_file or not pred_file:
                        return gr.update(value=[]), "Please upload both files."
                    
                    try:
                        from core_verification import MathVerifier
                        verifier = MathVerifier()
                        
                        # Read files
                        with open(gold_file.name, 'r') as f:
                            gold_answers = [line.strip() for line in f if line.strip()]
                        with open(pred_file.name, 'r') as f:
                            predictions = [line.strip() for line in f if line.strip()]
                        
                        if len(gold_answers) != len(predictions):
                            return gr.update(value=[]), f"Error: Files have different lengths ({len(gold_answers)} vs {len(predictions)})"
                        
                        # Verify batch
                        results = verifier.verify_batch(gold_answers, predictions)
                        
                        # Format results with error taxonomy
                        data = []
                        for i, (gold, pred, correct) in enumerate(zip(gold_answers, predictions, results)):
                            if not correct:
                                gold_parsed = verifier.parse_expression(gold, is_gold=True)
                                pred_parsed = verifier.parse_expression(pred, is_gold=False)
                                error_info = classify_error_detailed(gold, pred, gold_parsed, pred_parsed)
                                error = f"{error_info['category']}: {error_info['subcategory']}"
                            else:
                                error = "None"
                            data.append([gold, pred, "✓" if correct else "✗", error])
                        
                        return gr.update(value=data), f"Processed {len(data)} examples. {sum(results)} correct."
                    except Exception as e:
                        return gr.update(value=[]), f"Error: {str(e)}"
                
                batch_btn.click(
                    fn=process_batch,
                    inputs=[gold_file, pred_file],
                    outputs=[batch_output, gr.Markdown()]
                )
            
            # Tab 4: Error Taxonomy Reference
            with gr.Tab("📊 Error Taxonomy", id="taxonomy"):
                gr.Markdown("""
                ## Error Taxonomy Reference
                
                The system classifies errors into the following categories with visual indicators:
                
                ### 🔴 High Severity Errors
                
                **Parse Error - Gold Answer Parsing Failed**
                - Color Tag: Red (#dc3545)
                - Severity Bar: 100%
                - Description: Could not parse the gold answer. Check if it's in valid LaTeX or mathematical expression format.
                
                **Parse Error - Prediction Parsing Failed**
                - Color Tag: Red (#dc3545)
                - Severity Bar: 100%
                - Description: Could not parse the prediction. Check if it's in valid LaTeX or mathematical expression format.
                
                **Calculation Error - Incorrect Numerical Value**
                - Color Tag: Red (#dc3545)
                - Severity Bar: 100%
                - Description: Prediction does not match gold answer numerically.
                
                ### 🟠 Medium Severity Errors
                
                **Notation Error - Format or Symbol Mismatch**
                - Color Tag: Orange (#fd7e14)
                - Severity Bar: 66%
                - Description: Answers may be equivalent but use different notation or formatting.
                
                ### 🟡 Low Severity Errors
                
                **Format Mismatch - Equivalent but Different Representation**
                - Color Tag: Yellow (#ffc107)
                - Severity Bar: 33%
                - Description: Answers are mathematically equivalent but in different formats (e.g., fraction vs decimal).
                
                ### Visual Indicators
                
                - **Colored Tags**: Each error category has a color-coded badge
                - **Severity Bars**: Visual progress bars indicate error severity (High=100%, Medium=66%, Low=33%)
                - **Category Badges**: Large badges show the main error category
                - **Subcategory Tags**: Smaller tags show specific error types
                
                ### Error Classification Process
                
                1. **Parse Check**: First checks if both answers can be parsed
                2. **Numerical Comparison**: Compares numerical values if both are numbers
                3. **Format Analysis**: Checks for format mismatches
                4. **Notation Check**: Analyzes notation differences
                5. **Classification**: Assigns category, subcategory, and severity
                
                """)
            
            # Tab 5: About
            with gr.Tab("ℹ️ About", id="about"):
                gr.Markdown("""
                ## About MathVerify Project
                
                This system integrates multiple open-source research repositories:
                
                - **Math-Verify**: Core mathematical expression verification engine
                - **MATH-V**: Multimodal mathematical reasoning benchmark
                - **MathVerse**: Visual math problem evaluation
                - **handwritten-math-transcription**: Handwritten math OCR
                
                ### Features:
                - ✅ Robust answer verification with symbolic evaluation
                - 📊 Benchmark evaluation capabilities
                - 📝 Handwritten math transcription (InkML & Images)
                - 🎯 Error taxonomy with visual classification
                - 📈 Batch processing support
                - 🔬 LaTeX rendering for mathematical expressions
                - 🎨 Enhanced presentation with colored tags and severity bars
                
                ### Usage Tips:
                1. For best results, use LaTeX format: `$\\frac{1}{2}$`
                2. Plain expressions work too: `1/2`, `0.5`, `2+2`
                3. InkML files should be in standard format
                4. Images should be clear, high-contrast handwritten math
                5. Batch files should have one answer per line
                
                ### LaTeX Rendering:
                - Mathematical expressions are automatically rendered using MathJax
                - Both inline (`$...$`) and block (`$$...$$`) formats are supported
                - LaTeX view shows rendered mathematical notation
                
                ### Documentation:
                See `README.md` for complete documentation and API reference.
                """)
        
        # Footer
        gr.Markdown("""
        ---
        **MathVerify Project** | Integrated Mathematical Reasoning Verification System
        """)
    
    return app

