"""
Minimal launch script - simplest possible
"""

print("Starting...")

import gradio as gr
from core_verification import MathVerifier

def verify(gold, pred):
    verifier = MathVerifier()
    result = verifier.verify_answer(gold, pred, return_details=True)
    if result['valid']:
        return f"✅ CORRECT\n\nGold: {result['gold']}\nPrediction: {result['prediction']}"
    else:
        return f"❌ INCORRECT\n\nGold: {result['gold']}\nPrediction: {result['prediction']}\nError: {result.get('error_type', 'Unknown')}"

# Create simple interface
iface = gr.Interface(
    fn=verify,
    inputs=[
        gr.Textbox(label="Gold Answer", placeholder="e.g., 1/2"),
        gr.Textbox(label="Prediction", placeholder="e.g., 0.5")
    ],
    outputs=gr.Textbox(label="Result"),
    title="MathVerify - Simple Test",
    description="Enter gold answer and prediction to verify"
)

print("\n" + "="*60)
print("Launching interface...")
print("="*60)
print("\nThe interface will be available at:")
print("  → http://127.0.0.1:7860")
print("\nCopy and paste that URL into your browser!")
print("="*60)
print()

iface.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    show_error=True
)

