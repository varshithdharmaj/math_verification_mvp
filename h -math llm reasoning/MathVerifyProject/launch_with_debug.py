"""
Launch interface with detailed debugging
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("MathVerify - Debug Launch")
print("=" * 60)
print()

# Step 1: Check Gradio
print("Step 1: Checking Gradio...")
try:
    import gradio as gr
    print(f"  ✓ Gradio imported (version: {gr.__version__})")
except Exception as e:
    print(f"  ✗ Gradio import failed: {e}")
    sys.exit(1)

# Step 2: Check our modules
print("\nStep 2: Checking modules...")
try:
    from main_interface.gradio_app import create_gradio_app, GRADIO_AVAILABLE
    print(f"  ✓ Module imported (GRADIO_AVAILABLE: {GRADIO_AVAILABLE})")
except Exception as e:
    print(f"  ✗ Module import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Create app
print("\nStep 3: Creating app...")
try:
    app = create_gradio_app()
    print("  ✓ App created successfully")
except Exception as e:
    print(f"  ✗ App creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Launch
print("\nStep 4: Launching server...")
print("=" * 60)
print("\nIMPORTANT: Look for a line that says:")
print("  'Running on local URL: http://127.0.0.1:7860'")
print("\nThen open that URL in your browser!")
print("=" * 60)
print()

try:
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        inbrowser=False  # Don't auto-open browser
    )
except KeyboardInterrupt:
    print("\n\nServer stopped by user.")
except Exception as e:
    print(f"\n✗ Launch failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

