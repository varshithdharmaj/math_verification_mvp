"""
Standalone Gradio Demo Script
Run this to launch the enhanced Gradio web interface

Usage:
    python demo_gradio.py
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from main_interface import create_gradio_app

if __name__ == "__main__":
    print("=" * 60)
    print("MathVerify - Enhanced Gradio Web Interface")
    print("=" * 60)
    print()
    print("Features:")
    print("  ✓ Clean web interface")
    print("  ✓ Step-by-step reasoning display")
    print("  ✓ Answer verification with error classification")
    print("  ✓ LaTeX rendering")
    print("  ✓ Image upload support (OCR)")
    print("  ✓ Batch verification")
    print()
    print("Launching web interface...")
    print("The interface will open in your browser.")
    print("Press Ctrl+C to stop the server.")
    print()
    
    try:
        app = create_gradio_app()
        app.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,
            share=False,  # Set to True for public link
            show_error=True
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install Gradio:")
        print("  pip install gradio")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user.")
    except Exception as e:
        print(f"\nError launching interface: {e}")
        import traceback
        traceback.print_exc()

