"""
Simple script to launch the Gradio interface
Run this to start the web interface
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("MathVerify - Launching Web Interface")
print("=" * 60)
print()

try:
    from main_interface.gradio_app import create_gradio_app, GRADIO_AVAILABLE
    
    if not GRADIO_AVAILABLE:
        print("ERROR: Gradio is not installed!")
        print("\nPlease install it with:")
        print("  pip install gradio")
        sys.exit(1)
    
    print("✓ Gradio is available")
    print("\nCreating interface...")
    
    app = create_gradio_app()
    
    print("✓ Interface created")
    print("\n" + "=" * 60)
    print("Starting server...")
    print("=" * 60)
    print("\nThe interface will be available at:")
    print("  → http://localhost:7860")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Launch with clear settings
    app.launch(
        server_name="127.0.0.1",  # Local only
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )
    
except ImportError as e:
    print(f"ERROR: {e}")
    print("\nPlease install Gradio:")
    print("  pip install gradio")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

