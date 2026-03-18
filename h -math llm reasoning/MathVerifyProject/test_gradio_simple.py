"""
Simple test to see if Gradio works at all
"""

import gradio as gr

def greet(name):
    return f"Hello {name}!"

# Create a simple interface
iface = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Name"),
    outputs=gr.Textbox(label="Greeting"),
    title="Simple Test"
)

print("=" * 60)
print("Testing Gradio - Simple Interface")
print("=" * 60)
print("\nLaunching simple test interface...")
print("If this works, you should see a URL like: http://127.0.0.1:7860")
print("\nPress Ctrl+C to stop")
print("=" * 60)

iface.launch(server_name="127.0.0.1", server_port=7860, share=False)

