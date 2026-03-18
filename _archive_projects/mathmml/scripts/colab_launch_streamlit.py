"""Helper script to launch Streamlit app on Colab with ngrok."""

import subprocess
import sys
from pyngrok import ngrok


def main():
    """Launch Streamlit app via ngrok."""
    # Start ngrok tunnel
    public_url = ngrok.connect(8501)
    print(f"Public URL: {public_url}")
    
    # Launch Streamlit
    streamlit_path = sys.executable.replace("python", "streamlit")
    subprocess.run([
        streamlit_path, "run",
        "src/ui/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])


if __name__ == "__main__":
    main()

