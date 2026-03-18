"""
Upload MVM2 Math Verification System to Hugging Face Spaces.
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.environ.get("HF_TOKEN")
SPACE_NAME = "mvm2-math-verification"
PROJECT_DIR = Path(r"c:\Users\Varshith Dharmaj\Downloads\major\math_verification_mvp")

if not HF_TOKEN:
    print("ERROR: Please set HF_TOKEN environment variable.")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)
me = api.whoami()
username = me["name"]
repo_id = f"{username}/{SPACE_NAME}"
print(f"Uploading to Space: https://huggingface.co/spaces/{repo_id}")

# Create with gradio first (workaround for API bug), then switch via README
try:
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        token=HF_TOKEN,
        exist_ok=True,
        private=False,
    )
    print(f"Space exists/created.")
except Exception as e:
    print(f"Space creation note: {e}")

print(f"\nUploading project folder...")
try:
    api.upload_folder(
        folder_path=str(PROJECT_DIR),
        repo_id=repo_id,
        repo_type="space",
        token=HF_TOKEN,
        ignore_patterns=[
            "*.log", "*.pyc", "*.pyo", "*.csv", "*.h5", "*.npy", "*.npz", "*.pkl",
            ".git/**", "__pycache__/**", ".pytest_cache/**", ".venv/**", "venv/**", "services/**/venv/**", ".benchmarks/**",
            "external_resources/**", "handwritten-math-transcription/**", "node_modules/**"
        ]
    )
    print(f"\n🚀 Deployment complete!")
    print(f"App URL: https://huggingface.co/spaces/{repo_id}")
    print(f"\n📌 Add GEMINI_API_KEY Secret at:")
    print(f"   https://huggingface.co/spaces/{repo_id}/settings")
except Exception as e:
    print(f"Upload failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)
