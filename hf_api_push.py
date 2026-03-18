import os
from huggingface_hub import HfApi

HF_TOKEN = "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"
REPO_ID = "sayian99/mvm2-math-verification"

api = HfApi(token=HF_TOKEN)

local_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "math_verification_mvp", "services", "dashboard", "app.py"))

print(f"Uploading {local_app_path} to {REPO_ID} as app.py...")

try:
    api.upload_file(
        path_or_fileobj=local_app_path,
        path_in_repo="app.py",
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="Fix: Implemented Multimodal Image Upload passing via requests to OCR"
    )
    print("✅ Successfully patched app.py on Hugging Face!")
except Exception as e:
    print(f"❌ Upload failed: {e}")
