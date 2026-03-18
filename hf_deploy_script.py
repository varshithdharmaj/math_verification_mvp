from huggingface_hub import HfApi
import os
import sys

"""
Simple deployment helper for the MVM² Hugging Face Space.

Usage (PowerShell on Windows):

    $env:HF_TOKEN = "<YOUR_WRITE_TOKEN_HERE>"
    cd "C:\Users\Varshith Dharmaj\Downloads\major"
    python hf_deploy_script.py

Requirements:
- The Space `sayian99/mvm2-math-verification` must already exist.
- HF_TOKEN must be a valid WRITE token for the `sayian99` account.
"""

repo_id = "sayian99/mvm2-math-verification"
folder_path = "."  # Current directory (root containing app.py and modules)

# Read API token from environment for safety.
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN environment variable is not set.")
    print("Please set HF_TOKEN to a valid Hugging Face WRITE token and retry.")
    sys.exit(1)

# List of files to upload from the root folder.
# Add any new modules here as the project evolves.
files_to_upload = [
    "app.py",
    "ocr_module.py",
    "reasoning_engine.py",
    "verification_service.py",
    "consensus_fusion.py",
    "report_module.py",
    "image_enhancing.py",
    "llm_agent.py",
    "evaluation_module.py",
    "requirements.txt",
]

api = HfApi(token=HF_TOKEN)

print(f"Starting upload to HF Space: {repo_id}")

for file_name in files_to_upload:
    local_path = os.path.join(folder_path, file_name)
    if os.path.exists(local_path):
        print(f"Uploading {file_name}...")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="space",
            )
            print(f"Successfully uploaded {file_name}.")
        except Exception as e:
            print(f"Failed to upload {file_name}: {e}")
    else:
        print(f"Warning: {file_name} not found in {folder_path}.")

print("Deployment complete.")

