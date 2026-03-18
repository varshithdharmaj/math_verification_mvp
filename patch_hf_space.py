import os
import shutil
import subprocess

HF_REPO_URL = "https://huggingface.co/spaces/sayian99/mvm2-math-verification"
HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("Error: HF_TOKEN environment variable is not set. Cannot push to Hugging Face.")
    exit(1)

# Clone the repo temporarily to push the fix
repo_dir = "hf_repo_temp"
if os.path.exists(repo_dir):
    shutil.rmtree(repo_dir)

print(f"Cloning {HF_REPO_URL} into {repo_dir}...")
clone_cmd = f"git clone https://sayian99:{HF_TOKEN}@huggingface.co/spaces/sayian99/mvm2-math-verification {repo_dir}"
subprocess.run(clone_cmd, shell=True, check=True)

# Important: Copy only the updated app.py files.
print("Copying patched app.py to repo...")
src_app = os.path.abspath(os.path.join(os.path.dirname(__file__), "math_verification_mvp", "services", "dashboard", "app.py"))
dest_app = os.path.join(repo_dir, "app.py")
shutil.copy2(src_app, dest_app)

# The OCR microservice logic needs to be integrated into the main HuggingFace app.py 
# because HF Spaces only run a single app.py entrypoint by default. 
# We'll rely on the existing deploy_to_hf.py structure where we merged it.

print("Committing and pushing changes...")
subprocess.run("git config user.name 'Antigravity Agent'", cwd=repo_dir, shell=True)
subprocess.run("git config user.email 'agent@antigravity.google'", cwd=repo_dir, shell=True)

subprocess.run("git add app.py", cwd=repo_dir, shell=True)
subprocess.run("git commit -m 'Fix: Implement multimodal image upload requests integration'", cwd=repo_dir, shell=True)

push_cmd = f"git push https://sayian99:{HF_TOKEN}@huggingface.co/spaces/sayian99/mvm2-math-verification main"
result = subprocess.run(push_cmd, cwd=repo_dir, shell=True)

if result.returncode == 0:
    print("✅ Successfully patched the Hugging Face Space image uploader!")
else:
    print("❌ Push failed.")
    
# Cleanup
shutil.rmtree(repo_dir)
