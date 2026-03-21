import os
import shutil
import glob
from pathlib import Path

def print_action(action, detail):
    print(f"[{action}] {detail}")

def cleanup_repo(root_dir: str):
    root = Path(root_dir)
    
    # --- 1. Define Target Directories ---
    target_services = root / "services"
    target_config = root / "config"
    target_docs = root / "docs"
    target_scripts = root / "scripts"
    target_tests_integration = root / "tests" / "integration"
    
    for d in [target_services, target_config, target_docs, target_scripts, target_tests_integration]:
        os.makedirs(d, exist_ok=True)
        
    # --- 2. Move & Restructure Microservices -> services/ ---
    old_microservices = root / "microservices"
    if old_microservices.exists():
        service_names = ["input_receiver", "preprocessing", "ocr", "representation", "verification", "classifier", "reporting"]
        for svc in service_names:
            old_path = old_microservices / svc
            new_path = target_services / (svc if svc in ['input_receiver'] else f"{svc}_service")
            
            if old_path.exists():
                os.makedirs(new_path, exist_ok=True)
                # Move main.py -> app.py
                main_file = old_path / "main.py"
                if main_file.exists():
                    shutil.move(str(main_file), str(new_path / "app.py"))
                    print_action("RENAME", f"{svc}/main.py -> {new_path.name}/app.py")
                
                # Move Test files into tests/
                test_file = old_path / "test_main.py"
                if test_file.exists():
                    tests_dir = new_path / "tests"
                    os.makedirs(tests_dir, exist_ok=True)
                    shutil.move(str(test_file), str(tests_dir / f"test_{svc}.py"))
                    print_action("MOVE", f"{svc}/test_main.py -> {new_path.name}/tests/test_{svc}.py")
                    
                # Move Dockerfile and requirements
                for f in ["Dockerfile", "requirements.txt"]:
                    if (old_path / f).exists():
                        shutil.move(str(old_path / f), str(new_path / f))
                        print_action("MOVE", f"{svc}/{f} -> {new_path.name}/{f}")
                        
        print_action("DELETE", f"{old_microservices.name}/ (Empty after move)")
        shutil.rmtree(str(old_microservices), ignore_errors=True)
        
    # --- 3. Move Integration Tests ---
    old_tests = root / "tests" / "microservices"
    if old_tests.exists():
        for f in glob.glob(str(old_tests / "*.py")):
            shutil.move(f, str(target_tests_integration / Path(f).name))
            print_action("MOVE", f"{Path(f).name} -> tests/integration/")
        shutil.rmtree(str(old_tests), ignore_errors=True)

    # --- 4. Move Configs ---
    global_req = root / "requirements.txt"
    if global_req.exists():
        shutil.move(str(global_req), str(target_config / "requirements.txt"))
        print_action("MOVE", "requirements.txt -> config/")
        
    dc_new = root / "docker-compose_microservices.yml"
    if dc_new.exists():
        shutil.move(str(dc_new), str(target_config / "docker-compose.yml"))
        print_action("MOVE", "docker-compose_microservices.yml -> config/docker-compose.yml")
    else:
        dc_old = root / "docker-compose.yml"
        if dc_old.exists():
            shutil.move(str(dc_old), str(target_config / "docker-compose.yml"))
            print_action("MOVE", "docker-compose.yml -> config/docker-compose.yml")
            
    env_old = root / ".env.template"
    if env_old.exists():
        shutil.move(str(env_old), str(target_config / ".env"))
        print_action("MOVE", ".env.template -> config/.env")

    # --- 5. Clean up Docs ---
    # Assume README.md and potential architecture.md stay in docs. Move root README to docs.
    root_readme = root / "README.md"
    if root_readme.exists():
        shutil.move(str(root_readme), str(target_docs / "README.md"))
        print_action("MOVE", "README.md -> docs/")

    # --- 6. Deletions ---
    # Unnecessary folders
    folders_to_delete = [
        "backend", "frontend", "handwritten-math-transcription", 
        "external_resources", "datasets", "test_images"
    ]
    for d in folders_to_delete:
        p = root / d
        if p.exists() and p.is_dir():
            shutil.rmtree(str(p), ignore_errors=True)
            print_action("DELETE", f"Removed old monolithic/unused directory: {d}/")
            
    # Unnecessary root files
    files_to_delete = glob.glob(str(root / "*.csv")) + \
                      glob.glob(str(root / "*.json")) + \
                      glob.glob(str(root / "*.png")) + \
                      glob.glob(str(root / "*.bat")) + \
                      glob.glob(str(root / "*.ps1")) + \
                      [str(root / "Dockerfile")]
                      
    for f in files_to_delete:
        p = Path(f)
        if p.exists() and p.is_file():
            p.unlink()
            print_action("DELETE", f"Removed unused file: {p.name}")

    # Remove pyc and pycache
    for pyc in glob.glob(str(root / "**" / "*.pyc"), recursive=True):
        os.remove(pyc)
    for pycache in glob.glob(str(root / "**" / "__pycache__"), recursive=True):
        shutil.rmtree(pycache, ignore_errors=True)
    print_action("CLEAN", "Removed all .pyc and __pycache__ files.")

    # Remove other scripts in scripts/ except cleanup.py
    for s in glob.glob(str(target_scripts / "*.py")):
        if Path(s).name != "cleanup.py":
            os.remove(s)
            print_action("DELETE", f"Removed old script: {Path(s).name}")

if __name__ == "__main__":
    import sys
    # execute from script directory logic
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    print(f"Starting cleanup at: {project_root}\n")
    cleanup_repo(project_root)
    print("\n[SUCCESS] Cleanup and restructuring complete.")
