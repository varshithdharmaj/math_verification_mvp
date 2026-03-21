import subprocess
import time
import sys
import os
import signal

services = [
    ("services/input_receiver", 8000),
    ("services/preprocessing_service", 8001),
    ("services/ocr_service", 8002),
    ("services/representation_service", 8003),
    ("services/verification_service", 8004),
    ("services/classifier_service", 8005),
    ("services/reporting_service", 8006),
]

processes = []

def start_services():
    print("Starting all microservices locally...")
    for path, port in services:
        print(f"Starting {os.path.basename(path)} on port {port}...")
        env = os.environ.copy()
        
        # Use python -m uvicorn app:app --port 
        # (Assuming app.py is the main file name due to the recent cleanup script!)
        p = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.0", "--port", str(port)],
            cwd=path,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(p)

    print("Giving services 5 seconds to boot up...")
    time.sleep(5)

def stop_services():
    print("Stopping all microservices...")
    for p in processes:
        p.terminate()
        p.wait()
    print("All services stopped.")

if __name__ == "__main__":
    try:
        start_services()
        print("Running Benchmark...")
        subprocess.run([sys.executable, "scripts/benchmark_gsm8k.py"])
    finally:
        stop_services()
