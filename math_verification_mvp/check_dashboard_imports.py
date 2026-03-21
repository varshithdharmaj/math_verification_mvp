import sys
import os

# Mimic dashboard/app.py path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, PROJECT_ROOT)
SERVICES_PATH = os.path.join(PROJECT_ROOT, "services")
sys.path.insert(0, SERVICES_PATH)

print(f"DEBUG: sys.path: {sys.path[:3]}")

try:
    print("Trying from core_engine.pipeline_streamer...")
    from core_engine.pipeline_streamer import run_neurosymbolic_pipeline_stream
    print("SUCCESS")
except ImportError as e:
    print(f"FAILED core_engine.pipeline_streamer: {e}")

try:
    print("Trying from services.core_engine.pipeline_streamer...")
    from services.core_engine.pipeline_streamer import run_neurosymbolic_pipeline_stream
    print("SUCCESS")
except ImportError as e:
    print(f"FAILED services.core_engine.pipeline_streamer: {e}")

try:
    from core import run_verification_parallel
    print("SUCCESS from core")
except ImportError as e:
    print(f"FAILED core: {e}")

try:
    from utils.export_manager import export_manager
    print("SUCCESS from utils.export_manager")
except ImportError as e:
    print(f"FAILED utils.export_manager: {e}")

try:
    from preprocessing_service.image_enhancing import ImageEnhancer
    print("SUCCESS from preprocessing_service.image_enhancing")
except ImportError as e:
    print(f"FAILED preprocessing_service.image_enhancing: {e}")
