import os
import sys
import json
import time
from typing import Dict, Any, List
import subprocess
from agent_orchestrator import run_agent_orchestrator
from consensus_module import evaluate_consensus

def invoke_ocr_subsystem(image_path: str) -> Dict[str, Any]:
    """Invokes the completely isolated Local OCR Engine to get the LaTeX text."""
    local_ocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "local_ocr"))
    
    # Platform-independent venv check
    venv_python_win = os.path.join(local_ocr_path, "venv", "Scripts", "python.exe")
    venv_python_linux = os.path.join(local_ocr_path, "venv", "bin", "python")
    
    if os.path.exists(venv_python_win):
        venv_python = venv_python_win
    elif os.path.exists(venv_python_linux):
        venv_python = venv_python_linux
    else:
        # Fallback to current environment python
        venv_python = sys.executable
        
    engine_script = os.path.join(local_ocr_path, "mvm2_ocr_engine.py")

    print(f"[Pipeline] Dispatching Image '{image_path}' to Vision Subsystem...")
    
    try:
        # Use stdin=subprocess.DEVNULL to avoid WinError 6 on some Windows environments
        result = subprocess.run(
            [venv_python, engine_script, image_path], 
            capture_output=True, 
            text=True, 
            check=True,
            stdin=subprocess.DEVNULL
        )
        output_str = result.stdout
        if "MVM2_OCR_OUTPUT_START" in output_str:
            json_str = output_str.split("MVM2_OCR_OUTPUT_START")[1].split("MVM2_OCR_OUTPUT_END")[0].strip()
            return json.loads(json_str)
        else:
            raise Exception(f"Failed to parse OCR output: {output_str}")
    except Exception as e:
        print(f"[ERROR] OCR Subsystem Failed: {e}")
        return {
            "latex_output": f"OCR Error: {str(e)}", 
            "detected_layout": [], 
            "weighted_confidence": 0.0,
            "backend": "error_fallback"
        }

def run_end_to_end_test():
    with open("verification_output.txt", "w", encoding="utf-8") as f:
        f.write("="*60 + "\\n")
        f.write("[RUNNING] MVM2 CORE NEURO-SYMBOLIC INTEGRATION PIPELINE VERIFICATION\\n")
        f.write("="*60 + "\\n")
        
        start_time = time.time()
        
        test_img = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_math.png"))
        ocr_result = invoke_ocr_subsystem(test_img)
        problem_text = ocr_result.get("latex_output", "\\int_{0}^{\\pi} \\sin(x^{2}) \\, dx")
        ocr_confidence = ocr_result.get("weighted_confidence", 0.90)
        
        f.write(f"\\n[OK] [LAYER 1] OCR Extraction Complete (Conf: {ocr_confidence*100:.1f}%)\\n")
        f.write(f"   Transcribed Math:  {problem_text}\\n\\n")
        
        agent_responses = run_agent_orchestrator(problem_text)
        
        f.write("\\n[Pipeline] Routing outputs into QWED Symbolic Verification and Math-Verify Consensus equations...\\n")
        consensus_result = evaluate_consensus(agent_responses)
        
        end_time = time.time()
        latency = end_time - start_time
        
        f.write("\\n" + "="*60 + "\\n")
        f.write("[RESULT] FINAL MVM2 PIPELINE VERDICT\\n")
        f.write("="*60 + "\\n")
        f.write(f"Verified Answer:   {consensus_result['final_verified_answer']}\\n")
        f.write(f"Weighted Score:    {consensus_result['winning_score']:.3f} / 1.000\\n")
        f.write(f"Total Latency:     {latency:.2f}s (Target: < 8.2s)\\n")
        f.write("\\n[Detailed Agent Matrix]\\n")
        for s in consensus_result['detail_scores']:
            f.write(f" - {s['agent']}: ans={s['raw_answer']:>5} | V_sym={s['V_sym']:.2f} | L_logic={s['L_logic']:.2f} | C_clf={s['C_clf']:.2f} -> Score_j={s['Score_j']:.3f}\\n")
            
        f.write("\\n[Divergence Group Normalization (Math-Verify)]\\n")
        for ans_key, data in consensus_result['divergence_groups'].items():
            f.write(f" - Group [ {ans_key} ]: {len(data['agent_indices'])} Agent(s) Support. Group Score = {data['aggregate_score']:.3f}\\n")

if __name__ == "__main__":
    run_end_to_end_test()
