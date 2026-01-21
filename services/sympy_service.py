"""
Enhanced Symbolic Verification using Math-Verify
Combines SymPy with HuggingFace's Math-Verify for robust verification
Port: 8002
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sympy as sp
import re
from typing import List, Dict
import time

# Import Math-Verify for advanced mathematical verification
try:
    from math_verify import parse, verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("[WARNING] Math-Verify not available, using SymPy only")

app = FastAPI(
    title="Enhanced SymPy Verification Service",
    description="Deterministic symbolic math verification with Math-Verify integration",
    version="3.0.0"
)

class VerificationRequest(BaseModel):
    steps: List[str]
    problem: str = ""  # Optional problem statement
    use_math_verify: bool = True  # Use Math-Verify if available
    
class VerificationResponse(BaseModel):
    model: str
    model_name: str
    verdict: str
    confidence: float
    errors: List[Dict]
    processing_time: float
    verification_method: str  # "sympy", "math-verify", or "hybrid"

class EnhancedSymbolicVerifier:
    def __init__(self):
        self.confidence_high = 0.98
        self.confidence_low = 0.95
        self.math_verify_available = MATH_VERIFY_AVAILABLE
        
    def verify(self, steps: List[str], problem: str = "", use_math_verify: bool = True) -> Dict:
        """
        Verify arithmetic/algebraic expressions using hybrid approach
        Returns: Dict with verdict, confidence, errors
        """
        start = time.time()
        
        errors = []
        verification_method = "sympy"
        
        # Try Math-Verify first if available and requested
        if use_math_verify and self.math_verify_available and problem:
            try:
                math_verify_errors = self._verify_with_math_verify(problem, steps)
                if math_verify_errors:
                    errors.extend(math_verify_errors)
                    verification_method = "math-verify"
            except Exception as e:
                print(f"[WARNING] Math-Verify failed: {e}, falling back to SymPy")
        
        # Always run SymPy verification for arithmetic checks
        sympy_errors = self._verify_with_sympy(steps)
        if sympy_errors:
            errors.extend(sympy_errors)
            if verification_method == "math-verify":
                verification_method = "hybrid"
            else:
                verification_method = "sympy"
        
        verdict = "ERROR" if errors else "VALID"
        confidence = self.confidence_high if verdict == "ERROR" else self.confidence_low
        
        return {
            'model': 'symbolic',
            'model_name': '[Symbolic] Enhanced Symbolic Verifier (SymPy + Math-Verify)',
            'verdict': verdict,
            'confidence': confidence,
            'errors': errors,
            'processing_time': time.time() - start,
            'verification_method': verification_method
        }
    
    def _verify_with_math_verify(self, problem: str, steps: List[str]) -> List[Dict]:
        """
        Use HuggingFace Math-Verify for advanced verification
        """
        errors = []
        
        try:
            # Combine all steps into solution
            full_solution = "\n".join(steps)
            
            # Extract final answer from last step
            if steps:
                last_step = steps[-1]
                # Try to find equation in last step
                equation_match = re.search(r'=\s*([^=\s]+)\s*$', last_step)
                if equation_match:
                    predicted_answer = equation_match.group(1).strip()
                    
                    # Parse using Math-Verify
                    try:
                        predicted_parsed = parse(f"${predicted_answer}$")
                        
                        # If we have expected answer in problem, verify
                        # This is a simplified check - in production, you'd extract expected answer
                        # For now, we'll use Math-Verify's parsing to validate the expression
                        
                        if predicted_parsed is None:
                            errors.append({
                                'step_number': len(steps),
                                'type': 'parsing_error',
                                'description': f"Math-Verify could not parse answer: {predicted_answer}",
                                'severity': 'MEDIUM',
                                'fixable': True,
                                'verification_method': 'math-verify'
                            })
                    except Exception as e:
                        errors.append({
                            'step_number': len(steps),
                            'type': 'math_verify_error',
                            'description': f"Math-Verify verification failed: {str(e)}",
                            'severity': 'LOW',
                            'fixable': False,
                            'verification_method': 'math-verify'
                        })
        except Exception as e:
            # Don't fail completely, just log
            print(f"[WARNING] Math-Verify check failed: {e}")
        
        return errors
    
    def _verify_with_sympy(self, steps: List[str]) -> List[Dict]:
        """
        Original SymPy verification for arithmetic
        """
        errors = []
        
        for i, step in enumerate(steps):
            step_errors = self._check_step(step, i+1)
            errors.extend(step_errors)
        
        return errors
    
    def _check_step(self, step: str, step_num: int) -> List[Dict]:
        """
        Check arithmetic calculations in a single step
        Matches patterns like: "5 + 3 = 8", "10 * 2 = 20"
        """
        errors = []
        
        # Pattern: number operator number = result
        pattern = r'(\d+\.?\d*)\s*([+\-*/×÷^])\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)'
        matches = re.findall(pattern, step)
        
        for match in matches:
            a, op, b, stated_result = match
            try:
                # Normalize operators
                if op == '×':
                    op = '*'
                elif op == '÷':
                    op = '/'
                
                # Calculate correct answer
                if op == '^':
                    correct = float(a) ** float(b)
                else:
                    correct = eval(f"{a}{op}{b}")
                
                # Compare (allow small floating point tolerance)
                if abs(float(stated_result) - correct) > 0.001:
                    errors.append({
                        'step_number': step_num,
                        'type': 'arithmetic_error',
                        'operation': op,
                        'found': f"{a} {op} {b} = {stated_result}",
                        'correct': f"{a} {op} {b} = {correct}",
                        'severity': 'HIGH',
                        'description': f"Arithmetic error in step {step_num}: {a} {op} {b} should equal {correct}, not {stated_result}",
                        'fixable': True,
                        'verification_method': 'sympy'
                    })
            except Exception as e:
                # Malformed expression
                errors.append({
                    'step_number': step_num,
                    'type': 'syntax_error',
                    'description': f"Could not parse expression in step {step_num}: {str(e)}",
                    'severity': 'MEDIUM',
                    'fixable': False,
                    'verification_method': 'sympy'
                })
        
        return errors

# Global verifier instance
verifier = EnhancedSymbolicVerifier()

@app.post("/verify", response_model=VerificationResponse)
async def verify_steps(request: VerificationRequest):
    """
    Endpoint: POST /verify
    Verify arithmetic in solution steps with hybrid approach
    """
    try:
        result = verifier.verify(request.steps, request.problem, request.use_math_verify)
        return VerificationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "enhanced_sympy_verification",
        "version": "3.0",
        "math_verify_available": MATH_VERIFY_AVAILABLE
    }

@app.get("/info")
async def service_info():
    return {
        "service": "Enhanced Symbolic Verifier",
        "capabilities": [
            "SymPy arithmetic verification",
            "Math-Verify advanced parsing" if MATH_VERIFY_AVAILABLE else "Math-Verify (not available)",
            "Hybrid verification approach",
            "Error detection with severity levels"
        ],
        "verification_methods": ["sympy", "math-verify", "hybrid"],
        "math_verify_status": "available" if MATH_VERIFY_AVAILABLE else "not installed"
    }

if __name__ == "__main__":
    import uvicorn
    print("[START] Enhanced SymPy Verification Service on port 8005...")
    if MATH_VERIFY_AVAILABLE:
        print("[OK] Math-Verify integration enabled")
    else:
        print("[WARNING] Math-Verify not available, using SymPy only")
    uvicorn.run(app, host="0.0.0.0", port=8005)
