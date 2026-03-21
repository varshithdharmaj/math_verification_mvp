from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sympy as sp
import uvicorn

app = FastAPI(title="Representation Service", description="Converts normalized text into symbolic mathematical forms using SymPy.")

class RepresentationRequest(BaseModel):
    normalized_text: str
    tokens: Optional[List[str]] = []

class RepresentationResponse(BaseModel):
    status: str
    symbolic_expr: str
    latex_repr: str
    is_equation: bool

@app.get("/health")
async def health_check():
    """Returns the health status of the Representation service."""
    return {"status": "healthy", "service": "representation"}

def parse_to_sympy(text: str):
    """
    Attempts to parse a string into a SymPy expression or equation.
    """
    try:
        if "=" in text:
            lhs_str, rhs_str = text.split("=", 1)
            lhs = sp.sympify(lhs_str.strip())
            rhs = sp.sympify(rhs_str.strip())
            return sp.Eq(lhs, rhs), True
        else:
            expr = sp.sympify(text.strip())
            return expr, False
    except Exception as e:
        raise ValueError(f"SymPy Parsing Error: {str(e)}")

@app.post("/represent", response_model=RepresentationResponse)
async def create_representation(request: RepresentationRequest):
    """
    Accepts normalized mathematical text, attempts to convert it to a robust SymPy symbolic object,
    and returns its standard string and LaTeX representations.
    This canonical format is then sent to the Verification Service.
    """
    if not request.normalized_text.strip():
        raise HTTPException(status_code=400, detail="Normalized text cannot be empty.")
    
    try:
        sym_obj, is_eq = parse_to_sympy(request.normalized_text)
        
        return {
            "status": "success",
            "symbolic_expr": str(sym_obj),
            "latex_repr": sp.latex(sym_obj),
            "is_equation": is_eq
        }
    except ValueError as val_err:
        # If SymPy fails (e.g. invalid syntax for complex math), we return a fallback representation
        return {
            "status": "partial_success",
            "symbolic_expr": request.normalized_text, # fallback to raw
            "latex_repr": request.normalized_text,
            "is_equation": "=" in request.normalized_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Representation processing failed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
