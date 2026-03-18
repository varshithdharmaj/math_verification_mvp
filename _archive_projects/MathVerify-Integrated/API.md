# API Documentation

Complete API reference for MathVerify-Integrated system.

## Core Classes

### InputProcessor

**Location**: `src.input_module.processor`

Processes and normalizes mathematical problem input.

#### Methods

##### `normalize_expression(text: str) -> str`

Converts mathematical notation to canonical form.

**Parameters:**
- `text`: Input mathematical expression string

**Returns:**
- Standardized mathematical expression string

**Example:**
```python
processor = InputProcessor()
result = processor.normalize_expression("2x + 3")
# Returns: "2*x + 3"
```

##### `extract_problem_components(text: str) -> dict`

Extracts structured components from a mathematical problem.

**Parameters:**
- `text`: Input problem text

**Returns:**
- Dictionary with keys: `question`, `constraints`, `unknowns`

**Example:**
```python
components = processor.extract_problem_components("Solve for x: 2x + 3 = 7")
# Returns: {
#     "question": "Solve for x: 2x + 3 = 7",
#     "constraints": [],
#     "unknowns": ["x"]
# }
```

##### `validate_input(text: str) -> bool`

Checks if input is valid mathematical text.

**Parameters:**
- `text`: Input text to validate

**Returns:**
- `True` if valid, `False` otherwise

---

### ReasoningEngine

**Location**: `src.reasoning_module.engine`

LLM-based reasoning engine for mathematical problem solving.

#### Constructor

```python
ReasoningEngine(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    device: Optional[str] = None,
    max_steps: int = 10
)
```

**Parameters:**
- `model_name`: HuggingFace model identifier
- `device`: Device to use ("cuda", "cpu", or None for auto-detect)
- `max_steps`: Maximum number of reasoning steps

#### Methods

##### `generate_solution(problem: str) -> dict`

Generate step-by-step solution for a mathematical problem.

**Parameters:**
- `problem`: Problem statement

**Returns:**
- Dictionary with keys: `steps`, `final_answer`, `confidence`

**Example:**
```python
engine = ReasoningEngine()
result = engine.generate_solution("Solve: 2x + 3 = 7")
# Returns: {
#     "steps": [...],
#     "final_answer": "x = 2",
#     "confidence": 0.85
# }
```

##### `generate_single_step(problem: str, previous_steps: list) -> dict`

Generate the next reasoning step given previous steps.

**Parameters:**
- `problem`: Original problem statement
- `previous_steps`: List of previous step dictionaries

**Returns:**
- Dictionary with `content` and `rationale` keys

---

### SymbolicVerifier

**Location**: `src.verification_module.verifier`

Symbolic verifier for mathematical expressions using SymPy.

#### Methods

##### `verify_equation(equation: str) -> dict`

Verify if an equation is mathematically correct.

**Parameters:**
- `equation`: Equation string to verify

**Returns:**
- Dictionary with keys: `valid`, `error`, `details`

**Example:**
```python
verifier = SymbolicVerifier()
result = verifier.verify_equation("2 + 2 = 4")
# Returns: {"valid": True, "error": None, "details": {...}}
```

##### `verify_step(step: str, context: dict = None) -> dict`

Verify a single reasoning step.

**Parameters:**
- `step`: Reasoning step text
- `context`: Optional context dictionary

**Returns:**
- Verification result dictionary

##### `extract_expressions(text: str) -> list`

Extract all mathematical expressions from text.

**Parameters:**
- `text`: Text to extract expressions from

**Returns:**
- List of expression strings

##### `symbolic_simplify(expr: str) -> str`

Simplify a mathematical expression using SymPy.

**Parameters:**
- `expr`: Expression string

**Returns:**
- Simplified expression string

---

### ErrorTaxonomy

**Location**: `src.verification_module.error_taxonomy`

Comprehensive error taxonomy for mathematical reasoning.

#### Methods

##### `classify_error(step: str, verification_result: dict) -> str`

Classify the type of error in a failed verification.

**Parameters:**
- `step`: The reasoning step that failed
- `verification_result`: Verification result dictionary

**Returns:**
- Error type string (e.g., "CALCULATION_ERROR")

**Example:**
```python
taxonomy = ErrorTaxonomy()
error_type = taxonomy.classify_error("2 + 2 = 5", {"valid": False})
# Returns: "CALCULATION_ERROR"
```

##### `get_error_description(error_type: str) -> str`

Get human-readable description of an error type.

**Parameters:**
- `error_type`: Error type string

**Returns:**
- Human-readable description

##### `suggest_correction(error_type: str, step: str) -> str`

Suggest how to fix an error based on its type.

**Parameters:**
- `error_type`: Error type string
- `step`: The step containing the error

**Returns:**
- Suggestion string

##### `generate_report(errors: list) -> dict`

Generate comprehensive error report.

**Parameters:**
- `errors`: List of error dictionaries

**Returns:**
- Dictionary with `total_errors`, `by_type`, `percentage`

---

### MathVerifyPipeline

**Location**: `src.pipeline`

Complete end-to-end pipeline for mathematical reasoning.

#### Constructor

```python
MathVerifyPipeline(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    device: Optional[str] = None
)
```

**Parameters:**
- `model_name`: Model name for reasoning engine
- `device`: Device to run model on

#### Methods

##### `process_problem(problem: str) -> dict`

Process a mathematical problem through the complete pipeline.

**Parameters:**
- `problem`: Mathematical problem statement

**Returns:**
- Dictionary with keys: `problem`, `solution`, `verification`, `errors`, `final_answer`, `confidence`

**Example:**
```python
pipeline = MathVerifyPipeline()
result = pipeline.process_problem("Solve: 2x + 3 = 7")
# Returns complete pipeline result with verification and error classification
```

##### `verify_and_correct(steps: list) -> list`

Verify each step and attempt corrections on failures.

**Parameters:**
- `steps`: List of reasoning step dictionaries

**Returns:**
- List of verified steps with corrections applied

---

## Utility Functions

### `load_json(file_path: str) -> Any`

Load JSON data from file.

### `save_json(data: Any, file_path: str) -> None`

Save data to JSON file.

### `format_confidence(confidence: float) -> str`

Format confidence score as percentage string.

---

## Error Types

The system classifies errors into the following types:

- `CALCULATION_ERROR`: Arithmetic mistakes
- `LOGICAL_ERROR`: Invalid inference
- `NOTATION_ERROR`: Malformed expressions
- `REASONING_GAP`: Missing justification
- `VISUAL_MISINTERPRETATION`: Diagram misreading

---

## Usage Examples

### Complete Pipeline Example

```python
from src.pipeline import MathVerifyPipeline

# Initialize
pipeline = MathVerifyPipeline()

# Process problem
result = pipeline.process_problem(
    "Janet's ducks lay 16 eggs per day. "
    "She eats 3 for breakfast and uses 4 for baking. "
    "She sells the rest for $2 each. How much does she make?"
)

# Access results
print(f"Answer: {result['final_answer']}")
print(f"Confidence: {result['confidence']}")
print(f"Errors: {result['errors']['total_errors']}")

# View verified steps
for step in result['solution']['steps']:
    status = "✅" if step['is_valid'] else "❌"
    print(f"{status} Step {step['number']}: {step['content']}")
```

### Individual Component Usage

```python
from src.input_module.processor import InputProcessor
from src.verification_module.verifier import SymbolicVerifier
from src.verification_module.error_taxonomy import ErrorTaxonomy

# Input processing
processor = InputProcessor()
normalized = processor.normalize_expression("2x + 3")

# Verification
verifier = SymbolicVerifier()
result = verifier.verify_equation("2 + 2 = 4")

# Error classification
taxonomy = ErrorTaxonomy()
error_type = taxonomy.classify_error("2 + 2 = 5", {"valid": False})
```

---

For more examples, see the `demo.py` file and test cases in `tests/`.

