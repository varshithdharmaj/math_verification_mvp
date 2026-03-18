# 💡 QWEN CODER / CURSOR IDE - READY-TO-USE PROMPTS
## Copy-Paste These Directly Into AI Chat for Instant Code Generation

---

## 🎯 PROMPT PACK 1: CORE SYSTEM ARCHITECTURE

### Prompt 1A: Build Model 1 (Symbolic Verifier)
```
Build a Python class called SymbolicVerifier that:

1. Uses SymPy for mathematical calculation verification
2. Takes a list of solution steps as input
3. Detects arithmetic errors in patterns: "(\d+)\s*[+\-\*/]\s*(\d+)\s*=\s*(\d+)"
4. Supports: addition, subtraction, multiplication, division
5. Returns a dictionary with:
   - verdict: "VALID" or "ERROR"
   - confidence: 0.95 if no errors, 0.90 if errors found
   - errors: list of error dictionaries containing:
     - step_number: which step had error
     - type: "calculation_error"
     - operation: "+", "-", "*", "/"
     - found: what was written
     - correct: what it should be
     - severity: "HIGH"

Example input step: "3 + 2 = 5 apples"
Should detect: Valid
Example input step: "5 - 1 = 6"
Should detect: Error (correct is 4)

Make it handle edge cases and work with concurrent execution.
```

### Prompt 1B: Build Model 2 (LLM Logical Checker)
```
Build a Python class called LLMLogicalChecker that:

1. Takes model_name parameter (default "GPT-4")
2. Checks logical consistency in solution steps
3. Detects these error patterns:
   - Contradictions: if step contains both "and" and "but"
   - Operation mismatches: if text says "subtract" but math uses "+"
   - Semantic inconsistencies
4. Returns dictionary with:
   - verdict: "VALID" or "ERROR"
   - confidence: 0.87 if no errors, 0.82 if errors
   - errors: list with step_number, type, description, severity
   - model_name: the LLM being simulated

Make it flexible for different LLM names (GPT-4, Llama 2, Gemini).
Handle parallel execution and fast processing.
```

### Prompt 1C: Build Model 3 (Ensemble Neural Checker)
```
Build a Python class called EnsembleNeuralChecker that:

1. Takes list of model_names (default: ["GPT-4", "Llama 2", "Gemini"])
2. Simulates multiple LLMs voting on solution validity
3. For input steps, determines if each sub-model says VALID or ERROR
4. Uses voting logic:
   - Count ERROR votes
   - Count VALID votes
   - If ERROR votes > VALID votes: overall = ERROR
   - Else: overall = VALID
5. Calculates confidence from agreement ratio:
   - 3/3 agree: 90% confidence
   - 2/3 agree: 80% confidence
   - 1/3 agree: 60% confidence
6. Returns dictionary with:
   - verdict: final VALID/ERROR
   - confidence: calculated from voting
   - sub_models: dict mapping each model to its verdict
   - agreement: "2/3 agree" format

Make the simulation detect errors like "5 - 1 = 6" as all models voting ERROR.
```

---

## 🎯 PROMPT PACK 2: CONSENSUS & DECISION MAKING

### Prompt 2A: Build Consensus Mechanism
```
Build a function called compute_consensus that:

1. Takes 3 model results as input (from SymbolicVerifier, LLMLogicalChecker, EnsembleNeuralChecker)
2. Extracts verdicts (VALID/ERROR) and confidences (0.0-1.0) from each
3. Implements weighted voting with these weights:
   - Symbolic: 40% (0.40)
   - LLM Logical: 35% (0.35)
   - Ensemble: 25% (0.25)
4. Calculates error_score:
   - For each model voting ERROR: add (weight × confidence) to score
   - Example: if Symbolic votes ERROR with 0.95: add 0.40 × 0.95 = 0.38
5. Makes decision: if error_score > 0.50: final verdict = ERROR, else VALID
6. Calculates overall confidence:
   - If all 3 agree: average(confidences) × 1.1 (boost to max 0.99)
   - If 2/3 agree: average of agreeing models' confidences
   - If mixed: average(confidences) × 0.8
7. Determines agreement_type:
   - "UNANIMOUS ✓✓✓" if all 3 match
   - "MAJORITY (2/3) ✓✓" if 2/3 match
   - "MIXED ✓" if all different
8. Returns dictionary with:
   - final_verdict: VALID/ERROR
   - overall_confidence: 0.0-0.99
   - error_score: calculated score
   - individual_verdicts: dict of each model's verdict
   - individual_confidences: dict of each model's confidence
   - agreement_type: string describing agreement
   - all_errors: combined list from all models

This is the decision engine for the system.
```

### Prompt 2B: Error Classification & Taxonomy
```
Build a function called classify_error that:

1. Takes an error dictionary as input
2. Classifies into these 10+ error types:
   - Arithmetic Error: calculation mistakes
   - Algebraic Error: wrong operations on variables
   - Logical Error: contradictions, circular reasoning
   - Operation Mismatch: says one thing does another
   - Conceptual Error: fundamental misunderstanding
   - Notation Error: informal/incorrect symbols
   - Unit Error: wrong units in answer
   - Order of Operations: wrong calculation sequence
   - Sign Error: wrong +/- sign
   - Semantic Error: meaning doesn't match
3. Assigns severity:
   - HIGH: affects final answer
   - MEDIUM: affects reasoning quality
   - LOW: formatting/notation
4. Determines if fixable:
   - Arithmetic: almost always fixable (90%+)
   - Logical: sometimes fixable (60%)
   - Conceptual: rarely fixable (20%)
5. Returns enhanced error dict with classification details

Use regex patterns and heuristics for detection.
```

---

## 🎯 PROMPT PACK 3: EXPLANATION & CORRECTION

### Prompt 3A: Generate Natural Language Explanations
```
Build a function called generate_explanation that:

1. Takes error dictionary as input (has: type, found, correct, step_number, etc.)
2. Generates human-readable explanation:
   - For arithmetic: "You wrote X but Y actually equals Z. When you [operation], you get [result]."
   - For logical: "This contradicts your earlier statement that [previous statement]."
   - For operation: "You said to [operation] but your math shows [different operation]."
3. Includes a corrected step example
4. Provides learning context (why the mistake is common)
5. Suggests similar types of problems for practice
6. Returns explanation as natural language string (2-3 sentences)

Example input error:
  {type: "calculation_error", found: "5 - 1 = 6", correct: "5 - 1 = 4", ...}

Example output:
  "You wrote 6, but 5 - 1 actually equals 4. When you subtract 1 from 5, 
   you count down one number: 5, 4. So 5 - 1 = 4, not 6."

Make it friendly and educational for students.
```

### Prompt 3B: Automatic Error Correction
```
Build a function called correct_solution that:

1. Takes original solution steps list and error details
2. For each error:
   - If arithmetic error: calculate correct value, replace in step
   - If logical error: flag for manual review (don't auto-fix)
   - If conceptual error: flag for manual review
3. Returns:
   - corrected_steps: list with fixes applied
   - correction_log: what was changed and why
   - success_rate: % of errors successfully fixed
   - manual_review_needed: list of errors needing human review

For example:
  Input step: "She gives 1 away: 5 - 1 = 6 apples"
  Output: "She gives 1 away: 5 - 1 = 4 apples"

Track correction success rates:
  - Arithmetic: target 92%
  - Logical: target 68%
  - Conceptual: target 45%
```

---

## 🎯 PROMPT PACK 4: DASHBOARD & INTERFACE

### Prompt 4A: Build Streamlit Dashboard
```
Build a Streamlit web application with these sections:

LEFT PANEL:
  - Title: "📝 Input"
  - Text area for problem (80px height)
  - Text area for steps (120px height, one per line)
  - Note: can handle multi-line input

RIGHT PANEL:
  - Title: "🎯 Live Flowchart"
  - Show ASCII flowchart:
    ┌─────────────┐
    │ 📥 INPUT    │
    ├─────────────┤
    │ 🔍 PARSING  │
    ├─────────────┤
    │ 🔄 PARALLEL │
    │ (3 models)  │
    ├─────────────┤
    │ ⚖️ CONSENSUS│
    ├─────────────┤
    │ 📤 OUTPUT   │
    └─────────────┘

SIDEBAR:
  - Section: "⚙️ Configuration"
  - Checkboxes: GPT-4 (checked), Llama 2 (checked), Gemini (checked)
  - Info box showing selected models

BUTTONS:
  - "🚀 Verify Solution" - main action button
  - "🔄 Clear" - reset form

PROCESSING FLOW DISPLAY:
  - Section below buttons: "📊 Processing Flow"
  - Shows logs from verification process
  - Use color-coded boxes:
    - Green (valid-box): for ✓ and valid steps
    - Red (error-box): for ❌ and errors
    - Orange (step-box): for ⚠️ and processing steps

RESULTS SECTION:
  - 3 columns: Final Verdict | Confidence % | Processing Time
  - "🤖 Model Verdicts": 3 model cards side by side
  - Each card shows: model name, verdict (green/red), confidence %, errors count
  - "🔴 Error Details": expandable sections for each error
  - "⚖️ Consensus Mechanism": breakdown of voting

Use session state to track logs and results.
Make it responsive and colorful.
```

### Prompt 4B: Real-time Processing Logs
```
Build a logging system for Streamlit that:

1. Maintains session_state.steps_log as list of log entries
2. Each log entry has:
   - step: what happened (e.g., "Model 1 Started")
   - model: which model ("Symbolic", "LLM", "Ensemble", "Consensus")
   - status: emoji indicator (⏳, ✓, ❌, ⚠️, 🔢, 🧠, 🤖, 📊)
   - details: text description
3. Display logs in real-time as verification happens
4. Color-code display:
   - If status starts with ✓: use valid-box (green)
   - If status starts with ❌: use error-box (red)
   - If status starts with ⚠️: use step-box (orange)
   - Otherwise: use step-box with emoji
5. Show in expandable/scrollable container
6. Include timestamp for each entry

Example log entries:
  {step: "Model 1 Started", model: "Symbolic", status: "⏳", details: "Checking arithmetic..."}
  {step: "Error Found", model: "Symbolic", status: "❌", details: "5 - 1 = 6 should be 4"}
  {step: "Model 1 Completed", model: "Symbolic", status: "✓ ERROR", details: "Found 1 error"}

Make it update live as processing happens.
```

---

## 🎯 PROMPT PACK 5: INTEGRATION & PARALLEL EXECUTION

### Prompt 5A: Parallel Model Execution
```
Build a function called run_verification_parallel that:

1. Takes problem and steps as input
2. Initializes 3 models:
   - SymbolicVerifier()
   - LLMLogicalChecker(model_name)
   - EnsembleNeuralChecker(model_list)
3. Uses ThreadPoolExecutor with max_workers=3
4. Submits all 3 models to execute simultaneously:
   - symbolic_future = executor.submit(symbolic.verify, steps)
   - llm_future = executor.submit(llm.verify, steps)
   - ensemble_future = executor.submit(ensemble.verify, steps)
5. Collects results (blocks until all done):
   - r1 = symbolic_future.result()
   - r2 = llm_future.result()
   - r3 = ensemble_future.result()
6. Calls compute_consensus(r1, r2, r3)
7. Returns final result dict with:
   - problem
   - steps
   - model_results: {symbolic: r1, llm_logical: r2, ensemble: r3}
   - consensus: consensus result
   - processing_time: total time

Measure execution time and ensure parallel operation.
```

### Prompt 5B: Session State Management
```
Build session state initialization that:

1. Initializes on app start:
   if 'steps_log' not in st.session_state:
     st.session_state.steps_log = []
   if 'results' not in st.session_state:
     st.session_state.results = None

2. On "Verify" button click:
   - Clear steps_log: st.session_state.steps_log = []
   - Run verification with parallel execution
   - Capture all logs in steps_log
   - Store final result in st.session_state.results

3. On "Clear" button click:
   - Clear steps_log: st.session_state.steps_log = []
   - Clear results: st.session_state.results = None
   - Call st.rerun() to refresh UI

4. Display functions check session state:
   if st.session_state.steps_log: display logs
   if st.session_state.results: display results

Use this pattern throughout the app for state persistence.
```

---

## 🎯 PROMPT PACK 6: DEPLOYMENT & TESTING

### Prompt 6A: Google Colab Deployment
```
Create a setup for Google Colab that:

1. Cell 1 - Install dependencies:
   !pip install streamlit pyngrok -q
   from pyngrok import ngrok
   import threading, time

2. Cell 2 - Write app to file:
   %%writefile flowchart_app.py
   [paste complete Streamlit code]

3. Cell 3 - Launch public URL:
   - Start Streamlit in background thread
   - Use ngrok to expose public URL
   - Print public URL for sharing

4. Cell 4 - Test input:
   - Provide sample inputs for quick testing

Result: Everyone can access via public URL without installation.
```

### Prompt 6B: Test Suite
```
Build unit tests using pytest that:

1. Test SymbolicVerifier:
   - Test addition: "3 + 2 = 5" → VALID
   - Test subtraction error: "5 - 1 = 6" → ERROR
   - Test multiplication: "3 * 4 = 12" → VALID
   - Test division: "60 / 2 = 30" → VALID

2. Test LLMLogicalChecker:
   - Test valid logic: normal step → VALID
   - Test contradiction: "and...but" pattern → ERROR
   - Test operation mismatch: "subtract" + "+" → ERROR

3. Test EnsembleNeuralChecker:
   - Test unanimous agreement: all ERROR
   - Test majority vote: 2/3 ERROR
   - Test mixed vote: different opinions

4. Test ConsensusMechanism:
   - Test unanimous ERROR → final ERROR with boost
   - Test majority → uses majority confidence
   - Test mixed → penalizes confidence

5. Test ErrorClassification:
   - Categorizes errors correctly
   - Assigns proper severity
   - Determines fixability

Create test_data.json with 20+ test cases.
Report coverage and pass/fail status.
```

---

## 🚀 USAGE GUIDE FOR QWEN CODER / CURSOR

### Step 1: Copy Main Prompt
```
Copy the full QWEN-CURSOR-PROMPT.md file
```

### Step 2: Paste in AI Chat
```
Open Qwen Coder or Cursor IDE
Create new chat session
Paste the complete prompt document
```

### Step 3: Request Implementation
```
Example prompts:
"Build the complete system from this specification"
"Implement Model 1 and Model 2 classes first"
"Create the Streamlit dashboard interface"
"Generate the consensus mechanism function"
"Add error explanation generation"
"Set up parallel execution with ThreadPoolExecutor"
"Create Google Colab deployment script"
```

### Step 4: Customize & Refine
```
After initial code generation:
"Modify error_types to include [new type]"
"Change weights to 45%, 30%, 25%"
"Add logging for debugging"
"Optimize for faster processing"
```

### Step 5: Deploy & Test
```
Once complete:
"Generate unit tests"
"Create sample test cases"
"Set up Streamlit app"
"Deploy to Google Colab"
```

---

## 📋 QUICK CHECKLIST FOR AI ASSISTANT

When asking AI to build this:

**Provide:**
- ✅ Architecture diagram (in text format)
- ✅ Data structures (input/output formats)
- ✅ Algorithm explanations
- ✅ Error types to detect
- ✅ UI requirements
- ✅ Performance targets
- ✅ Integration points

**Verify AI Built:**
- ✅ Parallel execution (3 models simultaneously)
- ✅ Weighted consensus (40%, 35%, 25%)
- ✅ 10+ error type detection
- ✅ Explanation generation
- ✅ Streamlit dashboard
- ✅ Real-time processing logs
- ✅ Color-coded status
- ✅ Error correction
- ✅ Session state management
- ✅ Test coverage

**Before Using:**
- ✅ Test with sample inputs
- ✅ Verify accuracy (71.5% target)
- ✅ Check error detection (78.3% target)
- ✅ Monitor processing time (< 4.1s)
- ✅ Validate confidence scores
- ✅ Check for edge cases

---

**Ready to generate! Copy these prompts into Qwen Coder or Cursor now! 🚀**

