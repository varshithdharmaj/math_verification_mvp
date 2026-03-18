# How Sidebar Models Are Used - Current Implementation

## Current Architecture

### Three Main Models:

1. **Model 1: Symbolic Verifier (🔢)**
   - Uses: **SymPy** (mathematical library)
   - **Does NOT use sidebar models**
   - Purpose: Verifies arithmetic calculations
   - Weight in consensus: **40%**

2. **Model 2: LLM Logical Checker (🧠)**
   - Uses: **Pattern-based heuristics** (simulated LLM)
   - **Currently hardcoded to "GPT-4"** (not using sidebar selection)
   - Purpose: Checks logical consistency
   - Weight in consensus: **35%**

3. **Model 3: Ensemble Neural Checker (🤖)**
   - Uses: **Sidebar-selected models** (GPT-4, Llama 2, Gemini)
   - **This is where sidebar selection matters!**
   - Purpose: Simulates multiple LLMs voting
   - Weight in consensus: **25%**

## How Sidebar Models Work

### Current Flow:

```
Sidebar Selection (GPT-4, Llama 2, Gemini)
           │
           ▼
    EnsembleNeuralChecker
           │
           ├─→ Creates LLMLogicalChecker("GPT-4")
           ├─→ Creates LLMLogicalChecker("Llama 2")  
           └─→ Creates LLMLogicalChecker("Gemini")
           │
           ▼
    Each "model" votes on solution
           │
           ▼
    Majority voting determines verdict
```

### Important Notes:

1. **Currently a SIMULATION**: 
   - Not making real API calls to GPT-4, Llama 2, or Gemini
   - Uses pattern-based heuristics with different model names
   - All models use the same underlying logic

2. **Sidebar Selection Only Affects Model 3**:
   - If you uncheck "Llama 2", it won't vote in the ensemble
   - If you check all 3, all 3 vote
   - Model 2 (LLM Logical) is still hardcoded to "GPT-4"

3. **Voting Logic**:
   - Each selected model in ensemble gets 1 vote
   - If 2/3 vote ERROR → final = ERROR
   - If 2/3 vote VALID → final = VALID
   - Confidence based on agreement ratio

## Example:

**Sidebar: GPT-4 ✓, Llama 2 ✓, Gemini ✗**

```
Ensemble Model runs with 2 models:
├─ GPT-4 votes: ERROR (found calculation mistake)
└─ Llama 2 votes: ERROR (found calculation mistake)

Result: 2/2 agree → ERROR (100% agreement)
Confidence: 90% (all agree)
```

## To Use Real LLM APIs:

Currently, this is a **simulation**. To use real APIs, you would need to:

1. **Integrate OpenAI API** for GPT-4
2. **Integrate Anthropic/Claude API** or **HuggingFace** for Llama
3. **Integrate Google API** for Gemini
4. **Modify LLMLogicalChecker** to make actual API calls
5. **Add API keys** to environment variables

The current implementation is designed for:
- ✅ Fast testing without API costs
- ✅ Demonstrating the architecture
- ✅ Research/prototype purposes

For production, you'd want to integrate real LLM APIs.

