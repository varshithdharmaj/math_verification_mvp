# API Keys Setup Guide

## Overview

The system works **without API keys** by default. All verifiers have mock modes that don't require external APIs. However, if you want to use real LLM APIs for enhanced verification, you can optionally configure API keys.

## When Do You Need API Keys?

- **LLMLogicalChecker**: Optional - uses heuristics by default, but can use OpenAI/Anthropic/Google APIs for enhanced logical checking
- **EnsembleNeuralChecker**: Currently uses mock mode (no API keys needed)

## Setup Instructions

### Option 1: Environment Variables (Recommended)

Create a `.env` file in the project root:

```bash
# OpenAI (for LLMLogicalChecker)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic (future support)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google (future support)
GOOGLE_API_KEY=your_google_api_key_here
```

Then install python-dotenv and load it:
```bash
pip install python-dotenv
```

Add to your code:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Option 2: System Environment Variables

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your_openai_api_key_here"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your_openai_api_key_here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### Option 3: Streamlit Secrets (for Streamlit Cloud)

If deploying to Streamlit Cloud, use the secrets management:

1. Go to your Streamlit app settings
2. Add secrets in `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "your_openai_api_key_here"
```

## Enabling API Usage

### In Streamlit UI

1. Open the sidebar
2. Check "Use LLM API" checkbox
3. The system will automatically use your `OPENAI_API_KEY` if available

### In Python Code

```python
from src.models.llm_logical_checker import LLMLogicalChecker

# Enable API usage
checker = LLMLogicalChecker(
    use_api=True,  # Enable API
    api_provider="openai"  # or "anthropic", "google"
)
```

## Current Status

- ✅ **Works without API keys**: All verifiers have mock/heuristic modes
- ✅ **SymbolicVerifier**: No API needed (uses SymPy)
- ✅ **MLStepClassifier**: No API needed (uses local model)
- ✅ **LLMLogicalChecker**: Supports GPT, Gemini, Llama, Claude
- ✅ **EnsembleNeuralChecker**: Supports multi-model voting with different providers

## Supported Models

The system now supports multiple LLM providers:
- **OpenAI**: GPT-4, GPT-4-turbo, GPT-3.5-turbo
- **Google**: Gemini Pro, Gemini Pro Vision
- **Llama**: Any Ollama model (llama2, mistral, codellama, etc.)
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku)

See `MULTI_MODEL_SETUP.md` for detailed setup instructions.

## Getting API Keys

### OpenAI
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy and set as `OPENAI_API_KEY`

### Anthropic (Claude)
1. Go to https://console.anthropic.com/
2. Create an API key
3. Set as `ANTHROPIC_API_KEY`

### Google (Gemini)
1. Go to https://makersuite.google.com/app/apikey
2. Create an API key
3. Set as `GOOGLE_API_KEY`

## Testing API Setup

```python
import os
from src.models.llm_logical_checker import LLMLogicalChecker

# Check if API key is set
if os.getenv("OPENAI_API_KEY"):
    print("✅ API key found")
    checker = LLMLogicalChecker(use_api=True, api_provider="openai")
else:
    print("⚠️ No API key - using mock mode")
    checker = LLMLogicalChecker(use_api=False)
```

## Troubleshooting

**Issue**: "Warning: OPENAI_API_KEY not found, using mock mode"
- **Solution**: Set the environment variable or use mock mode (which works fine)

**Issue**: "Warning: openai package not installed"
- **Solution**: `pip install openai`

**Issue**: API calls failing
- **Solution**: Check your API key is valid and has credits/quota

## Recommendation

**For most use cases, you don't need API keys!** The system is designed to work well with:
- Symbolic verification (SymPy)
- Heuristic-based logical checking
- Mock ensemble voting
- Local ML classifier

Only enable APIs if you want enhanced logical checking capabilities.

