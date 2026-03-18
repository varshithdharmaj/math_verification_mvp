# Multi-Model Setup Guide (GPT, Gemini, Llama)

The system now supports multiple LLM providers for enhanced verification!

## Supported Models

### 1. **OpenAI (GPT)**
- Models: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`
- API Key: `OPENAI_API_KEY`
- Get key: https://platform.openai.com/api-keys

### 2. **Google (Gemini)**
- Models: `gemini-pro`, `gemini-pro-vision`
- API Key: `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- Get key: https://makersuite.google.com/app/apikey

### 3. **Llama (via Ollama)**
- Models: Any Ollama model (e.g., `llama2`, `mistral`, `codellama`)
- Setup: Install Ollama locally (no API key needed)
- Install: https://ollama.ai

### 4. **Anthropic (Claude)**
- Models: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`
- API Key: `ANTHROPIC_API_KEY`
- Get key: https://console.anthropic.com/

## Quick Setup

### 1. Install Dependencies

```bash
pip install openai google-generativeai anthropic requests
```

### 2. Set API Keys

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-openai-key"
$env:GOOGLE_API_KEY="your-google-key"
$env:ANTHROPIC_API_KEY="your-anthropic-key"
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### 3. Setup Ollama (for Llama models)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama2
ollama pull mistral
ollama pull codellama
```

## Usage

### In Streamlit UI

1. **LLM Logical Checker:**
   - Check "Use LLM API"
   - Select provider (OpenAI, Gemini, Llama, Anthropic)
   - Select specific model

2. **Ensemble Checker:**
   - Check "Use Ensemble API"
   - Set number of models (1-5)
   - Configure each model with provider and model name
   - Mix different providers for diverse voting!

### In Python Code

```python
from src.models.llm_logical_checker import LLMLogicalChecker
from src.models.ensemble_checker import EnsembleNeuralChecker

# Single model (GPT-4)
checker = LLMLogicalChecker(
    use_api=True,
    api_provider="openai",
    model="gpt-4"
)

# Single model (Gemini)
checker = LLMLogicalChecker(
    use_api=True,
    api_provider="gemini",
    model="gemini-pro"
)

# Single model (Llama via Ollama)
checker = LLMLogicalChecker(
    use_api=True,
    api_provider="llama",
    model="llama2"
)

# Ensemble with multiple models
ensemble = EnsembleNeuralChecker(
    use_apis=True,
    num_models=3,
    model_configs=[
        {"provider": "openai", "model": "gpt-4"},
        {"provider": "gemini", "model": "gemini-pro"},
        {"provider": "llama", "model": "llama2"}
    ]
)
```

## Example: Multi-Model Ensemble

Create a powerful ensemble using different providers:

```python
from src.models.ensemble_checker import EnsembleNeuralChecker

# Mix GPT-4, Gemini, and Llama for diverse opinions
ensemble = EnsembleNeuralChecker(
    use_apis=True,
    num_models=3,
    model_configs=[
        {"provider": "openai", "model": "gpt-4"},
        {"provider": "gemini", "model": "gemini-pro"},
        {"provider": "llama", "model": "mistral"}
    ]
)

result = ensemble.verify(
    step="5 + 3 = 9",  # Error!
    problem="Add 5 and 3",
    prev_steps=[]
)
# Result: Majority vote from 3 different models!
```

## Model Comparison

| Provider | Best For | Speed | Cost |
|----------|----------|-------|------|
| GPT-4 | Complex reasoning | Medium | High |
| GPT-3.5 | Fast checks | Fast | Low |
| Gemini | Balanced | Medium | Medium |
| Llama | Local/Private | Varies | Free |
| Claude | Detailed analysis | Medium | High |

## Troubleshooting

**"No LLM providers available"**
- Check API keys are set: `echo $OPENAI_API_KEY`
- For Ollama: Ensure it's running: `ollama list`

**Ollama connection error**
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Set custom URL: `export OLLAMA_BASE_URL="http://your-url:11434"`

**API rate limits**
- Use Llama (local) for unlimited requests
- Mix providers to distribute load

## Tips

1. **Start with GPT-3.5** - Fast and cheap for testing
2. **Use Llama locally** - No API costs, unlimited requests
3. **Mix providers** - Different models catch different errors
4. **Fallback to mock** - System works without any APIs!

## Cost Optimization

- Use **Llama (Ollama)** for local, free inference
- Use **GPT-3.5** for cost-effective cloud inference
- Use **GPT-4/Gemini** only for critical verifications
- Mix models in ensemble to balance cost and accuracy

