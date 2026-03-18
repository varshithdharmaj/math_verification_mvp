# Quick Start: API Keys Setup

Your API keys have been configured! Here's how to use them.

## ✅ Keys Configured

- ✅ **OpenAI** (GPT-4, GPT-3.5)
- ✅ **Google/Gemini** (Gemini Pro)
- ⚠️ **Anthropic** (Claude) - Add if needed
- ⚠️ **Ollama** (Llama) - Install locally if needed

## 🚀 Quick Start

### Option 1: Automatic (Recommended)

The system will automatically load keys from `.env` file:

```bash
# Just run the app - keys are already configured!
streamlit run src/ui/streamlit_app.py
```

### Option 2: Manual Setup Script

```bash
python scripts/setup_api_keys.py
```

### Option 3: Environment Variables

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="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"
$env:GOOGLE_API_KEY="AIzaSyAfG16zkOmBP2IHCoVkvXc5nf9caENW5QE"
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="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"
export GOOGLE_API_KEY="AIzaSyAfG16zkOmBP2IHCoVkvXc5nf9caENW5QE"
```

## 📝 Using in Code

```python
from src.models.llm_logical_checker import LLMLogicalChecker

# Use GPT-4
checker = LLMLogicalChecker(
    use_api=True,
    api_provider="openai",
    model="gpt-4"
)

# Use Gemini
checker = LLMLogicalChecker(
    use_api=True,
    api_provider="gemini",
    model="gemini-pro"
)
```

## 🎯 In Streamlit UI

1. Launch: `streamlit run src/ui/streamlit_app.py`
2. In sidebar, check "Use LLM API"
3. Select provider: OpenAI or Gemini
4. Select model
5. Click "Verify"!

## 🔒 Security Notes

- ✅ `.env` file is in `.gitignore` (won't be committed)
- ⚠️ **Never share your API keys publicly**
- ⚠️ **Rotate keys if exposed**
- ✅ Keys are loaded automatically from `.env`

## 🧪 Test Your Setup

```bash
python scripts/setup_api_keys.py
# Choose "y" when asked to test keys
```

Or test manually:

```python
from src.utils.llm_providers import get_available_providers
print(get_available_providers())
# Should show: ['openai', 'gemini']
```

## 🎉 You're Ready!

Your API keys are configured. The system will:
- Automatically use OpenAI for GPT models
- Automatically use Gemini for Google models
- Fall back to mock mode if keys are invalid

Enjoy multi-model verification! 🚀

