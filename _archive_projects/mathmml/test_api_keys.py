"""Quick test script to verify API keys are working."""

import os
from pathlib import Path

# Load .env if exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, using system environment variables")
except Exception as e:
    print(f"⚠️  Could not load .env: {e}")

print("\n" + "=" * 60)
print("🔑 Testing API Keys")
print("=" * 60)

# Check OpenAI
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key and "your_" not in openai_key:
    print(f"\n✅ OpenAI API Key: Found ({openai_key[:20]}...)")
    try:
        from src.utils.llm_providers import LLMProvider
        provider = LLMProvider("openai", "gpt-3.5-turbo")
        if provider.client:
            print("   ✅ OpenAI connection successful!")
        else:
            print("   ❌ OpenAI client not initialized")
    except Exception as e:
        print(f"   ❌ OpenAI error: {e}")
else:
    print("\n❌ OpenAI API Key: Not found")

# Check Gemini
gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if gemini_key and "your_" not in gemini_key:
    print(f"\n✅ Google/Gemini API Key: Found ({gemini_key[:20]}...)")
    try:
        from src.utils.llm_providers import LLMProvider
        provider = LLMProvider("gemini", "gemini-pro")
        if provider.client:
            print("   ✅ Gemini connection successful!")
        else:
            print("   ❌ Gemini client not initialized")
    except Exception as e:
        print(f"   ❌ Gemini error: {e}")
else:
    print("\n❌ Google/Gemini API Key: Not found")

# Check Anthropic
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
if anthropic_key and "your_" not in anthropic_key:
    print(f"\n✅ Anthropic API Key: Found ({anthropic_key[:20]}...)")
else:
    print("\n⚠️  Anthropic API Key: Not set (optional)")

# Check Ollama
print("\n🔍 Checking Ollama (Llama)...")
try:
    import requests
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    response = requests.get(f"{ollama_url}/api/tags", timeout=2)
    if response.status_code == 200:
        print("   ✅ Ollama is running!")
        models = response.json().get('models', [])
        if models:
            print(f"   Available models: {', '.join([m['name'] for m in models[:5]])}")
        else:
            print("   ⚠️  No models found. Run: ollama pull llama2")
    else:
        print("   ❌ Ollama not responding")
except Exception as e:
    print(f"   ⚠️  Ollama not running: {e}")
    print("   (This is OK - install from https://ollama.ai if you want local models)")

print("\n" + "=" * 60)
print("✅ API Key Test Complete!")
print("=" * 60)
print("\n💡 Tip: Run 'streamlit run src/ui/streamlit_app.py' to use these keys!")

