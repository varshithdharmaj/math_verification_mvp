"""Helper script to set up API keys interactively."""

import os
from pathlib import Path


def setup_api_keys():
    """Interactive setup for API keys."""
    print("🔑 API Keys Setup")
    print("=" * 60)
    print("\nThis script will help you set up your API keys.")
    print("You can skip any key you don't have.\n")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    # Create .env from example if it doesn't exist
    if not env_file.exists() and env_example.exists():
        print("Creating .env file from .env.example...")
        with open(env_example, 'r') as f:
            content = f.read()
        with open(env_file, 'w') as f:
            f.write(content)
    
    # Read existing .env if it exists
    env_vars = {}
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    
    # OpenAI
    print("\n1. OpenAI API Key (for GPT models)")
    print("   Get it from: https://platform.openai.com/api-keys")
    current = env_vars.get('OPENAI_API_KEY', '')
    if current and current != 'your_openai_api_key_here':
        print(f"   Current: {current[:20]}... (hidden)")
        update = input("   Update? (y/n): ").lower() == 'y'
    else:
        update = True
    
    if update:
        new_key = input("   Enter OpenAI API key (or press Enter to skip): ").strip()
        if new_key:
            env_vars['OPENAI_API_KEY'] = new_key
    
    # Google/Gemini
    print("\n2. Google/Gemini API Key")
    print("   Get it from: https://makersuite.google.com/app/apikey")
    current = env_vars.get('GOOGLE_API_KEY', '')
    if current and current != 'your_google_api_key_here':
        print(f"   Current: {current[:20]}... (hidden)")
        update = input("   Update? (y/n): ").lower() == 'y'
    else:
        update = True
    
    if update:
        new_key = input("   Enter Google API key (or press Enter to skip): ").strip()
        if new_key:
            env_vars['GOOGLE_API_KEY'] = new_key
            env_vars['GEMINI_API_KEY'] = new_key
    
    # Anthropic
    print("\n3. Anthropic API Key (for Claude models)")
    print("   Get it from: https://console.anthropic.com/")
    current = env_vars.get('ANTHROPIC_API_KEY', '')
    if current and current != 'your_anthropic_api_key_here':
        print(f"   Current: {current[:20]}... (hidden)")
        update = input("   Update? (y/n): ").lower() == 'y'
    else:
        update = True
    
    if update:
        new_key = input("   Enter Anthropic API key (or press Enter to skip): ").strip()
        if new_key:
            env_vars['ANTHROPIC_API_KEY'] = new_key
    
    # Ollama
    print("\n4. Ollama Base URL (for local Llama models)")
    print("   Default: http://localhost:11434")
    current = env_vars.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    new_url = input(f"   Enter Ollama URL (or press Enter for default): ").strip()
    if new_url:
        env_vars['OLLAMA_BASE_URL'] = new_url
    elif 'OLLAMA_BASE_URL' not in env_vars:
        env_vars['OLLAMA_BASE_URL'] = 'http://localhost:11434'
    
    # Write .env file
    print("\n" + "=" * 60)
    print("Writing .env file...")
    
    with open(env_file, 'w') as f:
        f.write("# API Keys Configuration\n")
        f.write("# DO NOT commit this file to git!\n\n")
        
        f.write("# OpenAI API Key (for GPT models)\n")
        f.write(f"OPENAI_API_KEY={env_vars.get('OPENAI_API_KEY', 'your_openai_api_key_here')}\n\n")
        
        f.write("# Google/Gemini API Key\n")
        f.write(f"GOOGLE_API_KEY={env_vars.get('GOOGLE_API_KEY', 'your_google_api_key_here')}\n")
        f.write(f"GEMINI_API_KEY={env_vars.get('GEMINI_API_KEY', env_vars.get('GOOGLE_API_KEY', 'your_google_api_key_here'))}\n\n")
        
        f.write("# Anthropic API Key (for Claude models)\n")
        f.write(f"ANTHROPIC_API_KEY={env_vars.get('ANTHROPIC_API_KEY', 'your_anthropic_api_key_here')}\n\n")
        
        f.write("# Ollama Base URL (for local Llama models)\n")
        f.write(f"OLLAMA_BASE_URL={env_vars.get('OLLAMA_BASE_URL', 'http://localhost:11434')}\n")
    
    print("✅ .env file created/updated!")
    print("\n⚠️  IMPORTANT: Make sure .env is in .gitignore!")
    print("   The system will automatically load these keys when you use the verifiers.")
    
    # Test keys
    print("\n" + "=" * 60)
    test = input("Test API keys now? (y/n): ").lower() == 'y'
    if test:
        test_api_keys(env_vars)


def test_api_keys(env_vars):
    """Test if API keys work."""
    print("\n🧪 Testing API Keys...")
    
    # Load environment
    for key, value in env_vars.items():
        if value and 'your_' not in value and 'here' not in value:
            os.environ[key] = value
    
    # Test OpenAI
    if env_vars.get('OPENAI_API_KEY') and 'your_' not in env_vars.get('OPENAI_API_KEY', ''):
        print("\n1. Testing OpenAI...")
        try:
            from src.utils.llm_providers import LLMProvider
            provider = LLMProvider("openai", "gpt-3.5-turbo")
            if provider.client:
                print("   ✅ OpenAI connection successful!")
            else:
                print("   ❌ OpenAI connection failed")
        except Exception as e:
            print(f"   ❌ OpenAI error: {e}")
    
    # Test Gemini
    if env_vars.get('GOOGLE_API_KEY') and 'your_' not in env_vars.get('GOOGLE_API_KEY', ''):
        print("\n2. Testing Gemini...")
        try:
            from src.utils.llm_providers import LLMProvider
            provider = LLMProvider("gemini", "gemini-pro")
            if provider.client:
                print("   ✅ Gemini connection successful!")
            else:
                print("   ❌ Gemini connection failed")
        except Exception as e:
            print(f"   ❌ Gemini error: {e}")
    
    # Test Ollama
    print("\n3. Testing Ollama (Llama)...")
    try:
        import requests
        ollama_url = env_vars.get('OLLAMA_BASE_URL', 'http://localhost:11434')
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
        print("   Install: https://ollama.ai")


if __name__ == "__main__":
    setup_api_keys()

