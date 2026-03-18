from huggingface_hub import HfApi
import os

token = "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"
api = HfApi(token=token)

try:
    user_info = api.whoami()
    print(f"Token is VALID for user: {user_info['name']}")
    print(f"Permissions: {user_info.get('auth', {}).get('type', 'Unknown')}")
    # Check if we can write to the repo
    # This is harder to check without trying, but we'll see the role
except Exception as e:
    print(f"Token is INVALID: {e}")
