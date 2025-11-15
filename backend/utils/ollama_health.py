"""
Ollama Health Check Utility

Validates that Ollama service is running and required models are available.
"""

import requests
from typing import Tuple


def check_ollama_health() -> Tuple[bool, str]:
    """
    Check if Ollama is running and qwen2.5-coder:7b is available

    Returns:
        Tuple[bool, str]: (is_healthy, message)
            - is_healthy: True if Ollama is running with required model
            - message: Status message or error description
    """
    try:
        # Try to connect to Ollama API
        response = requests.get("http://localhost:11434/api/tags", timeout=5)

        if response.status_code != 200:
            return False, f"Ollama returned status {response.status_code}"

        # Check available models
        data = response.json()
        models = [m.get('name', '') for m in data.get('models', [])]

        # Check for qwen2.5-coder model (allow any variant)
        required_model = 'qwen2.5-coder:7b'
        model_exists = any(required_model in m for m in models)

        if not model_exists:
            # Check for any qwen2.5-coder variant
            qwen_variants = [m for m in models if 'qwen2.5-coder' in m]

            if qwen_variants:
                # Found a variant, that's acceptable
                return True, f"OK (using {qwen_variants[0]})"

            available = ', '.join(models) if models else 'none'
            return False, f"Model '{required_model}' not found. Available models: {available}"

        return True, "OK"

    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to Ollama at localhost:11434. Is Ollama running?"
    except requests.exceptions.Timeout:
        return False, "Ollama health check timed out (>5 seconds)"
    except Exception as e:
        return False, f"Health check failed: {str(e)}"


def get_ollama_models() -> list:
    """
    Get list of available Ollama models

    Returns:
        list: List of model names, empty list if Ollama is unavailable
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m.get('name', '') for m in data.get('models', [])]
        return []
    except:
        return []
