"""
Minimal FastAPI Backend for Robotics Demo
Uses local Ollama (Qwen2.5-Robotics-Coder) for code generation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import requests
import json

# Import algorithm generator
try:
    from algorithm_generator import (
        AlgorithmGenerator,
        AlgorithmRequest,
        AlgorithmResponse,
        get_generator
    )
    from algorithm_templates import get_algorithm_list
except ImportError:
    from backend.algorithm_generator import (
        AlgorithmGenerator,
        AlgorithmRequest,
        AlgorithmResponse,
        get_generator
    )
    from backend.algorithm_templates import get_algorithm_list

# Import conversational chat router
try:
    from api.conversational_chat import router as chat_router
except ImportError:
    from backend.api.conversational_chat import router as chat_router

app = FastAPI(title="Robotics Demo API")

# Include conversational chat router
app.include_router(chat_router)

# CORS - allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:7b"  # Using base model for code generation

# Check if Ollama is available
def check_ollama():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

ollama_available = check_ollama()
if ollama_available:
    print("✅ Ollama server connected (qwen2.5-robotics-coder)")
else:
    print("⚠️  Ollama server not running - start with: ollama serve")


class GenerateRequest(BaseModel):
    prompt: str
    robot_type: str


@app.get("/")
async def root():
    return {
        "service": "Robotics Demo API",
        "status": "ready",
        "model": OLLAMA_MODEL,
        "ollama_available": check_ollama()
    }


@app.get("/health")
async def health():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "Robotics Demo API",
        "ollama": check_ollama()
    }


@app.post("/api/generate")
async def generate_code(request: GenerateRequest):
    """Generate robot code using Ollama (qwen2.5-robotics-coder)"""

    if not check_ollama():
        raise HTTPException(status_code=500, detail="Ollama server not running")

    # Build prompt for code generation
    # Note: Override modelfile template by being explicit
    prompt = f"""You are a robotics code generator. Generate production-ready Python code ONLY.

Task: {request.prompt}
Robot Type: {request.robot_type}

IMPORTANT: Do NOT generate JSON. Do NOT extract requirements. Generate PYTHON CODE ONLY.

Generate a complete, executable Python controller class with:
- Necessary imports (numpy, time, etc.)
- A Robot controller class with __init__ and control methods
- Safety checks and error handling
- Example usage in if __name__ == "__main__"
- Clear variable names and inline comments

Output ONLY the Python code, no explanations."""

    try:
        # Call Ollama API
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 2000,
                "top_p": 0.9,
                "top_k": 40
            }
        }, timeout=120)

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Ollama error: {response.text}")

        result = response.json()
        generated_code = result.get("response", "")

        # Clean up if wrapped in markdown code blocks
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0].strip()
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0].strip()

        print(f"✅ Generated {len(generated_code)} chars for {request.robot_type}")

        return {
            "code": generated_code,
            "robot_type": request.robot_type,
            "status": "success"
        }

    except requests.Timeout:
        print(f"⏱️  Ollama timeout for {request.robot_type}")
        raise HTTPException(status_code=504, detail="Code generation timeout")
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class GenerateAlgorithmRequest(BaseModel):
    """Request model for algorithm generation"""
    description: str
    robot_type: str
    algorithm_type: str
    current_code: Optional[str] = None
    modification_request: Optional[str] = None


@app.post("/api/generate-algorithm")
async def generate_algorithm(request: GenerateAlgorithmRequest):
    """
    Generate TypeScript algorithm code from natural language description

    This endpoint enables the dynamic algorithm generation system where:
    - Users describe behavior in natural language
    - Qwen generates TypeScript algorithm code
    - Code can be hot-swapped into running simulation
    """

    if not check_ollama():
        raise HTTPException(status_code=500, detail="Ollama server not running")

    try:
        # Get algorithm generator instance
        generator = get_generator()

        # Create algorithm request
        algo_request = AlgorithmRequest(
            description=request.description,
            robot_type=request.robot_type,
            algorithm_type=request.algorithm_type,
            current_code=request.current_code,
            modification_request=request.modification_request
        )

        # Generate or modify algorithm
        if request.current_code and request.modification_request:
            # Modify existing algorithm
            result = generator.modify(
                current_code=request.current_code,
                modification=request.modification_request,
                algorithm_type=request.algorithm_type
            )
            print(f"✅ Modified {request.algorithm_type} algorithm: {request.modification_request}")
        else:
            # Generate new algorithm
            result = generator.generate(algo_request)
            print(f"✅ Generated {request.algorithm_type} algorithm for {request.robot_type}")

        return {
            "code": result.code,
            "parameters": result.parameters,
            "algorithm_type": result.algorithm_type,
            "description": result.description,
            "complexity": result.estimated_complexity,
            "status": "success"
        }

    except Exception as e:
        print(f"❌ Algorithm generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate algorithm: {str(e)}")


@app.get("/api/algorithms")
async def list_algorithms():
    """
    Get list of available pre-built algorithms

    Returns algorithm templates from the algorithm library
    """
    try:
        algorithms = get_algorithm_list()
        return {
            "algorithms": algorithms,
            "count": len(algorithms),
            "status": "success"
        }
    except Exception as e:
        print(f"❌ Error listing algorithms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Robotics Demo API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
