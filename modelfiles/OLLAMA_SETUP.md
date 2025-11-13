# Ollama Setup Guide for Qwen2.5-Robotics-Coder

This guide explains how to set up and use the custom Qwen2.5-Robotics-Coder model with Ollama for robotics code generation.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Install Ollama](#install-ollama)
3. [Create Custom Model](#create-custom-model)
4. [Test the Model](#test-the-model)
5. [Integration with Robot Code Generator](#integration)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **macOS** (Apple Silicon or Intel)
- **8GB+ RAM** (16GB recommended for 7B model)
- **10GB+ free disk space**
- **Internet connection** (for initial model download)

---

## Install Ollama

### Option 1: Download from Website (Recommended)

1. Visit [ollama.com](https://ollama.com)
2. Download the macOS installer
3. Open the downloaded `.dmg` file
4. Drag Ollama to your Applications folder
5. Launch Ollama from Applications

### Option 2: Install via Homebrew

```bash
brew install ollama
```

### Verify Installation

```bash
# Check Ollama version
ollama --version

# Start Ollama service (runs in background)
ollama serve
```

You should see output like:
```
Ollama is running on http://localhost:11434
```

---

## Create Custom Model

### Step 1: Download Base Model

First, download the Qwen2.5-Coder base model:

```bash
ollama pull qwen2.5-coder:7b
```

This will download ~4.7GB. Wait for completion:
```
pulling manifest
pulling 8934d96d3f08... 100%
success
```

### Step 2: Create Custom Model from Modelfile

Navigate to the modelfiles directory:

```bash
cd /Users/samarth/Desktop/idea/modelfiles
```

Create the custom model using the Modelfile:

```bash
ollama create qwen2.5-robotics-coder -f Qwen2.5-Robotics-Coder
```

Expected output:
```
transferring model data
using existing layer sha256:8934d96d3f08...
creating new layer sha256:abc123def456...
writing manifest
success
```

### Step 3: Verify Model Creation

List all models:

```bash
ollama list
```

You should see:
```
NAME                         ID              SIZE      MODIFIED
qwen2.5-robotics-coder      abc123def456    4.7 GB    2 minutes ago
qwen2.5-coder:7b            8934d96d3f08    4.7 GB    5 minutes ago
```

---

## Test the Model

### Basic Test

Test the model with a simple robotics query:

```bash
ollama run qwen2.5-robotics-coder "Create a 6-DOF robotic arm controller in Python with serial communication"
```

Expected response should include:
- Robot type detection (arm/manipulator)
- DOF extraction (6)
- Communication method (serial)
- Python code structure

### Interactive Mode

Start an interactive chat session:

```bash
ollama run qwen2.5-robotics-coder
```

Try these example prompts:

**Example 1: Requirement Extraction**
```
>>> Build a mobile robot with 4 wheels for warehouse navigation using Arduino
```

**Example 2: Code Generation**
```
>>> Generate Python code for a pick-and-place robot with camera and gripper
```

**Example 3: Simulation Setup**
```
>>> Create a PyBullet simulation for a 3-DOF robot arm reaching task
```

Press `Ctrl+D` to exit interactive mode.

---

## Integration with Robot Code Generator

### Current Architecture

The Robot Code Generator uses **MLX-optimized Qwen2.5-Coder** by default for fast inference on Apple Silicon:

```python
# In src/generation/code_generator.py
self.model, self.tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
```

### Option 1: Use Ollama as Alternative Backend (Future Enhancement)

To integrate Ollama, you would modify `code_generator.py`:

```python
import requests

class OllamaCodeGenerator:
    def __init__(self, model_name="qwen2.5-robotics-coder"):
        self.model = model_name
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str, max_tokens: int = 2000) -> str:
        response = requests.post(self.url, json={
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.3,
                "top_p": 0.9
            }
        })
        return response.json()["response"]
```

### Option 2: Standalone Requirement Extraction

Use Ollama specifically for Stage 1 (requirement extraction):

```python
# In src/generation/requirement_extractor.py
def _llm_extraction(self, user_prompt: str) -> Dict[str, Any]:
    # Use Ollama for faster requirement extraction
    import requests
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "qwen2.5-robotics-coder",
        "prompt": f"Extract robot specifications from: {user_prompt}",
        "stream": False
    })
    # Parse JSON response
    return self._parse_llm_response(response.json()["response"])
```

### Why MLX vs Ollama?

| Feature | MLX (Current) | Ollama (Alternative) |
|---------|---------------|----------------------|
| Speed on Apple Silicon | ⚡️ Fastest | 🟢 Fast |
| Memory Usage | 🟢 Optimized | 🟡 Higher |
| Model Format | `.safetensors` | `GGUF` |
| Integration | Direct Python | REST API |
| Best For | Production inference | Development/testing |

**Recommendation:** Keep MLX for production, use Ollama for testing and development.

---

## Troubleshooting

### Issue: "Ollama command not found"

**Solution:**
```bash
# Add Ollama to PATH
export PATH=$PATH:/Applications/Ollama.app/Contents/MacOS
```

Or restart terminal after installation.

### Issue: "Model not found"

**Solution:**
```bash
# List all models
ollama list

# Re-create model
ollama create qwen2.5-robotics-coder -f Qwen2.5-Robotics-Coder
```

### Issue: Slow generation

**Cause:** Limited RAM or CPU resources

**Solutions:**
1. Close other applications
2. Use smaller model:
   ```bash
   ollama pull qwen2.5-coder:1.5b
   ```
3. Reduce context window in Modelfile:
   ```
   PARAMETER num_ctx 4096
   ```

### Issue: "Connection refused" error

**Cause:** Ollama service not running

**Solution:**
```bash
# Start Ollama server
ollama serve

# In another terminal, run your command
ollama run qwen2.5-robotics-coder
```

### Issue: Out of memory

**Solution:**
```bash
# Use quantized model
ollama pull qwen2.5-coder:7b-q4_0

# Adjust Modelfile
FROM qwen2.5-coder:7b-q4_0
```

---

## Advanced Usage

### Custom Parameters

Edit `Qwen2.5-Robotics-Coder` Modelfile to adjust:

```
# Faster but less creative (default)
PARAMETER temperature 0.3

# More creative but less consistent
PARAMETER temperature 0.7

# Larger context for complex prompts
PARAMETER num_ctx 16384

# Adjust sampling
PARAMETER top_p 0.9
PARAMETER top_k 40
```

After editing, recreate the model:
```bash
ollama create qwen2.5-robotics-coder -f Qwen2.5-Robotics-Coder
```

### Multi-Model Comparison

Compare different models for the same task:

```bash
# Test base model
ollama run qwen2.5-coder:7b "Create 6-DOF robot controller"

# Test custom model
ollama run qwen2.5-robotics-coder "Create 6-DOF robot controller"
```

The custom model should provide more robotics-specific insights.

### REST API Usage

Use Ollama's REST API directly:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-robotics-coder",
  "prompt": "Create a mobile robot controller",
  "stream": false
}'
```

---

## Performance Benchmarks

### Apple Silicon (M1/M2/M3)

| Model Size | Tokens/sec | Memory Usage | Load Time |
|------------|-----------|--------------|-----------|
| 1.5B       | ~60 t/s   | 2 GB         | 1-2 sec   |
| 7B         | ~20 t/s   | 6 GB         | 3-5 sec   |
| 14B        | ~10 t/s   | 12 GB        | 8-10 sec  |

### Typical Generation Times

- **Requirement Extraction**: 2-5 seconds
- **Simple Code Generation**: 10-20 seconds
- **Complex Code Generation**: 30-60 seconds

---

## Model Specialization Details

The `qwen2.5-robotics-coder` model is optimized for:

### 1. Requirement Extraction
- Parses natural language robot descriptions
- Identifies: robot type, DOF, sensors, task, platform
- Extracts hardware specifications

### 2. Code Generation Speed
- Reduced temperature (0.3) for consistent output
- Optimized context window (8192 tokens)
- Focused system prompts reduce unnecessary verbosity

### 3. Simulation Accuracy
- Understands PyBullet and MuJoCo APIs
- Generates realistic physics parameters
- Creates proper Gymnasium environments

### 4. Hardware Integration
- Knows Arduino, ROS, and Python robotics libraries
- Generates serial/I2C/SPI communication code
- Handles sensor integration (cameras, grippers, encoders)

---

## Next Steps

1. ✅ Model installed and tested
2. 🔄 Integrate with requirement extractor (optional)
3. 📊 Benchmark against MLX implementation
4. 🚀 Use for development and testing

---

## Resources

- **Ollama Documentation**: https://github.com/ollama/ollama
- **Qwen2.5-Coder**: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- **Modelfile Reference**: https://github.com/ollama/ollama/blob/main/docs/modelfile.md

---

## Support

For issues specific to:
- **Ollama**: https://github.com/ollama/ollama/issues
- **This Project**: /Users/samarth/Desktop/idea/README.md
- **Model Performance**: Adjust parameters in Modelfile and recreate

---

**Last Updated**: 2025-11-07
**Ollama Version**: 0.1.x+
**Model**: qwen2.5-robotics-coder (based on Qwen2.5-Coder-7B)
