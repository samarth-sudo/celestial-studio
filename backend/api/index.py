"""
Vercel Serverless Entry Point

This module wraps the FastAPI application for deployment on Vercel's serverless platform.
It imports the main FastAPI app and exposes it as a serverless function handler.

Note:
- Vercel has a 4.5MB deployment size limit for serverless functions
- Request timeout is 10 seconds on free plan, 60 seconds on pro
- No persistent filesystem storage
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import backend modules
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Import the main FastAPI app
try:
    from main import app
except ImportError:
    # Fallback for different import paths
    from backend.main import app

# Vercel serverless handler
# This is the entry point that Vercel will call
handler = app

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
