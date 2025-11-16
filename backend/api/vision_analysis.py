"""
Vision Analysis API Endpoint

Provides computer vision analysis using LLaVA vision-language model.
Processes camera renders and provides scene understanding, object detection insights.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
import base64
from typing import Dict, Any, List, Optional
import json

router = APIRouter()

# Ollama API configuration
OLLAMA_URL = "http://localhost:11434"
VISION_MODEL = "llava:7b"


class VisionAnalysisRequest(BaseModel):
    """Request model for vision analysis"""
    image_data: str  # base64 encoded image
    prompt: Optional[str] = "Describe what you see in this robotics simulation scene. Identify all objects, their positions, and any notable features."
    max_tokens: Optional[int] = 300


class ObjectDetectionInfo(BaseModel):
    """Detected object information from vision model"""
    label: str
    confidence: float
    description: str
    position_description: str  # e.g., "center-left", "top-right"


class VisionAnalysisResponse(BaseModel):
    """Response model for vision analysis"""
    description: str
    objects_identified: List[ObjectDetectionInfo]
    scene_understanding: str
    spatial_relationships: List[str]
    model_used: str


async def call_llava_vision(image_base64: str, prompt: str, max_tokens: int = 300) -> Dict[str, Any]:
    """
    Call LLaVA vision model via Ollama API

    Args:
        image_base64: Base64 encoded image data
        prompt: Text prompt for the vision model
        max_tokens: Maximum tokens in response

    Returns:
        Dict containing model response
    """
    try:
        # Ollama API endpoint for vision models
        url = f"{OLLAMA_URL}/api/generate"

        payload = {
            "model": VISION_MODEL,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }

        print(f"🔍 Calling LLaVA vision model...")
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        print(f"✅ LLaVA response received")

        return result

    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling LLaVA: {e}")
        raise HTTPException(status_code=500, detail=f"Vision model error: {str(e)}")


def parse_vision_response(response_text: str) -> Dict[str, Any]:
    """
    Parse LLaVA response into structured format

    Args:
        response_text: Raw text response from LLaVA

    Returns:
        Structured dictionary with parsed information
    """
    # Extract objects mentioned in response
    objects_identified = []

    # Common robotics objects to detect
    object_keywords = {
        'robot': 'mobile robot',
        'box': 'obstacle/target box',
        'cube': 'cube object',
        'sphere': 'spherical object',
        'cylinder': 'cylindrical object',
        'wall': 'wall/barrier',
        'floor': 'ground plane',
        'obstacle': 'obstacle',
        'target': 'target location',
        'waypoint': 'navigation waypoint'
    }

    response_lower = response_text.lower()

    for keyword, label in object_keywords.items():
        if keyword in response_lower:
            # Estimate confidence based on context
            confidence = 0.85 if response_lower.count(keyword) > 1 else 0.7

            # Extract position description if mentioned
            position_desc = "unknown location"
            if "center" in response_lower:
                position_desc = "center"
            elif "left" in response_lower:
                position_desc = "left side"
            elif "right" in response_lower:
                position_desc = "right side"
            elif "top" in response_lower or "above" in response_lower:
                position_desc = "upper region"
            elif "bottom" in response_lower or "below" in response_lower:
                position_desc = "lower region"

            objects_identified.append(ObjectDetectionInfo(
                label=label,
                confidence=confidence,
                description=f"{label} detected in scene",
                position_description=position_desc
            ))

    # Extract spatial relationships
    spatial_relationships = []
    if "next to" in response_lower or "beside" in response_lower:
        spatial_relationships.append("Objects are positioned next to each other")
    if "behind" in response_lower:
        spatial_relationships.append("Some objects are behind others")
    if "in front" in response_lower or "ahead" in response_lower:
        spatial_relationships.append("Objects positioned in front/ahead")
    if "between" in response_lower:
        spatial_relationships.append("Objects positioned between others")
    if "surrounded" in response_lower or "around" in response_lower:
        spatial_relationships.append("Objects surrounding central element")

    return {
        "objects_identified": objects_identified,
        "spatial_relationships": spatial_relationships,
        "scene_understanding": response_text[:200] + "..." if len(response_text) > 200 else response_text
    }


@router.post("/api/vision/analyze", response_model=VisionAnalysisResponse)
async def analyze_vision(request: VisionAnalysisRequest):
    """
    Analyze camera render using LLaVA vision model

    This endpoint:
    1. Accepts base64 encoded camera image
    2. Calls LLaVA vision-language model
    3. Parses response for object detection
    4. Returns structured scene understanding

    Args:
        request: VisionAnalysisRequest with image_data and optional prompt

    Returns:
        VisionAnalysisResponse with detected objects and scene description
    """
    try:
        print(f"📸 Received vision analysis request")
        print(f"   - Image data length: {len(request.image_data)} bytes")
        print(f"   - Prompt: {request.prompt[:50]}...")

        # Remove data URL prefix if present (e.g., "data:image/png;base64,")
        image_data = request.image_data
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]

        # Call LLaVA vision model
        llava_response = await call_llava_vision(
            image_base64=image_data,
            prompt=request.prompt,
            max_tokens=request.max_tokens
        )

        # Extract response text
        response_text = llava_response.get('response', '')

        # Parse response into structured format
        parsed = parse_vision_response(response_text)

        print(f"✅ Vision analysis complete:")
        print(f"   - Objects identified: {len(parsed['objects_identified'])}")
        print(f"   - Spatial relationships: {len(parsed['spatial_relationships'])}")

        return VisionAnalysisResponse(
            description=response_text,
            objects_identified=parsed['objects_identified'],
            scene_understanding=parsed['scene_understanding'],
            spatial_relationships=parsed['spatial_relationships'],
            model_used=VISION_MODEL
        )

    except Exception as e:
        print(f"❌ Vision analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Vision analysis failed: {str(e)}")


@router.get("/api/vision/status")
async def vision_model_status():
    """
    Check if vision model is available

    Returns:
        Status of LLaVA vision model
    """
    try:
        # Check if Ollama is running and model is available
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()

        models = response.json().get('models', [])
        llava_available = any(VISION_MODEL in model.get('name', '') for model in models)

        return {
            "ollama_connected": True,
            "vision_model": VISION_MODEL,
            "model_available": llava_available,
            "status": "ready" if llava_available else "model not found"
        }

    except Exception as e:
        return {
            "ollama_connected": False,
            "vision_model": VISION_MODEL,
            "model_available": False,
            "status": f"error: {str(e)}"
        }
