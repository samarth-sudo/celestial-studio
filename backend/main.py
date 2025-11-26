"""
Minimal FastAPI Backend for Robotics Demo
Uses local Ollama (Qwen2.5-Robotics-Coder) for code generation
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import requests
import json
import os
import tempfile
import shutil
import asyncio

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

# Import vision analysis router
try:
    from api.vision_analysis import router as vision_router
except ImportError:
    from backend.api.vision_analysis import router as vision_router

# Import Genesis streaming router
try:
    from api.genesis_streaming import router as genesis_router
except ImportError:
    from backend.api.genesis_streaming import router as genesis_router

# Import robot API router
# Temporarily disabled for startup
# try:
#     from api.robot_api import router as robot_router
# except ImportError:
#     from backend.api.robot_api import router as robot_router
robot_router = None

# Import export package generator
try:
    from export.package_generator import PackageGenerator
except ImportError:
    from backend.export.package_generator import PackageGenerator

# Import optimization modules
try:
    from optimization.benchmark import AlgorithmBenchmark
    from optimization.comparator import AlgorithmComparator
except ImportError:
    from backend.optimization.benchmark import AlgorithmBenchmark
    from backend.optimization.comparator import AlgorithmComparator

# Import URDF modules
try:
    from urdf.parser import URDFParser
    from urdf.threejs_converter import ThreeJSConverter
    from urdf.generator import URDFGenerator, LinkBuilder, JointBuilder, RobotPresets
except ImportError:
    from backend.urdf.parser import URDFParser
    from backend.urdf.threejs_converter import ThreeJSConverter
    from backend.urdf.generator import URDFGenerator, LinkBuilder, JointBuilder, RobotPresets

# Import multi-robot modules
try:
    from multi_robot.manager import get_multi_robot_manager, reset_multi_robot_manager
except ImportError:
    from backend.multi_robot.manager import get_multi_robot_manager, reset_multi_robot_manager

# Import Genesis teleoperation server
try:
    from control.genesis_teleop_server import GenesisTeleopServer
except ImportError:
    from backend.control.genesis_teleop_server import GenesisTeleopServer

app = FastAPI(title="Robotics Demo API")

# Include conversational chat router
app.include_router(chat_router)

# Include vision analysis router
app.include_router(vision_router)

# Include Genesis streaming router
app.include_router(genesis_router)

# Include multi-robot API router (temporarily disabled)
# app.include_router(robot_router)

# CORS - allow frontend to call API
# In production, set FRONTEND_URL environment variable to your frontend domain
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000",  # Alternative dev port
    "http://localhost:5173",  # Vite default
    "https://*.vercel.app",   # Vercel deployments
]

# Allow all origins in development, specific origins in production
if os.getenv("ENVIRONMENT") == "production":
    # In production, be more restrictive but support Vercel domains
    allowed_origins = ALLOWED_ORIGINS
    # Also allow Vercel preview deployments if VERCEL_URL is set
    vercel_url = os.getenv("VERCEL_URL")
    if vercel_url and vercel_url not in allowed_origins:
        allowed_origins.append(f"https://{vercel_url}")
else:
    # Development: allow all origins for easier testing
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ollama configuration
# Use environment variable for Ollama URL (supports ngrok tunnel for production)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder-robotics:latest")

# Check if Ollama is available
def check_ollama():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
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
            "function_name": result.function_name,  # Required function name
            "function_signature": result.function_signature,  # Function signature info
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


class ExportRequest(BaseModel):
    """Request model for package export"""
    export_format: str  # 'react', 'ros', 'python', or 'algorithms'
    algorithms: List[Dict[str, Any]]
    scene_config: Dict[str, Any]
    robots: List[Dict[str, Any]]
    project_name: Optional[str] = "celestial_simulation"


@app.post("/api/export/package")
async def export_package(request: ExportRequest):
    """
    Generate complete simulation package for download

    Supports multiple export formats:
    - 'react': Complete React/TypeScript project with Three.js + Rapier
    - 'ros': ROS workspace with Python nodes, launch files, and URDF
    - 'python': Standalone Python scripts with algorithm implementations
    - 'algorithms': Algorithm code files only

    Returns the filename of the generated ZIP package
    """
    try:
        # Validate export format
        valid_formats = ['react', 'ros', 'python', 'algorithms']
        if request.export_format not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid export format. Must be one of: {', '.join(valid_formats)}"
            )

        # Create package generator
        generator = PackageGenerator()

        # Generate package
        print(f"🔄 Generating {request.export_format} package: {request.project_name}")
        zip_path = generator.generate_package(
            export_format=request.export_format,
            algorithms=request.algorithms,
            scene_config=request.scene_config,
            robots=request.robots,
            project_name=request.project_name
        )

        # Get just the filename for the response
        filename = os.path.basename(zip_path)

        print(f"✅ Package generated successfully: {filename}")

        return {
            "status": "success",
            "filename": filename,
            "export_format": request.export_format,
            "project_name": request.project_name,
            "download_url": f"/api/export/download/{filename}"
        }

    except ValueError as e:
        print(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Export error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate package: {str(e)}")


# TEMPORARILY DISABLED FOR SERVERLESS DEPLOYMENT
# This endpoint relies on local file system storage which doesn't work in serverless environments
# TODO: Implement using cloud storage (Vercel Blob, S3, or Railway Volumes) for production
#
# @app.get("/api/export/download/{filename}")
# async def download_package(filename: str):
#     """
#     Download generated package ZIP file
#
#     Returns the ZIP file for download
#     """
#     try:
#         # Security: Validate filename to prevent path traversal attacks
#         if not filename.endswith('.zip'):
#             raise HTTPException(status_code=400, detail="Invalid file format")
#
#         # Prevent path traversal: no directory separators or parent directory refs
#         if '/' in filename or '\\' in filename or '..' in filename:
#             raise HTTPException(status_code=400, detail="Invalid filename: path traversal detected")
#
#         # Only allow alphanumeric, dash, underscore, and dot
#         import re
#         if not re.match(r'^[a-zA-Z0-9_\-\.]+\.zip$', filename):
#             raise HTTPException(status_code=400, detail="Invalid filename format")
#
#         # Find the file in temp directory
#         # PackageGenerator creates temp files in system temp directory
#         import tempfile
#         temp_dir = tempfile.gettempdir()
#
#         # Look for the file in celestial_export_* directories
#         file_path = None
#         for item in os.listdir(temp_dir):
#             if item.startswith('celestial_export_'):
#                 potential_path = os.path.join(temp_dir, item, filename)
#                 # Additional security: verify the resolved path is still within temp_dir
#                 resolved_path = os.path.realpath(potential_path)
#                 if os.path.exists(resolved_path) and resolved_path.startswith(os.path.realpath(temp_dir)):
#                     file_path = resolved_path
#                     break
#
#         if not file_path or not os.path.exists(file_path):
#             raise HTTPException(status_code=404, detail="Package file not found")
#
#         print(f"📦 Downloading package: {filename}")
#
#         return FileResponse(
#             path=file_path,
#             media_type='application/zip',
#             filename=filename
#         )
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"❌ Download error: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to download package: {str(e)}")


class BenchmarkRequest(BaseModel):
    """Request model for benchmarking algorithms"""
    algorithms: List[Dict[str, Any]]
    scenario: Optional[str] = "all"
    runs_per_scenario: Optional[int] = 3


@app.post("/api/algorithms/benchmark")
async def benchmark_algorithms(request: BenchmarkRequest):
    """
    Benchmark one or more algorithms

    Tests algorithms across different scenarios and returns performance metrics
    """
    try:
        if not request.algorithms:
            raise HTTPException(status_code=400, detail="No algorithms provided")

        benchmark = AlgorithmBenchmark()
        all_results = []

        for algo in request.algorithms:
            results = benchmark.benchmark_algorithm(
                algorithm=algo,
                scenario_name=request.scenario,
                runs_per_scenario=request.runs_per_scenario
            )
            all_results.append({
                "algorithm_id": algo.get('id'),
                "algorithm_name": algo.get('name'),
                "results": [
                    {
                        "test_scenario": r.test_scenario,
                        "execution_time_ms": round(r.execution_time_ms, 2),
                        "success_rate": round(r.success_rate, 3),
                        "path_length": round(r.path_length, 2) if r.path_length else None,
                        "path_smoothness": round(r.path_smoothness, 3) if r.path_smoothness else None,
                        "collision_count": r.collision_count,
                        "goal_reached": r.goal_reached,
                        "optimality_score": round(r.optimality_score, 3) if r.optimality_score else None
                    }
                    for r in results
                ]
            })

        return {
            "status": "success",
            "benchmark_results": all_results,
            "test_info": {
                "scenario": request.scenario,
                "runs_per_scenario": request.runs_per_scenario,
                "total_tests": len(all_results) * request.runs_per_scenario
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Benchmark error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


class CompareRequest(BaseModel):
    """Request model for comparing algorithms"""
    algorithms: List[Dict[str, Any]]
    scenario: Optional[str] = "all"
    runs_per_scenario: Optional[int] = 3
    criteria: Optional[str] = "overall"


@app.post("/api/algorithms/compare")
async def compare_algorithms(request: CompareRequest):
    """
    Compare multiple algorithms and rank them

    Benchmarks all algorithms and returns rankings with best algorithm recommendation
    """
    try:
        if not request.algorithms:
            raise HTTPException(status_code=400, detail="No algorithms provided")

        if len(request.algorithms) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 algorithms to compare")

        comparator = AlgorithmComparator()

        comparison = comparator.compare_algorithms(
            algorithms=request.algorithms,
            scenario=request.scenario,
            runs_per_scenario=request.runs_per_scenario
        )

        return {
            "status": "success",
            "comparison": comparison
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Comparison error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


class SelectBestRequest(BaseModel):
    """Request model for selecting best algorithm"""
    algorithms: List[Dict[str, Any]]
    criteria: Optional[str] = "overall"  # "overall", "speed", "accuracy", "robustness"


@app.post("/api/algorithms/select-best")
async def select_best_algorithm(request: SelectBestRequest):
    """
    Automatically select the best algorithm based on benchmarks

    Returns the best algorithm and comparison data
    """
    try:
        if not request.algorithms:
            raise HTTPException(status_code=400, detail="No algorithms provided")

        comparator = AlgorithmComparator()

        best_algo, comparison = comparator.select_best_algorithm(
            algorithms=request.algorithms,
            criteria=request.criteria
        )

        return {
            "status": "success",
            "best_algorithm": best_algo,
            "comparison": comparison,
            "criteria": request.criteria
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Selection error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Selection failed: {str(e)}")


@app.post("/api/urdf/parse")
async def parse_urdf(file: UploadFile = File(...)):
    """
    Upload and parse URDF file

    Accepts URDF XML file and returns parsed robot structure
    with Three.js scene configuration
    """
    try:
        # Validate file type
        if not file.filename.endswith('.urdf'):
            raise HTTPException(status_code=400, detail="File must be a .urdf file")

        # Save uploaded file to temp location
        temp_dir = tempfile.mkdtemp(prefix='urdf_upload_')
        temp_path = os.path.join(temp_dir, file.filename)

        with open(temp_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        print(f"📁 Uploaded URDF: {file.filename}")

        # Parse URDF
        parser = URDFParser()
        urdf_data = parser.parse_file(temp_path)

        # Convert to Three.js format
        converter = ThreeJSConverter()
        scene_config = converter.convert(urdf_data)

        # Cleanup temp file
        shutil.rmtree(temp_dir)

        return {
            "status": "success",
            "filename": file.filename,
            "urdf_data": urdf_data,
            "scene_config": scene_config
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ URDF parse error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to parse URDF: {str(e)}")


class URDFStringRequest(BaseModel):
    """Request model for URDF string parsing"""
    urdf_xml: str


@app.post("/api/urdf/parse-string")
async def parse_urdf_string(request: URDFStringRequest):
    """
    Parse URDF from XML string

    Accepts URDF XML as string and returns parsed structure
    """
    try:
        parser = URDFParser()
        urdf_data = parser.parse_string(request.urdf_xml)

        converter = ThreeJSConverter()
        scene_config = converter.convert(urdf_data)

        return {
            "status": "success",
            "urdf_data": urdf_data,
            "scene_config": scene_config
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ URDF parse error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse URDF: {str(e)}")


@app.get("/api/urdf/templates")
async def list_urdf_templates():
    """
    List available pre-built URDF robot templates

    Returns templates for common robots like UR5, TurtleBot3, etc.
    """
    templates = [
        {
            "id": "simple_arm",
            "name": "Simple 2-DOF Arm",
            "description": "Basic 2 degree-of-freedom robotic arm",
            "type": "arm",
            "dof": 2,
            "thumbnail": "/api/urdf/templates/simple_arm/thumbnail.png"
        },
        {
            "id": "mobile_robot",
            "name": "Differential Drive Robot",
            "description": "Simple mobile robot with differential drive",
            "type": "mobile",
            "dof": 0,
            "thumbnail": "/api/urdf/templates/mobile_robot/thumbnail.png"
        },
        {
            "id": "ur5",
            "name": "Universal Robots UR5",
            "description": "Industrial 6-DOF robotic arm",
            "type": "arm",
            "dof": 6,
            "thumbnail": "/api/urdf/templates/ur5/thumbnail.png"
        },
        {
            "id": "turtlebot3",
            "name": "TurtleBot3",
            "description": "Popular ROS educational robot",
            "type": "mobile",
            "dof": 0,
            "thumbnail": "/api/urdf/templates/turtlebot3/thumbnail.png"
        }
    ]

    return {
        "status": "success",
        "templates": templates,
        "count": len(templates)
    }


@app.get("/api/urdf/templates/{template_id}")
async def get_urdf_template(template_id: str):
    """
    Get URDF template by ID

    Returns the URDF XML for a specific template
    """
    # Simple URDF templates
    templates = {
        "simple_arm": """<?xml version="1.0"?>
<robot name="simple_arm">
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.2"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <link name="link1">
    <visual>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.1 0.5"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.4 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.1 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>
  </joint>

  <link name="link2">
    <visual>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.08 0.08 0.4"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <geometry>
        <box size="0.08 0.08 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  </joint>
</robot>""",
        "mobile_robot": """<?xml version="1.0"?>
<robot name="mobile_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.4 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.2 0" rpy="1.57 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.2 0" rpy="1.57 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>"""
    }

    if template_id not in templates:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")

    try:
        urdf_xml = templates[template_id]

        # Parse and convert
        parser = URDFParser()
        urdf_data = parser.parse_string(urdf_xml)

        converter = ThreeJSConverter()
        scene_config = converter.convert(urdf_data)

        return {
            "status": "success",
            "template_id": template_id,
            "urdf_xml": urdf_xml,
            "urdf_data": urdf_data,
            "scene_config": scene_config
        }

    except Exception as e:
        print(f"❌ Template error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load template: {str(e)}")


class RobotBuildRequest(BaseModel):
    """Request model for building custom robots"""
    robot_name: str
    links: List[Dict[str, Any]]
    joints: List[Dict[str, Any]]


@app.post("/api/urdf/generate")
async def generate_urdf(request: RobotBuildRequest):
    """
    Generate URDF from programmatic robot description

    Build custom robots by specifying links and joints in JSON format
    """
    try:
        gen = URDFGenerator(request.robot_name)

        # Add links
        for link_data in request.links:
            link = LinkBuilder(link_data['name'])

            # Add visual geometry
            if 'visual' in link_data:
                visual = link_data['visual']
                geom_type = visual.get('geometry', {}).get('type')

                if geom_type == 'box':
                    link.with_box_visual(
                        visual['geometry']['size'],
                        visual.get('color'),
                        visual.get('origin')
                    )
                elif geom_type == 'cylinder':
                    link.with_cylinder_visual(
                        visual['geometry']['radius'],
                        visual['geometry']['length'],
                        visual.get('color'),
                        visual.get('origin')
                    )
                elif geom_type == 'sphere':
                    link.with_sphere_visual(
                        visual['geometry']['radius'],
                        visual.get('color'),
                        visual.get('origin')
                    )
                elif geom_type == 'mesh':
                    link.with_mesh_visual(
                        visual['geometry']['filename'],
                        visual['geometry'].get('scale'),
                        visual.get('origin')
                    )

            # Add collision
            if 'collision' in link_data:
                link.with_collision(
                    link_data['collision']['geometry'],
                    link_data['collision'].get('origin')
                )

            # Add inertia
            if 'inertia' in link_data:
                link.with_inertia(
                    link_data['inertia']['mass'],
                    link_data['inertia']['tensor'],
                    link_data['inertia'].get('origin')
                )

            gen.add_link(link)

        # Add joints
        for joint_data in request.joints:
            joint_type = joint_data['type']

            if joint_type == 'fixed':
                joint = JointBuilder.fixed(
                    joint_data['name'],
                    joint_data['parent'],
                    joint_data['child'],
                    joint_data.get('origin')
                )
            elif joint_type == 'revolute':
                joint = JointBuilder.revolute(
                    joint_data['name'],
                    joint_data['parent'],
                    joint_data['child'],
                    joint_data.get('origin'),
                    joint_data.get('axis'),
                    joint_data.get('limits', {}).get('lower', -3.14),
                    joint_data.get('limits', {}).get('upper', 3.14),
                    joint_data.get('limits', {}).get('effort', 100.0),
                    joint_data.get('limits', {}).get('velocity', 1.0)
                )
            elif joint_type == 'continuous':
                joint = JointBuilder.continuous(
                    joint_data['name'],
                    joint_data['parent'],
                    joint_data['child'],
                    joint_data.get('origin'),
                    joint_data.get('axis'),
                    joint_data.get('limits', {}).get('effort', 100.0),
                    joint_data.get('limits', {}).get('velocity', 1.0)
                )
            elif joint_type == 'prismatic':
                joint = JointBuilder.prismatic(
                    joint_data['name'],
                    joint_data['parent'],
                    joint_data['child'],
                    joint_data.get('origin'),
                    joint_data.get('axis'),
                    joint_data.get('limits', {}).get('lower', 0.0),
                    joint_data.get('limits', {}).get('upper', 1.0),
                    joint_data.get('limits', {}).get('effort', 100.0),
                    joint_data.get('limits', {}).get('velocity', 1.0)
                )
            else:
                raise ValueError(f"Unsupported joint type: {joint_type}")

            gen.add_joint(joint)

        # Generate URDF XML
        urdf_xml = gen.generate()

        # Parse and convert to scene config
        parser = URDFParser()
        urdf_data = parser.parse_string(urdf_xml)

        converter = ThreeJSConverter()
        scene_config = converter.convert(urdf_data)

        return {
            "status": "success",
            "robot_name": request.robot_name,
            "urdf_xml": urdf_xml,
            "urdf_data": urdf_data,
            "scene_config": scene_config
        }

    except Exception as e:
        print(f"❌ URDF generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate URDF: {str(e)}")


@app.get("/api/urdf/presets")
async def list_robot_presets():
    """
    List available robot presets

    Returns preset robot templates that can be instantiated
    """
    return {
        "status": "success",
        "presets": [
            {
                "id": "simple_arm",
                "name": "Simple 2-DOF Arm",
                "description": "Basic articulated arm with 2 revolute joints",
                "type": "arm",
                "dof": 2
            },
            {
                "id": "mobile_base",
                "name": "Differential Drive Robot",
                "description": "Mobile robot with two wheels",
                "type": "mobile",
                "dof": 0
            }
        ]
    }


@app.get("/api/urdf/presets/{preset_id}")
async def generate_robot_preset(preset_id: str):
    """
    Generate a robot from preset template

    Instantiates a preset robot and returns URDF + scene config
    """
    try:
        # Generate preset
        if preset_id == "simple_arm":
            gen = RobotPresets.simple_arm()
        elif preset_id == "mobile_base":
            gen = RobotPresets.mobile_base()
        else:
            raise HTTPException(status_code=404, detail=f"Preset not found: {preset_id}")

        # Generate URDF
        urdf_xml = gen.generate()

        # Parse and convert
        parser = URDFParser()
        urdf_data = parser.parse_string(urdf_xml)

        converter = ThreeJSConverter()
        scene_config = converter.convert(urdf_data)

        return {
            "status": "success",
            "preset_id": preset_id,
            "urdf_xml": urdf_xml,
            "urdf_data": urdf_data,
            "scene_config": scene_config
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Preset generation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate preset: {str(e)}")


# ================== MULTI-ROBOT ENDPOINTS ==================

class AddRobotRequest(BaseModel):
    """Request to add robot to multi-robot scene"""
    scene_config: Dict[str, Any]
    name: Optional[str] = None
    position: Optional[List[float]] = None
    orientation: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


@app.post("/api/multi-robot/add")
async def add_robot_to_scene(request: AddRobotRequest):
    """Add a robot to the multi-robot scene"""
    try:
        manager = get_multi_robot_manager()

        robot = manager.add_robot(
            scene_config=request.scene_config,
            name=request.name,
            position=request.position,
            orientation=request.orientation,
            metadata=request.metadata
        )

        return {
            "status": "success",
            "message": f"Robot {robot.name} added to scene",
            "robot": robot.to_dict(),
            "scene": manager.get_combined_scene()
        }

    except Exception as e:
        print(f"❌ Failed to add robot: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/multi-robot/scene")
async def get_multi_robot_scene():
    """Get the combined multi-robot scene"""
    try:
        manager = get_multi_robot_manager()
        scene = manager.get_combined_scene()

        return {
            "status": "success",
            "scene": scene
        }

    except Exception as e:
        print(f"❌ Failed to get scene: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/multi-robot/robots")
async def list_robots():
    """List all robots in the scene"""
    try:
        manager = get_multi_robot_manager()
        robots = manager.list_robots()

        return {
            "status": "success",
            "count": len(robots),
            "robots": [robot.to_dict() for robot in robots]
        }

    except Exception as e:
        print(f"❌ Failed to list robots: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/multi-robot/robots/{robot_id}")
async def get_robot(robot_id: str):
    """Get specific robot by ID"""
    try:
        manager = get_multi_robot_manager()
        robot = manager.get_robot(robot_id)

        if not robot:
            raise HTTPException(status_code=404, detail=f"Robot not found: {robot_id}")

        return {
            "status": "success",
            "robot": robot.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Failed to get robot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/multi-robot/robots/{robot_id}")
async def remove_robot(robot_id: str):
    """Remove robot from scene"""
    try:
        manager = get_multi_robot_manager()
        success = manager.remove_robot(robot_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Robot not found: {robot_id}")

        return {
            "status": "success",
            "message": f"Robot {robot_id} removed",
            "scene": manager.get_combined_scene()
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Failed to remove robot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class UpdatePositionRequest(BaseModel):
    """Request to update robot position"""
    position: List[float]
    orientation: Optional[List[float]] = None
    check_collision: bool = True


@app.put("/api/multi-robot/robots/{robot_id}/position")
async def update_robot_position(robot_id: str, request: UpdatePositionRequest):
    """Update robot position with optional collision checking"""
    try:
        manager = get_multi_robot_manager()

        if robot_id not in manager.robots:
            raise HTTPException(status_code=404, detail=f"Robot not found: {robot_id}")

        # Check for collision if requested
        if request.check_collision:
            if manager.check_collision(robot_id, request.position):
                # Try to find safe position
                safe_position = manager.suggest_safe_position(robot_id, request.position)

                if safe_position:
                    return {
                        "status": "warning",
                        "message": "Collision detected, suggesting alternative position",
                        "suggested_position": safe_position,
                        "original_position": request.position
                    }
                else:
                    raise HTTPException(
                        status_code=409,
                        detail="Position would cause collision and no safe alternative found"
                    )

        # Update position
        manager.update_robot_position(robot_id, request.position, request.orientation)

        return {
            "status": "success",
            "message": "Robot position updated",
            "robot": manager.get_robot(robot_id).to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Failed to update position: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/multi-robot/clear")
async def clear_scene():
    """Clear all robots from the scene"""
    try:
        manager = get_multi_robot_manager()
        manager.clear_scene()

        return {
            "status": "success",
            "message": "Scene cleared"
        }

    except Exception as e:
        print(f"❌ Failed to clear scene: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/multi-robot/bounds")
async def get_scene_bounds():
    """Get bounding box of all robots"""
    try:
        manager = get_multi_robot_manager()
        bounds = manager.get_scene_bounds()

        return {
            "status": "success",
            "bounds": bounds
        }

    except Exception as e:
        print(f"❌ Failed to get bounds: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================== ISAAC LAB / MODAL ENDPOINTS ==================

try:
    from isaac_lab.scene_converter import convert_scene_to_isaac, suggest_training_task
    from isaac_lab.webrtc_server import (
        create_webrtc_session,
        handle_webrtc_offer,
        close_webrtc_session
    )
    ISAAC_LAB_AVAILABLE = True
except ImportError:
    try:
        from backend.isaac_lab.scene_converter import convert_scene_to_isaac, suggest_training_task
        from backend.isaac_lab.webrtc_server import (
            create_webrtc_session,
            handle_webrtc_offer,
            close_webrtc_session
        )
        ISAAC_LAB_AVAILABLE = True
    except ImportError:
        ISAAC_LAB_AVAILABLE = False
        print("⚠️  Isaac Lab module not available - Modal simulation disabled")


class RunIsaacSimulationRequest(BaseModel):
    """Request to run Isaac Lab simulation on Modal"""
    scene_config: Dict[str, Any]
    duration: float = 10.0
    record_video: bool = True
    headless: bool = True
    fps: int = 30
    stream_session_id: Optional[str] = None


@app.post("/api/isaac-lab/simulate")
async def run_isaac_simulation(request: RunIsaacSimulationRequest):
    """
    Run Isaac Lab simulation on Modal GPU infrastructure

    Converts scene config to Isaac Lab format and runs simulation
    on cloud GPU with video recording.
    """
    if not ISAAC_LAB_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Isaac Lab integration not available"
        )

    try:
        # Convert scene to Isaac Lab format
        isaac_scene = convert_scene_to_isaac(request.scene_config)

        print(f"Starting Isaac Lab simulation on Modal (duration: {request.duration}s)")

        # Import Modal function dynamically
        try:
            from isaac_lab.modal_runner import stub, run_isaac_simulation as modal_sim
        except ImportError:
            from backend.isaac_lab.modal_runner import stub, run_isaac_simulation as modal_sim

        # Run simulation on Modal (this blocks until complete)
        with stub.run():
            result = modal_sim.remote(
                scene_config=isaac_scene,
                duration=request.duration,
                record_video=request.record_video,
                headless=request.headless,
                fps=request.fps
            )

        if not result.get('success'):
            raise HTTPException(
                status_code=500,
                detail=f"Simulation failed: {result.get('error', 'Unknown error')}"
            )

        print(f"Isaac Lab simulation completed: {result.get('metrics')}")

        return {
            "status": "success",
            "result": result,
            "metrics": result.get('metrics'),
            "video_path": result.get('video_path')
        }

    except Exception as e:
        print(f"❌ Isaac Lab simulation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {str(e)}"
        )


class TrainIsaacPolicyRequest(BaseModel):
    """Request to train robot policy with Isaac Lab"""
    task_name: str
    num_envs: int = 2048
    max_iterations: int = 1000
    algorithm: str = "PPO"
    headless: bool = True
    scene_config: Optional[Dict[str, Any]] = None


@app.post("/api/isaac-lab/train")
async def train_isaac_policy(request: TrainIsaacPolicyRequest):
    """
    Train robot policy using Isaac Lab with RL

    Runs training on Modal GPU with specified algorithm (PPO, SAC, or RSL).
    For pre-built tasks, use task_name. For custom scenes, provide scene_config.
    """
    if not ISAAC_LAB_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Isaac Lab integration not available"
        )

    try:
        print(f"Starting Isaac Lab training: {request.task_name}")
        print(f"Config: {request.num_envs} envs, {request.max_iterations} iterations, {request.algorithm}")

        # Import Modal function dynamically
        try:
            from isaac_lab.modal_runner import stub, train_isaac_policy as modal_train
        except ImportError:
            from backend.isaac_lab.modal_runner import stub, train_isaac_policy as modal_train

        # Run training on Modal (this blocks until complete or timeout)
        with stub.run():
            result = modal_train.remote(
                task_name=request.task_name,
                num_envs=request.num_envs,
                max_iterations=request.max_iterations,
                algorithm=request.algorithm,
                headless=request.headless
            )

        if not result.get('success'):
            raise HTTPException(
                status_code=500,
                detail=f"Training failed: {result.get('error', 'Unknown error')}"
            )

        print(f"Training completed: {result.get('model_path')}")

        return {
            "status": "success",
            "result": result,
            "model_path": result.get('model_path'),
            "task": request.task_name,
            "algorithm": request.algorithm
        }

    except Exception as e:
        print(f"❌ Isaac Lab training error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )


class SuggestTaskRequest(BaseModel):
    """Request to suggest Isaac Lab task from scene"""
    scene_config: Dict[str, Any]


@app.post("/api/isaac-lab/suggest-task")
async def suggest_isaac_task(request: SuggestTaskRequest):
    """
    Suggest appropriate Isaac Lab pre-built task based on scene configuration

    Analyzes the scene and recommends a task like 'Isaac-Reach-Franka-v0'
    """
    if not ISAAC_LAB_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Isaac Lab integration not available"
        )

    try:
        task_name = suggest_training_task(request.scene_config)

        return {
            "status": "success",
            "suggested_task": task_name,
            "scene_type": request.scene_config.get('robot', {}).get('type')
        }

    except Exception as e:
        print(f"❌ Task suggestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/isaac-lab/tasks")
async def list_isaac_tasks():
    """
    List available pre-built Isaac Lab tasks

    Returns common tasks for manipulation, navigation, and locomotion
    """
    tasks = [
        {
            "id": "isaac-reach-franka",
            "name": "Isaac-Reach-Franka-v0",
            "description": "Franka robot reaching task",
            "type": "manipulation",
            "robot": "robotic_arm",
            "difficulty": "easy"
        },
        {
            "id": "isaac-lift-cube-franka",
            "name": "Isaac-Lift-Cube-Franka-v0",
            "description": "Franka robot cube lifting task",
            "type": "manipulation",
            "robot": "robotic_arm",
            "difficulty": "medium"
        },
        {
            "id": "isaac-velocity-flat-anymal",
            "name": "Isaac-Velocity-Flat-Anymal-C-v0",
            "description": "Quadruped navigation on flat terrain",
            "type": "navigation",
            "robot": "quadruped",
            "difficulty": "medium"
        },
        {
            "id": "isaac-velocity-rough-anymal",
            "name": "Isaac-Velocity-Rough-Anymal-C-v0",
            "description": "Quadruped navigation on rough terrain",
            "type": "navigation",
            "robot": "quadruped",
            "difficulty": "hard"
        },
        {
            "id": "isaac-velocity-humanoid",
            "name": "Isaac-Velocity-Rough-Humanoid-v0",
            "description": "Humanoid locomotion task",
            "type": "locomotion",
            "robot": "humanoid",
            "difficulty": "hard"
        }
    ]

    return {
        "status": "success",
        "tasks": tasks,
        "count": len(tasks)
    }


# ================== WEBRTC STREAMING ENDPOINTS ==================

@app.post("/api/webrtc/session")
async def create_streaming_session(metadata: Optional[Dict[str, Any]] = None):
    """
    Create WebRTC streaming session for Isaac Lab video

    Returns session_id for subsequent offer/answer exchange
    """
    if not ISAAC_LAB_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="WebRTC streaming not available"
        )

    try:
        result = await create_webrtc_session(metadata)
        return result

    except Exception as e:
        print(f"❌ Session creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class WebRTCOfferRequest(BaseModel):
    """WebRTC offer from client"""
    session_id: str
    offer: Dict[str, str]  # {sdp, type}
    video_source: Optional[str] = None


@app.post("/api/webrtc/offer")
async def handle_offer(request: WebRTCOfferRequest):
    """
    Handle WebRTC offer from client

    Processes client's SDP offer and returns SDP answer
    """
    if not ISAAC_LAB_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="WebRTC streaming not available"
        )

    try:
        result = await handle_webrtc_offer(
            session_id=request.session_id,
            offer=request.offer,
            video_source=request.video_source
        )

        return result

    except Exception as e:
        print(f"❌ Offer handling error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/webrtc/session/{session_id}")
async def close_streaming_session(session_id: str):
    """Close WebRTC streaming session"""
    if not ISAAC_LAB_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="WebRTC streaming not available"
        )

    try:
        result = await close_webrtc_session(session_id)
        return result

    except Exception as e:
        print(f"❌ Session close error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================== TRAINING PROGRESS SSE ENDPOINTS ==================

try:
    from isaac_lab.training_tracker import get_training_tracker
    TRAINING_TRACKER_AVAILABLE = True
except ImportError:
    try:
        from backend.isaac_lab.training_tracker import get_training_tracker
        TRAINING_TRACKER_AVAILABLE = True
    except ImportError:
        TRAINING_TRACKER_AVAILABLE = False
        print("⚠️  Training tracker not available")


@app.get("/api/training/progress/{training_id}")
async def stream_training_progress(training_id: str):
    """
    Stream training progress via Server-Sent Events (SSE)

    Returns real-time training updates including iteration progress,
    metrics, and completion status.
    """
    if not TRAINING_TRACKER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Training tracker not available"
        )

    async def event_generator():
        """Generate SSE events"""
        tracker = get_training_tracker()

        try:
            async for progress_update in tracker.stream_progress(training_id):
                # Format as SSE
                data = json.dumps(progress_update)
                yield f"data: {data}\n\n"

        except Exception as e:
            error_data = json.dumps({'error': str(e)})
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/training/start")
async def start_training_with_progress(request: TrainIsaacPolicyRequest):
    """
    Start Isaac Lab training with progress tracking

    Creates a training session and returns training_id for progress streaming.
    Training runs in background.
    """
    if not ISAAC_LAB_AVAILABLE or not TRAINING_TRACKER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Training not available"
        )

    try:
        tracker = get_training_tracker()

        # Create training session
        training_id = tracker.create_session(
            task_name=request.task_name,
            total_iterations=request.max_iterations,
            metadata={
                'algorithm': request.algorithm,
                'num_envs': request.num_envs
            }
        )

        # Start training in background
        asyncio.create_task(
            run_training_with_progress(training_id, request)
        )

        return {
            "status": "started",
            "training_id": training_id,
            "task": request.task_name,
            "message": "Training started. Use /api/training/progress/{training_id} to stream progress."
        }

    except Exception as e:
        print(f"❌ Training start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_training_with_progress(training_id: str, request: TrainIsaacPolicyRequest):
    """
    Run training and update progress tracker

    This runs in the background and updates the tracker periodically
    """
    import time

    tracker = get_training_tracker()

    try:
        # Import Modal function
        try:
            from isaac_lab.modal_runner import stub, train_isaac_policy as modal_train
        except ImportError:
            from backend.isaac_lab.modal_runner import stub, train_isaac_policy as modal_train

        await tracker.update_progress(
            training_id,
            status='running',
            message=f"Launching Modal container for {request.task_name}..."
        )

        start_time = time.time()

        # Simulate progress updates (since Modal blocks)
        # In a real implementation, you'd parse Modal logs for actual progress
        async def simulate_progress():
            for i in range(1, request.max_iterations + 1):
                await asyncio.sleep(0.5)  # Simulate iteration time

                await tracker.update_progress(
                    training_id,
                    current_iteration=i,
                    elapsed_time=time.time() - start_time,
                    metrics={
                        'iteration': i,
                        'simulated': True
                    },
                    message=f"Training iteration {i}/{request.max_iterations}"
                )

        # Start progress simulation
        progress_task = asyncio.create_task(simulate_progress())

        # Run actual training on Modal (blocks)
        with stub.run():
            result = modal_train.remote(
                task_name=request.task_name,
                num_envs=request.num_envs,
                max_iterations=request.max_iterations,
                algorithm=request.algorithm,
                headless=request.headless
            )

        # Cancel progress simulation
        progress_task.cancel()

        if result.get('success'):
            await tracker.set_completed(
                training_id,
                model_path=result.get('model_path'),
                final_metrics={
                    'final': True,
                    'model_path': result.get('model_path')
                }
            )
        else:
            await tracker.set_failed(training_id, result.get('error', 'Unknown error'))

    except Exception as e:
        print(f"❌ Training error: {e}")
        await tracker.set_failed(training_id, str(e))


@app.get("/api/training/status/{training_id}")
async def get_training_status(training_id: str):
    """Get current training status (non-streaming)"""
    if not TRAINING_TRACKER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Training tracker not available"
        )

    tracker = get_training_tracker()
    progress = tracker.get_progress(training_id)

    if not progress:
        raise HTTPException(status_code=404, detail="Training session not found")

    return progress.to_dict()


# ================== MODEL DOWNLOAD ENDPOINTS ==================

@app.get("/api/models/download/{training_id}")
async def download_trained_model(training_id: str, format: str = "pytorch"):
    """
    Download trained model from Modal volume

    Args:
        training_id: Training session ID
        format: Model format ('pytorch', 'onnx', 'torchscript')

    Returns:
        Model file for download
    """
    if not TRAINING_TRACKER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Training tracker not available"
        )

    try:
        tracker = get_training_tracker()
        progress = tracker.get_progress(training_id)

        if not progress:
            raise HTTPException(status_code=404, detail="Training session not found")

        if progress.status != 'completed':
            raise HTTPException(
                status_code=400,
                detail=f"Training not completed. Current status: {progress.status}"
            )

        model_path = progress.metrics.get('model_path')
        if not model_path:
            raise HTTPException(status_code=404, detail="No model path found for this training")

        # In a real implementation, you would:
        # 1. Download model from Modal volume
        # 2. Convert to requested format if needed
        # 3. Return the file

        # For now, return info about the model
        return {
            "status": "success",
            "training_id": training_id,
            "model_path": model_path,
            "format": format,
            "message": "Model ready for download",
            "download_url": f"/api/models/file/{training_id}",
            "note": "Direct download from Modal volume not yet implemented. Model is stored at: " + model_path
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Model download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ModelConversionRequest(BaseModel):
    """Request to convert model format"""
    model_path: str
    from_format: str = "pytorch"
    to_format: str = "onnx"


@app.post("/api/models/convert")
async def convert_model_format(request: ModelConversionRequest):
    """
    Convert model to different format

    Supports: PyTorch (.pth) -> ONNX (.onnx) -> TorchScript (.pt)
    """
    try:
        print(f"Converting model from {request.from_format} to {request.to_format}")

        # Placeholder for model conversion
        # In a real implementation, you would:
        # 1. Load the model from Modal volume
        # 2. Convert using torch.onnx.export() or torch.jit.script()
        # 3. Save converted model
        # 4. Return download link

        return {
            "status": "success",
            "message": f"Model conversion from {request.from_format} to {request.to_format} queued",
            "original_path": request.model_path,
            "converted_path": f"{request.model_path}.{request.to_format}",
            "note": "Model conversion not yet fully implemented"
        }

    except Exception as e:
        print(f"❌ Model conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/list")
async def list_trained_models():
    """
    List all trained models available for download

    Returns models from completed training sessions
    """
    if not TRAINING_TRACKER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Training tracker not available"
        )

    try:
        tracker = get_training_tracker()

        models = []
        for training_id, progress in tracker.training_sessions.items():
            if progress.status == 'completed' and 'model_path' in progress.metrics:
                models.append({
                    'training_id': training_id,
                    'model_path': progress.metrics['model_path'],
                    'task': progress.message,
                    'completed_at': progress.timestamp,
                    'metrics': progress.metrics
                })

        return {
            "status": "success",
            "models": models,
            "count": len(models)
        }

    except Exception as e:
        print(f"❌ Model list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# GENESIS SIMULATION API
# ============================================================================

# Import Genesis modules
try:
    from genesis_service import (
        GenesisSimulation,
        GenesisConfig,
        BackendType,
        RobotType,
        get_simulation,
        reset_simulation,
    )
    from genesis_renderer import (
        VideoStreamer,
        StreamConfig,
        StreamQuality,
    )
    GENESIS_AVAILABLE = True
except ImportError:
    try:
        from backend.genesis_service import (
            GenesisSimulation,
            GenesisConfig,
            BackendType,
            RobotType,
            get_simulation,
            reset_simulation,
        )
        from backend.genesis_renderer import (
            VideoStreamer,
            StreamConfig,
            StreamQuality,
        )
        GENESIS_AVAILABLE = True
    except ImportError:
        GENESIS_AVAILABLE = False
        print("⚠️  Genesis not available - install with: pip install genesis-world")

from fastapi import WebSocket, WebSocketDisconnect

# Global video streamer
_video_streamer: Optional[VideoStreamer] = None


class GenesisInitRequest(BaseModel):
    """Request to initialize Genesis simulation"""
    backend: str = "auto"  # auto, metal, cuda, cpu, vulkan
    fps: int = 60
    render_width: int = 1920
    render_height: int = 1080
    stream_quality: str = "medium"  # draft, medium, high, ultra


class RobotAddRequest(BaseModel):
    """Request to add a robot to the simulation"""
    robot_id: str
    robot_type: str  # mobile, arm, drone, franka, go2
    position: List[float] = [0, 0, 0.5]


class ObstacleAddRequest(BaseModel):
    """Request to add an obstacle to the simulation"""
    obstacle_id: str
    position: List[float]
    size: List[float]


class SimulationControlRequest(BaseModel):
    """Request to control simulation"""
    action: str  # start, stop, reset, step


@app.get("/api/genesis/status")
async def genesis_status():
    """Check if Genesis is available and get status"""
    if not GENESIS_AVAILABLE:
        return {
            "available": False,
            "message": "Genesis not installed. Install with: pip install genesis-world"
        }

    sim = get_simulation()

    return {
        "available": True,
        "initialized": sim.is_initialized,
        "running": sim.is_running,
        "step_count": sim.step_count,
        "robot_count": len(sim.robots),
        "obstacle_count": len(sim.obstacles),
    }


@app.get("/api/genesis/models")
async def genesis_available_models():
    """Get all available robot models from local assets"""
    if not GENESIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Genesis not available"
        )

    try:
        from genesis_service import GenesisSimulation
        models = GenesisSimulation.discover_available_models()

        return {
            "status": "success",
            "models": models,
            "total_urdf": len(models["urdf"]),
            "total_xml": len(models["xml"]),
        }
    except Exception as e:
        logger.error(f"Failed to discover models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to discover models: {str(e)}"
        )


@app.post("/api/genesis/init")
async def genesis_init(request: GenesisInitRequest):
    """Initialize Genesis simulation with configuration"""
    if not GENESIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Genesis not available"
        )

    try:
        # Parse backend type
        backend_map = {
            "auto": BackendType.METAL,  # Will auto-detect
            "metal": BackendType.METAL,
            "cuda": BackendType.CUDA,
            "cpu": BackendType.CPU,
            "vulkan": BackendType.VULKAN,
        }
        backend = backend_map.get(request.backend.lower(), BackendType.METAL)

        # Parse quality
        quality_map = {
            "draft": StreamQuality.DRAFT,
            "medium": StreamQuality.MEDIUM,
            "high": StreamQuality.HIGH,
            "ultra": StreamQuality.ULTRA,
        }
        quality = quality_map.get(request.stream_quality.lower(), StreamQuality.MEDIUM)

        # Create config
        config = GenesisConfig(
            backend=backend,
            fps=request.fps,
            render_width=request.render_width,
            render_height=request.render_height,
            show_viewer=False,  # Headless mode
        )

        # Reset existing simulation if any
        reset_simulation()

        # Create new simulation
        sim = get_simulation(config)
        sim.initialize()

        # Initialize video streamer
        global _video_streamer
        stream_config = StreamConfig.from_quality(quality)
        _video_streamer = VideoStreamer(stream_config)
        _video_streamer.start()

        print(f"✅ Genesis initialized with {backend.value} backend")

        return {
            "status": "success",
            "message": "Genesis initialized",
            "backend": backend.value,
            "config": {
                "fps": config.fps,
                "resolution": f"{config.render_width}x{config.render_height}",
                "quality": request.stream_quality,
            }
        }

    except Exception as e:
        print(f"❌ Genesis initialization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/genesis/robot/add")
async def genesis_add_robot(request: RobotAddRequest):
    """Add a robot to the Genesis simulation"""
    if not GENESIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Genesis not available")

    try:
        sim = get_simulation()

        # Parse robot type
        robot_type_map = {
            "mobile": RobotType.MOBILE_ROBOT,
            "arm": RobotType.ROBOTIC_ARM,
            "drone": RobotType.DRONE,
            "franka": RobotType.FRANKA,
            "go2": RobotType.GO2,
        }
        robot_type = robot_type_map.get(request.robot_type.lower(), RobotType.MOBILE_ROBOT)

        # Add robot
        robot = sim.add_robot(
            robot_id=request.robot_id,
            robot_type=robot_type,
            position=tuple(request.position)
        )

        if robot is None:
            raise HTTPException(status_code=500, detail="Failed to create robot")

        print(f"✅ Added robot: {request.robot_id} ({request.robot_type})")

        return {
            "status": "success",
            "robot_id": request.robot_id,
            "robot_type": request.robot_type,
            "position": request.position,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Add robot error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/genesis/obstacle/add")
async def genesis_add_obstacle(request: ObstacleAddRequest):
    """Add an obstacle to the Genesis simulation"""
    if not GENESIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Genesis not available")

    try:
        sim = get_simulation()

        obstacle = sim.add_obstacle(
            obstacle_id=request.obstacle_id,
            position=tuple(request.position),
            size=tuple(request.size)
        )

        print(f"✅ Added obstacle: {request.obstacle_id}")

        return {
            "status": "success",
            "obstacle_id": request.obstacle_id,
            "position": request.position,
            "size": request.size,
        }

    except Exception as e:
        print(f"❌ Add obstacle error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/genesis/scene/build")
async def genesis_build_scene():
    """Build the Genesis scene (required before running simulation)"""
    if not GENESIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Genesis not available")

    try:
        sim = get_simulation()
        sim.build_scene()

        print("✅ Genesis scene built")

        return {
            "status": "success",
            "message": "Scene built successfully",
            "robot_count": len(sim.robots),
            "obstacle_count": len(sim.obstacles),
        }

    except Exception as e:
        print(f"❌ Build scene error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/genesis/control")
async def genesis_control(request: SimulationControlRequest):
    """Control simulation (start, stop, reset, step)"""
    if not GENESIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Genesis not available")

    try:
        sim = get_simulation()

        if request.action == "start":
            sim.start()
            message = "Simulation started"
        elif request.action == "stop":
            sim.stop()
            message = "Simulation stopped"
        elif request.action == "reset":
            sim.reset()
            message = "Simulation reset"
        elif request.action == "step":
            state = sim.step()
            return {
                "status": "success",
                "action": "step",
                "state": state,
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

        print(f"✅ {message}")

        return {
            "status": "success",
            "action": request.action,
            "message": message,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Control error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/genesis/stream/frame")
async def genesis_get_frame():
    """Get latest frame as JPEG"""
    if not GENESIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Genesis not available")

    global _video_streamer

    if _video_streamer is None:
        raise HTTPException(status_code=400, detail="Video streamer not initialized")

    try:
        sim = get_simulation()

        # Get latest frame from simulation
        if sim.last_frame is not None and _video_streamer is not None:
            _video_streamer.add_frame(sim.last_frame)
            jpeg_bytes = _video_streamer.get_latest_frame_jpeg()

            if jpeg_bytes is not None:
                return StreamingResponse(
                    io.BytesIO(jpeg_bytes),
                    media_type="image/jpeg"
                )

        raise HTTPException(status_code=404, detail="No frame available")

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Get frame error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/genesis/stream/stats")
async def genesis_stream_stats():
    """Get video streaming statistics"""
    if not GENESIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Genesis not available")

    global _video_streamer

    if _video_streamer is None:
        return {"status": "not_initialized"}

    try:
        stats = _video_streamer.get_stats()

        return {
            "status": "running",
            "stats": stats,
        }

    except Exception as e:
        print(f"❌ Get stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/api/genesis/ws")
async def genesis_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time state updates
    Sends simulation state and receives control commands
    """
    if not GENESIS_AVAILABLE:
        await websocket.close(code=1003, reason="Genesis not available")
        return

    try:
        sim = get_simulation()
        await sim.add_websocket_client(websocket)

        print("🔌 WebSocket client connected")

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_json()

                # Handle different message types
                msg_type = data.get('type')

                if msg_type == 'control':
                    # Control simulation
                    action = data.get('action')
                    if action == 'start':
                        sim.start()
                    elif action == 'stop':
                        sim.stop()
                    elif action == 'reset':
                        sim.reset()

                elif msg_type == 'get_frame':
                    # Send latest frame
                    frame_base64 = sim.get_frame_base64()
                    if frame_base64:
                        await websocket.send_json({
                            'type': 'frame',
                            'data': frame_base64
                        })

                elif msg_type == 'ping':
                    # Respond to ping
                    await websocket.send_json({'type': 'pong'})

            except WebSocketDisconnect:
                print("🔌 WebSocket client disconnected")
                break
            except Exception as e:
                print(f"⚠️  WebSocket error: {e}")
                break

    finally:
        sim.remove_websocket_client(websocket)


# Global teleoperation server instance (initialized on demand)
teleop_server: Optional[GenesisTeleopServer] = None


def get_teleop_server():
    """Get or create teleoperation server instance"""
    global teleop_server
    if teleop_server is None:
        # Get Genesis service
        if GENESIS_AVAILABLE:
            sim = get_simulation()
            teleop_server = GenesisTeleopServer(sim)
        else:
            raise HTTPException(status_code=503, detail="Genesis not available")
    return teleop_server


@app.websocket("/api/control/teleop")
async def teleoperation_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for keyboard teleoperation with recording

    Receives keyboard state from browser and controls robots
    Supports recording demonstrations to CSV for imitation learning

    Message Types:
    - keyboard_input: {robot_id, keys: {...}}
    - start_recording: {task_name}
    - stop_recording: {}
    - reset_robot: {robot_id}
    """
    if not GENESIS_AVAILABLE:
        await websocket.close(code=1003, reason="Genesis not available")
        return

    await websocket.accept()
    teleop = get_teleop_server()

    print("🎮 Teleoperation WebSocket client connected")

    try:
        while True:
            try:
                # Receive message from browser
                data = await websocket.receive_json()
                msg_type = data.get('type')

                if msg_type == 'keyboard_input':
                    # Process keyboard state
                    robot_id = data.get('robot_id', 'robot-1')
                    pressed_keys = data.get('keys', {})

                    # Ensure robot is initialized
                    if robot_id not in teleop.robot_states:
                        # Initialize robot (infer type from Genesis service)
                        robot_type = data.get('robot_type', 'mobile')
                        teleop.initialize_robot(robot_id, robot_type)

                    # Process keys and get result
                    result = teleop.process_keyboard_input(robot_id, pressed_keys)

                    # Send back action and state
                    await websocket.send_json({
                        'type': 'teleop_update',
                        'robot_id': robot_id,
                        'result': result
                    })

                elif msg_type == 'start_recording':
                    # Start recording demonstration
                    task_name = data.get('task_name', 'demo')
                    result = teleop.start_recording(task_name)

                    await websocket.send_json({
                        'type': 'recording_started',
                        'result': result
                    })

                elif msg_type == 'stop_recording':
                    # Stop and save recording
                    result = teleop.stop_recording()

                    await websocket.send_json({
                        'type': 'recording_stopped',
                        'result': result
                    })

                elif msg_type == 'reset_robot':
                    # Reset robot to initial state
                    robot_id = data.get('robot_id', 'robot-1')
                    result = teleop.reset_robot(robot_id)

                    await websocket.send_json({
                        'type': 'robot_reset',
                        'result': result
                    })

                elif msg_type == 'get_recording_status':
                    # Get current recording status
                    status = teleop.get_recording_status()

                    await websocket.send_json({
                        'type': 'recording_status',
                        'status': status
                    })

                elif msg_type == 'ping':
                    # Heartbeat
                    await websocket.send_json({'type': 'pong'})

            except WebSocketDisconnect:
                print("🎮 Teleoperation client disconnected")
                break
            except Exception as e:
                print(f"⚠️  Teleoperation WebSocket error: {e}")
                import traceback
                traceback.print_exc()
                break

    finally:
        print("🎮 Teleoperation session ended")


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Robotics Demo API on http://localhost:8000")

    if ISAAC_LAB_AVAILABLE:
        print("✅ Isaac Lab integration enabled")

    uvicorn.run(app, host="0.0.0.0", port=8000)
