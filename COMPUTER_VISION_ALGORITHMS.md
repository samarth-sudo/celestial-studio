# Computer Vision Algorithms

The algorithm generator now supports comprehensive computer vision capabilities for robotics applications.

## Available CV Algorithms

### 1. **Object Detection** (`object_detection`)
- **Description**: YOLO-style object detection with bounding boxes and 3D position estimation
- **Based on**: "Combining YOLO11 and Depth Pro for Accurate Distance Estimation"
- **Complexity**: O(n) where n = number of objects in scene
- **Use cases**:
  - Detecting obstacles in robot's field of view
  - Identifying target objects for manipulation
  - Visual servoing and navigation

**Example Request**:
```json
{
  "description": "Detect objects in front of the robot and estimate their distance",
  "robot_type": "mobile",
  "algorithm_type": "computer_vision"
}
```

### 2. **Object Tracking** (`object_tracking`)
- **Description**: Kalman filter-based tracking for consistent object identification across frames
- **Complexity**: O(n*m) where n=current detections, m=tracked objects
- **Use cases**:
  - Following moving targets
  - Multi-object tracking in dynamic environments
  - Maintaining object identity over time

**Features**:
- Velocity estimation
- Lost track recovery (up to 30 frames)
- Automatic track association

### 3. **Feature Detection** (`feature_detection`)
- **Description**: Detects and matches distinctive visual features for localization and mapping
- **Complexity**: O(n) where n = number of scene points
- **Use cases**:
  - Visual SLAM and localization
  - Place recognition
  - Visual odometry

**Features**:
- Corner response strength computation
- Feature descriptor matching
- RANSAC outlier filtering

### 4. **Optical Flow** (`optical_flow`)
- **Description**: Estimates motion between frames (camera ego-motion or object motion)
- **Complexity**: O(n) where n = number of tracked points
- **Use cases**:
  - Visual odometry
  - Obstacle motion detection
  - Camera motion estimation

**Features**:
- Flow vector smoothing
- Ego-motion estimation (translation + rotation)
- Outlier rejection

### 5. **Semantic Segmentation** (`semantic_segmentation`)
- **Description**: Classifies different regions of the scene (floor, walls, objects, goals)
- **Complexity**: O(n) where n = number of scene elements
- **Use cases**:
  - Terrain classification
  - Navigable region identification
  - Scene understanding

**Segmentation Classes**:
- Floor
- Wall
- Obstacle
- Goal
- Robot
- Unknown

## How to Use

### Via API Endpoint

**Endpoint**: `POST /api/generate-algorithm`

**Request Body**:
```json
{
  "description": "Detect and track moving objects in the scene",
  "robot_type": "mobile",
  "algorithm_type": "computer_vision",
  "current_code": null,
  "modification_request": null
}
```

**Response**:
```json
{
  "code": "// Generated TypeScript code...",
  "parameters": [
    {
      "name": "detectionThreshold",
      "type": "number",
      "value": 0.5,
      "min": 0,
      "max": 1.5,
      "step": 0.05,
      "description": "Detection Confidence Threshold"
    }
  ],
  "algorithm_type": "computer_vision",
  "description": "Detect and track moving objects in the scene",
  "complexity": "O(n) - Linear complexity",
  "status": "success"
}
```

### Natural Language Examples

The CV algorithm generator understands natural language descriptions:

1. **Object Detection**:
   - "Detect boxes and obstacles in front of the robot"
   - "Find all targets within 5 meters and estimate their distance"
   - "Identify objects the robot should pick up"

2. **Object Tracking**:
   - "Track moving people in the warehouse"
   - "Follow the target object and predict its next position"
   - "Keep track of multiple robots in the workspace"

3. **Feature Detection**:
   - "Find distinctive visual features for localization"
   - "Detect corners and edges for mapping"
   - "Match features between consecutive frames"

4. **Optical Flow**:
   - "Estimate camera motion from visual data"
   - "Detect which objects are moving in the scene"
   - "Calculate the robot's velocity using optical flow"

5. **Semantic Segmentation**:
   - "Classify different regions of the scene"
   - "Identify navigable floor areas"
   - "Segment the scene into objects, walls, and free space"

## Integration with Robot Types

### Mobile Robots
- Object detection for obstacle avoidance
- Optical flow for visual odometry
- Semantic segmentation for path planning

### Robotic Arms
- Object detection for pick-and-place
- Feature detection for precise positioning
- Object tracking for dynamic grasping

### Drones
- Optical flow for stabilization
- Semantic segmentation for landing zone detection
- Object tracking for aerial surveillance

## Testing

Run the test suite to verify CV algorithms are working:

```bash
python3 test_cv_generation.py
```

Expected output:
```
🎉 All tests passed! Computer vision algorithms are ready.

Available CV algorithms:
  • object_detection - YOLO-style object detection
  • object_tracking - Kalman filter tracking
  • feature_detection - Corner/feature detection and matching
  • optical_flow - Motion estimation
  • semantic_segmentation - Scene segmentation
```

## Example Generated Code

### Object Detection Example

```typescript
import * as THREE from 'three'

const DETECTION_CONFIDENCE_THRESHOLD = 0.5
const DETECTION_RANGE = 10.0  // meters
const CAMERA_FOV = 75  // degrees

interface Detection {
  label: string
  confidence: number
  bbox: {x: number, y: number, width: number, height: number}
  position3D: THREE.Vector3
  distance: number
  color?: string
}

function detectObjects(
  camera: CameraState,
  objects: Array<{position: THREE.Vector3, label: string, radius: number}>
): Detection[] {
  const detections: Detection[] = []

  for (const obj of objects) {
    // Check if in camera frustum
    const toObject = obj.position.clone().sub(camera.position)
    const distance = toObject.length()

    if (distance > DETECTION_RANGE) continue

    // Project to 2D and create bounding box
    const screenPos = projectToScreen(obj.position, camera)
    if (!screenPos) continue

    const confidence = calculateConfidence(distance, obj.radius)

    if (confidence >= DETECTION_CONFIDENCE_THRESHOLD) {
      detections.push({
        label: obj.label,
        confidence,
        bbox: calculateBBox(screenPos, obj.radius, distance),
        position3D: obj.position.clone(),
        distance
      })
    }
  }

  return detections
}
```

## Hot Swapping

CV algorithms can be modified in real-time:

```json
{
  "description": "Increase detection range to 15 meters",
  "robot_type": "mobile",
  "algorithm_type": "computer_vision",
  "current_code": "// existing code...",
  "modification_request": "Increase detection range to 15 meters and lower confidence threshold"
}
```

The system will regenerate the algorithm with the requested modifications while preserving the overall structure.

## Performance Considerations

All CV algorithms are optimized for:
- **Real-time execution** at 60 FPS in browser
- **Efficient data structures** (typed arrays when applicable)
- **Minimal memory allocation** in hot loops
- **Early termination** for out-of-view objects

## Future Enhancements

Planned additions:
- [ ] Real YOLO11 model integration
- [ ] Depth estimation with monocular cameras
- [ ] Visual SLAM integration
- [ ] Gesture recognition
- [ ] QR code / marker detection
- [ ] 3D object pose estimation
