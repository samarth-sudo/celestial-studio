"""
Algorithm Templates for Celestial Studio

Research-grade algorithm templates based on 2024 robotics papers.
These templates guide Qwen when generating algorithm code.

References:
- A* + DWA: "Robot obstacle avoidance optimization by A* and DWA fusion algorithm" (PLOS One, April 2024)
- FABRIK IK: "A Combined Inverse Kinematics Algorithm Using FABRIK with Optimization"
- YOLO11: "Combining YOLO11 and Depth Pro for Accurate Distance Estimation"
"""

from typing import Dict, List


class AlgorithmTemplates:
    """Collection of algorithm templates for different robotics tasks"""

    @staticmethod
    def get_all_templates() -> Dict[str, Dict[str, str]]:
        """Get all algorithm templates organized by category"""
        return {
            "path_planning": AlgorithmTemplates.path_planning_templates(),
            "obstacle_avoidance": AlgorithmTemplates.obstacle_avoidance_templates(),
            "inverse_kinematics": AlgorithmTemplates.inverse_kinematics_templates(),
            "computer_vision": AlgorithmTemplates.computer_vision_templates(),
            "motion_control": AlgorithmTemplates.motion_control_templates(),
        }

    @staticmethod
    def path_planning_templates() -> Dict[str, str]:
        """Path planning algorithm templates"""
        return {
            "astar": """
// A* Path Planning Algorithm
// Based on: "Robot obstacle avoidance optimization by A* and DWA fusion algorithm" (PLOS One, 2024)
//
// Finds optimal path from start to goal using grid-based search
// Time Complexity: O(b^d) where b=branching factor, d=depth
// Space Complexity: O(b^d)

import * as THREE from 'three'

// Configuration parameters
const GRID_SIZE = 0.5  // meters per grid cell
const HEURISTIC_WEIGHT = 1.0  // A* heuristic weight (1.0 = optimal, >1.0 = faster but suboptimal)

interface GridCell {
  x: number
  z: number
  walkable: boolean
  g: number  // Cost from start
  h: number  // Heuristic to goal
  f: number  // Total cost (g + h)
  parent: GridCell | null
}

interface PathPlanningResult {
  path: THREE.Vector3[]
  pathLength: number
  nodesExplored: number
}

function findPath(
  start: THREE.Vector3,
  goal: THREE.Vector3,
  obstacles: Array<{position: THREE.Vector3, radius: number}>,
  worldBounds: {min: THREE.Vector3, max: THREE.Vector3}
): PathPlanningResult {
  // 1. Create grid
  const grid = createGrid(worldBounds, obstacles)

  // 2. Initialize start and goal cells
  const startCell = worldToGrid(start, worldBounds)
  const goalCell = worldToGrid(goal, worldBounds)

  // 3. A* search
  const openSet: GridCell[] = [startCell]
  const closedSet = new Set<GridCell>()
  let nodesExplored = 0

  startCell.g = 0
  startCell.h = heuristic(startCell, goalCell)
  startCell.f = startCell.h

  while (openSet.length > 0) {
    // Get node with lowest f score
    openSet.sort((a, b) => a.f - b.f)
    const current = openSet.shift()!
    nodesExplored++

    // Goal reached
    if (current.x === goalCell.x && current.z === goalCell.z) {
      const path = reconstructPath(current, worldBounds)
      return {
        path,
        pathLength: calculatePathLength(path),
        nodesExplored
      }
    }

    closedSet.add(current)

    // Check neighbors (8-connected grid for smooth paths)
    const neighbors = getNeighbors(current, grid)

    for (const neighbor of neighbors) {
      if (!neighbor.walkable || closedSet.has(neighbor)) continue

      const tentativeG = current.g + distance(current, neighbor)

      if (tentativeG < neighbor.g) {
        neighbor.parent = current
        neighbor.g = tentativeG
        neighbor.h = heuristic(neighbor, goalCell) * HEURISTIC_WEIGHT
        neighbor.f = neighbor.g + neighbor.h

        if (!openSet.includes(neighbor)) {
          openSet.push(neighbor)
        }
      }
    }
  }

  // No path found
  return { path: [], pathLength: 0, nodesExplored }
}

function heuristic(a: GridCell, b: GridCell): number {
  // Euclidean distance (admissible heuristic)
  const dx = a.x - b.x
  const dz = a.z - b.z
  return Math.sqrt(dx * dx + dz * dz)
}

function distance(a: GridCell, b: GridCell): number {
  // Actual movement cost
  const dx = Math.abs(a.x - b.x)
  const dz = Math.abs(a.z - b.z)
  // Diagonal movement costs sqrt(2), orthogonal costs 1
  return (dx && dz) ? Math.SQRT2 : 1
}

export { findPath, type PathPlanningResult }
""",
            "rrt": """
// RRT (Rapidly-exploring Random Tree) Path Planning
// Good for high-dimensional spaces and complex environments
// Time Complexity: Probabilistically complete
// Space Complexity: O(n) where n = number of samples

import * as THREE from 'three'

const MAX_ITERATIONS = 1000
const STEP_SIZE = 0.5  // meters
const GOAL_THRESHOLD = 0.3  // meters

interface RRTNode {
  position: THREE.Vector3
  parent: RRTNode | null
}

function findPathRRT(
  start: THREE.Vector3,
  goal: THREE.Vector3,
  obstacles: Array<{position: THREE.Vector3, radius: number}>,
  worldBounds: {min: THREE.Vector3, max: THREE.Vector3}
): THREE.Vector3[] {
  const tree: RRTNode[] = [{ position: start.clone(), parent: null }]

  for (let i = 0; i < MAX_ITERATIONS; i++) {
    // Sample random point (bias towards goal 10% of the time)
    const randomPoint = Math.random() < 0.1 ? goal.clone() : sampleRandomPoint(worldBounds)

    // Find nearest node in tree
    const nearest = findNearest(tree, randomPoint)

    // Extend towards random point
    const newPoint = extend(nearest.position, randomPoint, STEP_SIZE)

    // Check collision
    if (!isColliding(nearest.position, newPoint, obstacles)) {
      const newNode: RRTNode = { position: newPoint, parent: nearest }
      tree.push(newNode)

      // Check if reached goal
      if (newPoint.distanceTo(goal) < GOAL_THRESHOLD) {
        return reconstructPath(newNode)
      }
    }
  }

  return []  // Failed to find path
}

export { findPathRRT }
""",
        }

    @staticmethod
    def obstacle_avoidance_templates() -> Dict[str, str]:
        """Obstacle avoidance algorithm templates"""
        return {
            "dwa": """
// Dynamic Window Approach (DWA) for Obstacle Avoidance
// Based on: "Robot obstacle avoidance optimization by A* and DWA fusion algorithm" (PLOS One, 2024)
//
// Local trajectory optimization that considers robot dynamics
// Runs in real-time at 60 FPS for reactive obstacle avoidance
// Time Complexity: O(n * m) where n=velocity samples, m=angular velocity samples

import * as THREE from 'three'

// Configuration parameters
const MAX_SPEED = 2.0  // m/s
const MAX_ANGULAR_SPEED = 2.0  // rad/s
const MAX_ACCELERATION = 1.5  // m/s²
const MAX_ANGULAR_ACCELERATION = 3.0  // rad/s²
const PREDICTION_TIME = 2.0  // seconds to predict trajectory
const DT = 0.1  // time step for trajectory simulation
const SAFETY_MARGIN = 0.8  // meters from obstacles

const VELOCITY_SAMPLES = 10
const ANGULAR_SAMPLES = 20

// Weight factors for scoring function
const HEADING_WEIGHT = 0.4  // Align with goal direction
const CLEARANCE_WEIGHT = 0.3  // Stay away from obstacles
const VELOCITY_WEIGHT = 0.3  // Prefer higher velocities

interface DWAState {
  position: THREE.Vector3
  velocity: number  // Linear velocity (m/s)
  angularVelocity: number  // rad/s
  heading: number  // Current orientation (radians)
}

interface DWACommand {
  velocity: number
  angularVelocity: number
  predictedPath: THREE.Vector3[]
}

function calculateSafeVelocity(
  currentState: DWAState,
  goal: THREE.Vector3,
  obstacles: Array<{position: THREE.Vector3, radius: number}>,
  deltaTime: number
): DWACommand {
  // 1. Calculate dynamic window (reachable velocities)
  const dynamicWindow = calculateDynamicWindow(currentState, deltaTime)

  // 2. Sample velocities within dynamic window
  let bestCommand: DWACommand = {
    velocity: 0,
    angularVelocity: 0,
    predictedPath: []
  }
  let bestScore = -Infinity

  for (let v = dynamicWindow.minV; v <= dynamicWindow.maxV; v += (dynamicWindow.maxV - dynamicWindow.minV) / VELOCITY_SAMPLES) {
    for (let w = dynamicWindow.minW; w <= dynamicWindow.maxW; w += (dynamicWindow.maxW - dynamicWindow.minW) / ANGULAR_SAMPLES) {

      // 3. Predict trajectory for this velocity pair
      const trajectory = predictTrajectory(currentState, v, w, PREDICTION_TIME, DT)

      // 4. Check collision
      const clearance = calculateClearance(trajectory, obstacles)
      if (clearance < SAFETY_MARGIN) continue  // Skip unsafe trajectories

      // 5. Calculate score
      const headingScore = calculateHeadingScore(trajectory, goal)
      const clearanceScore = clearance / (SAFETY_MARGIN * 3)  // Normalize
      const velocityScore = v / MAX_SPEED

      const score =
        HEADING_WEIGHT * headingScore +
        CLEARANCE_WEIGHT * clearanceScore +
        VELOCITY_WEIGHT * velocityScore

      // 6. Track best command
      if (score > bestScore) {
        bestScore = score
        bestCommand = { velocity: v, angularVelocity: w, predictedPath: trajectory }
      }
    }
  }

  return bestCommand
}

function calculateDynamicWindow(state: DWAState, dt: number) {
  // Velocities reachable within one time step given acceleration limits
  return {
    minV: Math.max(0, state.velocity - MAX_ACCELERATION * dt),
    maxV: Math.min(MAX_SPEED, state.velocity + MAX_ACCELERATION * dt),
    minW: Math.max(-MAX_ANGULAR_SPEED, state.angularVelocity - MAX_ANGULAR_ACCELERATION * dt),
    maxW: Math.min(MAX_ANGULAR_SPEED, state.angularVelocity + MAX_ANGULAR_ACCELERATION * dt)
  }
}

function predictTrajectory(
  state: DWAState,
  v: number,
  w: number,
  time: number,
  dt: number
): THREE.Vector3[] {
  const trajectory: THREE.Vector3[] = []
  let x = state.position.x
  let z = state.position.z
  let theta = state.heading

  for (let t = 0; t < time; t += dt) {
    x += v * Math.cos(theta) * dt
    z += v * Math.sin(theta) * dt
    theta += w * dt
    trajectory.push(new THREE.Vector3(x, 0.5, z))
  }

  return trajectory
}

function calculateClearance(
  trajectory: THREE.Vector3[],
  obstacles: Array<{position: THREE.Vector3, radius: number}>
): number {
  let minClearance = Infinity

  for (const point of trajectory) {
    for (const obstacle of obstacles) {
      const dist = point.distanceTo(obstacle.position) - obstacle.radius
      minClearance = Math.min(minClearance, dist)
    }
  }

  return minClearance
}

function calculateHeadingScore(trajectory: THREE.Vector3[], goal: THREE.Vector3): number {
  // Score based on alignment with goal direction
  const lastPoint = trajectory[trajectory.length - 1]
  const directionToGoal = goal.clone().sub(lastPoint).normalize()
  const trajectoryDirection = lastPoint.clone().sub(trajectory[0]).normalize()

  // Dot product gives cos(angle), range [-1, 1]
  const alignment = directionToGoal.dot(trajectoryDirection)
  return (alignment + 1) / 2  // Normalize to [0, 1]
}

export { calculateSafeVelocity, type DWAState, type DWACommand }
""",
            "apf": """
// Artificial Potential Field (APF) for Obstacle Avoidance
// Simple and computationally efficient
// Good for sparse environments with few obstacles
// Time Complexity: O(n) where n = number of obstacles

import * as THREE from 'three'

const ATTRACTIVE_GAIN = 1.0  // Pull towards goal
const REPULSIVE_GAIN = 2.0  // Push away from obstacles
const INFLUENCE_DISTANCE = 2.0  // meters - obstacles beyond this don't affect robot

function calculateAPFVelocity(
  position: THREE.Vector3,
  goal: THREE.Vector3,
  obstacles: Array<{position: THREE.Vector3, radius: number}>,
  maxSpeed: number
): THREE.Vector3 {
  // Attractive force towards goal
  const attractiveForce = goal.clone().sub(position).normalize().multiplyScalar(ATTRACTIVE_GAIN)

  // Repulsive force from obstacles
  const repulsiveForce = new THREE.Vector3()

  for (const obstacle of obstacles) {
    const diff = position.clone().sub(obstacle.position)
    const distance = diff.length() - obstacle.radius

    if (distance < INFLUENCE_DISTANCE && distance > 0) {
      const magnitude = REPULSIVE_GAIN * (1.0 / distance - 1.0 / INFLUENCE_DISTANCE) * (1.0 / (distance * distance))
      repulsiveForce.add(diff.normalize().multiplyScalar(magnitude))
    }
  }

  // Combine forces
  const totalForce = attractiveForce.add(repulsiveForce)

  // Limit to max speed
  if (totalForce.length() > maxSpeed) {
    totalForce.normalize().multiplyScalar(maxSpeed)
  }

  return totalForce
}

export { calculateAPFVelocity }
""",
        }

    @staticmethod
    def inverse_kinematics_templates() -> Dict[str, str]:
        """Inverse kinematics algorithm templates"""
        return {
            "fabrik": """
// FABRIK (Forward And Backward Reaching Inverse Kinematics)
// Based on: "A Combined Inverse Kinematics Algorithm Using FABRIK with Optimization"
//
// Efficient IK solver for serial manipulators
// Time Complexity: O(n) per iteration, typically converges in 5-10 iterations
// Space Complexity: O(n) where n = number of joints

import * as THREE from 'three'

const MAX_ITERATIONS = 20
const TOLERANCE = 0.01  // meters
const JOINT_LIMITS = true  // Enforce joint angle constraints

interface JointChain {
  positions: THREE.Vector3[]  // Joint positions in 3D space
  lengths: number[]  // Link lengths
  limits?: Array<{min: number, max: number}>  // Joint angle limits (radians)
}

interface IKResult {
  jointAngles: number[]
  endEffectorPos: THREE.Vector3
  reachedTarget: boolean
  iterations: number
}

function solveIK(
  chain: JointChain,
  targetPos: THREE.Vector3,
  basePos: THREE.Vector3
): IKResult {
  const n = chain.positions.length
  let positions = chain.positions.map(p => p.clone())
  let iterations = 0

  // Check if target is reachable
  const totalLength = chain.lengths.reduce((sum, len) => sum + len, 0)
  const distanceToTarget = targetPos.distanceTo(basePos)

  if (distanceToTarget > totalLength) {
    // Target unreachable - extend fully towards target
    return extendTowardsTarget(chain, targetPos, basePos)
  }

  // FABRIK iteration
  for (iterations = 0; iterations < MAX_ITERATIONS; iterations++) {
    // Check if reached target
    const endEffector = positions[n - 1]
    if (endEffector.distanceTo(targetPos) < TOLERANCE) {
      break
    }

    // Backward reaching - start from end effector
    positions[n - 1] = targetPos.clone()

    for (let i = n - 2; i >= 0; i--) {
      const direction = positions[i].clone().sub(positions[i + 1]).normalize()
      positions[i] = positions[i + 1].clone().add(direction.multiplyScalar(chain.lengths[i]))
    }

    // Forward reaching - start from base
    positions[0] = basePos.clone()

    for (let i = 0; i < n - 1; i++) {
      const direction = positions[i + 1].clone().sub(positions[i]).normalize()
      positions[i + 1] = positions[i].clone().add(direction.multiplyScalar(chain.lengths[i]))
    }
  }

  // Convert positions to joint angles
  const jointAngles = positionsToAngles(positions, basePos)

  // Apply joint limits if enabled
  if (JOINT_LIMITS && chain.limits) {
    applyJointLimits(jointAngles, chain.limits)
  }

  return {
    jointAngles,
    endEffectorPos: positions[n - 1],
    reachedTarget: positions[n - 1].distanceTo(targetPos) < TOLERANCE,
    iterations
  }
}

function positionsToAngles(positions: THREE.Vector3[], basePos: THREE.Vector3): number[] {
  const angles: number[] = []

  for (let i = 0; i < positions.length - 1; i++) {
    const current = positions[i]
    const next = positions[i + 1]
    const direction = next.clone().sub(current)

    // Calculate angle in XZ plane (yaw)
    const angle = Math.atan2(direction.z, direction.x)
    angles.push(angle)
  }

  return angles
}

function applyJointLimits(angles: number[], limits: Array<{min: number, max: number}>) {
  for (let i = 0; i < angles.length && i < limits.length; i++) {
    angles[i] = Math.max(limits[i].min, Math.min(limits[i].max, angles[i]))
  }
}

function extendTowardsTarget(
  chain: JointChain,
  targetPos: THREE.Vector3,
  basePos: THREE.Vector3
): IKResult {
  // Target unreachable - extend arm fully in that direction
  const direction = targetPos.clone().sub(basePos).normalize()
  const positions: THREE.Vector3[] = [basePos.clone()]

  for (let i = 0; i < chain.lengths.length; i++) {
    const nextPos = positions[i].clone().add(direction.clone().multiplyScalar(chain.lengths[i]))
    positions.push(nextPos)
  }

  const jointAngles = positionsToAngles(positions, basePos)

  return {
    jointAngles,
    endEffectorPos: positions[positions.length - 1],
    reachedTarget: false,
    iterations: MAX_ITERATIONS
  }
}

export { solveIK, type JointChain, type IKResult }
""",
            "ccd": """
// CCD (Cyclic Coordinate Descent) Inverse Kinematics
// Simpler than FABRIK, iterates through joints one at a time
// Good for real-time applications with many joints
// Time Complexity: O(n) per iteration

import * as THREE from 'three'

const MAX_ITERATIONS = 15
const TOLERANCE = 0.05

function solveIKCCD(
  jointPositions: THREE.Vector3[],
  targetPos: THREE.Vector3
): number[] {
  const n = jointPositions.length
  const angles: number[] = new Array(n - 1).fill(0)

  for (let iter = 0; iter < MAX_ITERATIONS; iter++) {
    // Check if reached target
    const endEffector = jointPositions[n - 1]
    if (endEffector.distanceTo(targetPos) < TOLERANCE) {
      break
    }

    // Iterate backwards through joints
    for (let i = n - 2; i >= 0; i--) {
      const joint = jointPositions[i]
      const toEnd = endEffector.clone().sub(joint)
      const toTarget = targetPos.clone().sub(joint)

      // Calculate rotation angle
      const angle = toEnd.angleTo(toTarget)
      const axis = new THREE.Vector3().crossVectors(toEnd, toTarget).normalize()

      // Rotate all joints after this one
      for (let j = i + 1; j < n; j++) {
        jointPositions[j].sub(joint).applyAxisAngle(axis, angle).add(joint)
      }

      angles[i] += angle
    }
  }

  return angles
}

export { solveIKCCD }
""",
        }

    @staticmethod
    def computer_vision_templates() -> Dict[str, str]:
        """Computer vision algorithm templates"""
        return {
            "object_detection": """
// Object Detection with Bounding Boxes
// Based on: "Combining YOLO11 and Depth Pro for Accurate Distance Estimation"
//
// Simulated object detection for web-based robotics
// In production, would use actual YOLO11 model
// Time Complexity: O(n) where n = number of objects in scene

import * as THREE from 'three'

const DETECTION_CONFIDENCE_THRESHOLD = 0.5
const DETECTION_RANGE = 10.0  // meters
const CAMERA_FOV = 75  // degrees
const IMAGE_WIDTH = 640
const IMAGE_HEIGHT = 480

interface Detection {
  label: string
  confidence: number
  bbox: {
    x: number  // pixels
    y: number
    width: number
    height: number
  }
  position3D: THREE.Vector3  // World position
  distance: number  // meters from camera
  color?: string  // For visualization
}

interface CameraState {
  position: THREE.Vector3
  direction: THREE.Vector3  // Forward direction
  up: THREE.Vector3
}

function detectObjects(
  camera: CameraState,
  objects: Array<{
    position: THREE.Vector3
    label: string
    radius: number
    color?: string
  }>
): Detection[] {
  const detections: Detection[] = []

  for (const obj of objects) {
    // 1. Check if object is in camera frustum
    const toObject = obj.position.clone().sub(camera.position)
    const distance = toObject.length()

    if (distance > DETECTION_RANGE) continue

    // 2. Check if object is in front of camera
    const dotProduct = toObject.normalize().dot(camera.direction)
    if (dotProduct < 0.5) continue  // Not in view cone

    // 3. Project to 2D screen space
    const screenPos = projectToScreen(obj.position, camera)

    if (!screenPos) continue  // Behind camera or out of view

    // 4. Calculate bounding box
    const apparentSize = (obj.radius * IMAGE_HEIGHT) / distance
    const bbox = {
      x: Math.max(0, screenPos.x - apparentSize),
      y: Math.max(0, screenPos.y - apparentSize),
      width: Math.min(IMAGE_WIDTH, apparentSize * 2),
      height: Math.min(IMAGE_HEIGHT, apparentSize * 2)
    }

    // 5. Simulate confidence based on distance and size
    const confidence = calculateConfidence(distance, apparentSize)

    if (confidence >= DETECTION_CONFIDENCE_THRESHOLD) {
      detections.push({
        label: obj.label,
        confidence,
        bbox,
        position3D: obj.position.clone(),
        distance,
        color: obj.color || '#00ff00'
      })
    }
  }

  return detections
}

function projectToScreen(
  worldPos: THREE.Vector3,
  camera: CameraState
): {x: number, y: number} | null {
  // Simple perspective projection
  const cameraToPoint = worldPos.clone().sub(camera.position)

  // Transform to camera space
  const right = new THREE.Vector3().crossVectors(camera.direction, camera.up).normalize()
  const up = camera.up.clone()
  const forward = camera.direction.clone()

  const x = cameraToPoint.dot(right)
  const y = cameraToPoint.dot(up)
  const z = cameraToPoint.dot(forward)

  if (z <= 0) return null  // Behind camera

  // Perspective divide
  const fov = (CAMERA_FOV * Math.PI) / 180
  const f = IMAGE_HEIGHT / (2 * Math.tan(fov / 2))

  const screenX = (x / z) * f + IMAGE_WIDTH / 2
  const screenY = -(y / z) * f + IMAGE_HEIGHT / 2

  return { x: screenX, y: screenY }
}

function calculateConfidence(distance: number, apparentSize: number): number {
  // Confidence decreases with distance and small apparent size
  const distanceFactor = Math.max(0, 1 - distance / DETECTION_RANGE)
  const sizeFactor = Math.min(1, apparentSize / 50)
  return distanceFactor * sizeFactor
}

export { detectObjects, type Detection, type CameraState }
""",
            "object_tracking": """
// Object Tracking with Kalman Filter
// Tracks detected objects across frames for consistent identification
// Time Complexity: O(n*m) where n=current detections, m=tracked objects

import * as THREE from 'three'

const MAX_TRACKING_DISTANCE = 2.0  // meters
const MAX_FRAMES_LOST = 30  // Drop track after 30 frames without detection
const VELOCITY_SMOOTHING = 0.3  // Low-pass filter coefficient

interface TrackedObject {
  id: number
  label: string
  position: THREE.Vector3
  velocity: THREE.Vector3
  lastSeen: number  // Frame number
  confidence: number
  predictedPosition: THREE.Vector3
}

interface TrackingState {
  trackedObjects: TrackedObject[]
  nextId: number
  frameCount: number
}

function trackObjects(
  detections: Array<{
    label: string
    position3D: THREE.Vector3
    confidence: number
  }>,
  state: TrackingState,
  deltaTime: number
): TrackedObject[] {
  state.frameCount++

  // 1. Predict positions of existing tracks
  for (const track of state.trackedObjects) {
    track.predictedPosition = track.position.clone().add(
      track.velocity.clone().multiplyScalar(deltaTime)
    )
  }

  // 2. Associate detections with existing tracks
  const assigned = new Set<number>()
  const updatedTracks: TrackedObject[] = []

  for (const detection of detections) {
    let bestMatch: TrackedObject | null = null
    let bestDistance = MAX_TRACKING_DISTANCE

    // Find closest track with matching label
    for (let i = 0; i < state.trackedObjects.length; i++) {
      if (assigned.has(i)) continue

      const track = state.trackedObjects[i]
      if (track.label !== detection.label) continue

      const distance = track.predictedPosition.distanceTo(detection.position3D)
      if (distance < bestDistance) {
        bestDistance = distance
        bestMatch = track
      }
    }

    if (bestMatch) {
      // Update existing track
      const newVelocity = detection.position3D.clone()
        .sub(bestMatch.position)
        .divideScalar(deltaTime)

      bestMatch.velocity.lerp(newVelocity, VELOCITY_SMOOTHING)
      bestMatch.position.copy(detection.position3D)
      bestMatch.lastSeen = state.frameCount
      bestMatch.confidence = detection.confidence

      updatedTracks.push(bestMatch)
      assigned.add(state.trackedObjects.indexOf(bestMatch))
    } else {
      // Create new track
      updatedTracks.push({
        id: state.nextId++,
        label: detection.label,
        position: detection.position3D.clone(),
        velocity: new THREE.Vector3(),
        lastSeen: state.frameCount,
        confidence: detection.confidence,
        predictedPosition: detection.position3D.clone()
      })
    }
  }

  // 3. Keep tracks that were recently seen
  for (const track of state.trackedObjects) {
    const framesSinceSeen = state.frameCount - track.lastSeen
    if (framesSinceSeen < MAX_FRAMES_LOST && !updatedTracks.includes(track)) {
      updatedTracks.push(track)
    }
  }

  state.trackedObjects = updatedTracks
  return updatedTracks
}

export { trackObjects, type TrackedObject, type TrackingState }
""",
            "feature_detection": """
// Feature Detection and Matching
// Detects distinctive features in the scene for localization and mapping
// Time Complexity: O(n) where n = number of scene points

import * as THREE from 'three'

const FEATURE_DETECTION_THRESHOLD = 0.3
const FEATURE_MATCH_THRESHOLD = 0.7
const RANSAC_ITERATIONS = 100
const RANSAC_THRESHOLD = 0.1

interface Feature {
  position: THREE.Vector2  // 2D image coordinates
  position3D: THREE.Vector3  // 3D world coordinates
  descriptor: number[]  // Feature descriptor (simplified)
  strength: number  // Corner response strength
}

interface FeatureMatch {
  feature1: Feature
  feature2: Feature
  distance: number
  isInlier: boolean
}

function detectFeatures(
  camera: {
    position: THREE.Vector3
    direction: THREE.Vector3
  },
  scenePoints: THREE.Vector3[],
  imageWidth: number,
  imageHeight: number
): Feature[] {
  const features: Feature[] = []

  for (const point of scenePoints) {
    // Project to 2D
    const projected = projectPoint(point, camera, imageWidth, imageHeight)
    if (!projected) continue

    // Compute descriptor (simplified - use surrounding points)
    const descriptor = computeDescriptor(point, scenePoints)

    // Compute feature strength (simplified corner response)
    const strength = computeFeatureStrength(point, scenePoints)

    if (strength > FEATURE_DETECTION_THRESHOLD) {
      features.push({
        position: projected,
        position3D: point.clone(),
        descriptor,
        strength
      })
    }
  }

  return features.sort((a, b) => b.strength - a.strength).slice(0, 100)  // Top 100 features
}

function matchFeatures(
  features1: Feature[],
  features2: Feature[]
): FeatureMatch[] {
  const matches: FeatureMatch[] = []

  for (const f1 of features1) {
    let bestMatch: Feature | null = null
    let bestDistance = Infinity

    for (const f2 of features2) {
      const distance = descriptorDistance(f1.descriptor, f2.descriptor)
      if (distance < bestDistance) {
        bestDistance = distance
        bestMatch = f2
      }
    }

    if (bestMatch && bestDistance < FEATURE_MATCH_THRESHOLD) {
      matches.push({
        feature1: f1,
        feature2: bestMatch,
        distance: bestDistance,
        isInlier: false  // Will be determined by RANSAC
      })
    }
  }

  // RANSAC to filter outliers
  return ransacFilterMatches(matches)
}

function projectPoint(
  point: THREE.Vector3,
  camera: { position: THREE.Vector3, direction: THREE.Vector3 },
  width: number,
  height: number
): THREE.Vector2 | null {
  const toPoint = point.clone().sub(camera.position)
  const distance = toPoint.dot(camera.direction)

  if (distance <= 0) return null  // Behind camera

  const x = (toPoint.x / distance) * width / 2 + width / 2
  const y = (toPoint.y / distance) * height / 2 + height / 2

  return new THREE.Vector2(x, y)
}

function computeDescriptor(point: THREE.Vector3, neighbors: THREE.Vector3[]): number[] {
  // Simplified descriptor - average distances to nearby points
  const descriptor: number[] = []
  const nearby = neighbors
    .filter(p => p.distanceTo(point) < 1.0 && p !== point)
    .slice(0, 8)

  for (const n of nearby) {
    descriptor.push(point.distanceTo(n))
  }

  return descriptor
}

function computeFeatureStrength(point: THREE.Vector3, neighbors: THREE.Vector3[]): number {
  // Simplified corner response - variance in nearby point distances
  const nearby = neighbors.filter(p => p.distanceTo(point) < 0.5 && p !== point)
  if (nearby.length < 3) return 0

  const distances = nearby.map(p => point.distanceTo(p))
  const mean = distances.reduce((a, b) => a + b, 0) / distances.length
  const variance = distances.reduce((sum, d) => sum + (d - mean) ** 2, 0) / distances.length

  return Math.sqrt(variance)
}

function descriptorDistance(desc1: number[], desc2: number[]): number {
  let sum = 0
  const len = Math.min(desc1.length, desc2.length)
  for (let i = 0; i < len; i++) {
    sum += (desc1[i] - desc2[i]) ** 2
  }
  return Math.sqrt(sum)
}

function ransacFilterMatches(matches: FeatureMatch[]): FeatureMatch[] {
  let bestInliers: FeatureMatch[] = []

  for (let i = 0; i < RANSAC_ITERATIONS; i++) {
    // Sample random match
    const sample = matches[Math.floor(Math.random() * matches.length)]

    // Count inliers
    const inliers = matches.filter(m => {
      const dist = m.feature1.position3D.distanceTo(m.feature2.position3D)
      return dist < RANSAC_THRESHOLD
    })

    if (inliers.length > bestInliers.length) {
      bestInliers = inliers
    }
  }

  // Mark inliers
  bestInliers.forEach(m => m.isInlier = true)
  return bestInliers
}

export { detectFeatures, matchFeatures, type Feature, type FeatureMatch }
""",
            "optical_flow": """
// Optical Flow - Track Motion Between Frames
// Estimates velocity field of moving objects/camera
// Time Complexity: O(n) where n = number of tracked points

import * as THREE from 'three'

const FLOW_WINDOW_SIZE = 5  // Pixels
const MAX_FLOW_MAGNITUDE = 50  // Pixels per frame
const FLOW_SMOOTHING = 0.4

interface FlowVector {
  position: THREE.Vector2
  velocity: THREE.Vector2  // Pixels per frame
  magnitude: number
}

interface OpticalFlowState {
  previousPoints: THREE.Vector2[]
  flowVectors: FlowVector[]
}

function computeOpticalFlow(
  currentPoints: THREE.Vector2[],
  state: OpticalFlowState
): FlowVector[] {
  const flows: FlowVector[] = []

  if (state.previousPoints.length === 0) {
    state.previousPoints = currentPoints
    return flows
  }

  // Match points and compute flow
  for (let i = 0; i < Math.min(currentPoints.length, state.previousPoints.length); i++) {
    const current = currentPoints[i]
    const previous = state.previousPoints[i]

    // Compute flow vector
    const velocity = current.clone().sub(previous)
    const magnitude = velocity.length()

    // Filter large flows (likely mismatches)
    if (magnitude > MAX_FLOW_MAGNITUDE) continue

    // Smooth with previous flow if available
    if (state.flowVectors[i]) {
      velocity.lerp(state.flowVectors[i].velocity, FLOW_SMOOTHING)
    }

    flows.push({
      position: current.clone(),
      velocity,
      magnitude: velocity.length()
    })
  }

  state.previousPoints = currentPoints
  state.flowVectors = flows

  return flows
}

function estimateEgoMotion(flows: FlowVector[]): {
  translation: THREE.Vector2
  rotation: number
} {
  // Estimate camera motion from flow field
  const avgFlow = new THREE.Vector2()
  let rotationSum = 0

  for (const flow of flows) {
    avgFlow.add(flow.velocity)

    // Estimate rotation from flow curl
    const angle = Math.atan2(flow.velocity.y, flow.velocity.x)
    rotationSum += angle
  }

  avgFlow.divideScalar(flows.length || 1)
  const rotation = rotationSum / (flows.length || 1)

  return {
    translation: avgFlow,
    rotation
  }
}

export { computeOpticalFlow, estimateEgoMotion, type FlowVector, type OpticalFlowState }
""",
            "semantic_segmentation": """
// Semantic Segmentation - Classify Each Pixel/Region
// Labels different parts of the scene (floor, walls, objects, etc.)
// Time Complexity: O(n) where n = number of scene elements

import * as THREE from 'three'

const SEGMENTATION_CLASSES = [
  'floor',
  'wall',
  'obstacle',
  'goal',
  'robot',
  'unknown'
]

interface SegmentedRegion {
  class: string
  confidence: number
  points: THREE.Vector3[]
  centroid: THREE.Vector3
  color: string
}

function segmentScene(
  objects: Array<{
    position: THREE.Vector3
    label: string
    radius: number
  }>,
  floorLevel: number = 0
): SegmentedRegion[] {
  const segments: SegmentedRegion[] = []

  // Segment floor
  const floorPoints: THREE.Vector3[] = []
  for (let x = -10; x <= 10; x += 0.5) {
    for (let z = -10; z <= 10; z += 0.5) {
      floorPoints.push(new THREE.Vector3(x, floorLevel, z))
    }
  }

  segments.push({
    class: 'floor',
    confidence: 1.0,
    points: floorPoints,
    centroid: new THREE.Vector3(0, floorLevel, 0),
    color: '#808080'
  })

  // Segment objects
  for (const obj of objects) {
    const objClass = classifyObject(obj.label)
    const points = generateObjectPoints(obj.position, obj.radius)

    segments.push({
      class: objClass,
      confidence: 0.9,
      points,
      centroid: obj.position.clone(),
      color: getClassColor(objClass)
    })
  }

  return segments
}

function classifyObject(label: string): string {
  const lowerLabel = label.toLowerCase()

  if (lowerLabel.includes('goal') || lowerLabel.includes('target')) {
    return 'goal'
  } else if (lowerLabel.includes('robot')) {
    return 'robot'
  } else if (lowerLabel.includes('wall')) {
    return 'wall'
  } else if (lowerLabel.includes('obstacle') || lowerLabel.includes('box')) {
    return 'obstacle'
  }

  return 'unknown'
}

function generateObjectPoints(center: THREE.Vector3, radius: number): THREE.Vector3[] {
  const points: THREE.Vector3[] = []
  const samples = 20

  for (let i = 0; i < samples; i++) {
    const theta = (i / samples) * Math.PI * 2
    const x = center.x + radius * Math.cos(theta)
    const z = center.z + radius * Math.sin(theta)
    points.push(new THREE.Vector3(x, center.y, z))
  }

  return points
}

function getClassColor(className: string): string {
  const colorMap: Record<string, string> = {
    'floor': '#808080',
    'wall': '#654321',
    'obstacle': '#ff0000',
    'goal': '#00ff00',
    'robot': '#0000ff',
    'unknown': '#888888'
  }

  return colorMap[className] || '#888888'
}

export { segmentScene, type SegmentedRegion, SEGMENTATION_CLASSES }
""",
        }

    @staticmethod
    def motion_control_templates() -> Dict[str, str]:
        """Motion control algorithm templates"""
        return {
            "pid_controller": """
// PID Controller for Motion Control
// Classic control algorithm for position/velocity tracking
// Time Complexity: O(1)

const KP = 2.0  // Proportional gain
const KI = 0.1  // Integral gain
const KD = 0.5  // Derivative gain
const MAX_INTEGRAL = 10.0  // Anti-windup limit

interface PIDState {
  integral: number
  previousError: number
}

function pidControl(
  current: number,
  target: number,
  state: PIDState,
  deltaTime: number
): number {
  const error = target - current

  // Proportional term
  const P = KP * error

  // Integral term with anti-windup
  state.integral += error * deltaTime
  state.integral = Math.max(-MAX_INTEGRAL, Math.min(MAX_INTEGRAL, state.integral))
  const I = KI * state.integral

  // Derivative term
  const derivative = (error - state.previousError) / deltaTime
  const D = KD * derivative

  state.previousError = error

  return P + I + D
}

export { pidControl, type PIDState }
""",
        }


def get_template(algorithm_type: str, algorithm_name: str = None) -> str:
    """
    Get a specific algorithm template

    Args:
        algorithm_type: Category (path_planning, obstacle_avoidance, etc.)
        algorithm_name: Specific algorithm (astar, dwa, fabrik, etc.)

    Returns:
        Template code string
    """
    templates = AlgorithmTemplates.get_all_templates()

    if algorithm_type not in templates:
        return ""

    category = templates[algorithm_type]

    if algorithm_name:
        return category.get(algorithm_name, "")

    # Return first template in category if no specific name given
    return next(iter(category.values()), "")


def get_algorithm_list() -> List[Dict[str, str]]:
    """
    Get list of all available algorithms with metadata

    Returns:
        List of algorithm info dicts
    """
    algorithms = []

    templates = AlgorithmTemplates.get_all_templates()

    for category, algos in templates.items():
        for name, code in algos.items():
            # Extract description from code comments
            lines = code.split('\n')
            description = ""
            for line in lines[1:10]:  # Check first 10 lines for description
                if line.strip().startswith('//'):
                    description += line.strip()[2:].strip() + " "

            algorithms.append({
                "name": name,
                "category": category,
                "description": description.strip(),
                "complexity": extract_complexity(code)
            })

    return algorithms


def extract_complexity(code: str) -> str:
    """Extract time complexity from code comments"""
    for line in code.split('\n'):
        if 'Time Complexity' in line:
            return line.split('Time Complexity:')[1].strip()
    return "Unknown"
