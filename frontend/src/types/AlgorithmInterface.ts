/**
 * Standardized Algorithm Interfaces for Celestial Studio
 * 
 * All generated algorithms MUST implement one of these interfaces.
 * This ensures consistent function names and signatures.
 */

import * as THREE from 'three'

// Common types used across algorithms
export interface Obstacle {
  position: THREE.Vector3
  radius: number
}

export interface Vector2D {
  x: number
  z: number
}

// ============================================
// Path Planning Algorithm Interface
// ============================================
export interface PathPlanningAlgorithm {
  /**
   * Find a path from start to goal avoiding obstacles
   * 
   * @param start - Starting position
   * @param goal - Goal position
   * @param obstacles - List of obstacles to avoid
   * @returns Array of waypoints from start to goal
   */
  findPath(
    start: THREE.Vector3,
    goal: THREE.Vector3,
    obstacles: Obstacle[]
  ): THREE.Vector3[]
}

// ============================================
// Obstacle Avoidance Algorithm Interface
// ============================================
export interface ObstacleAvoidanceAlgorithm {
  /**
   * Compute a safe velocity that avoids obstacles while moving toward goal
   * 
   * @param currentPos - Current position (2D: x, z)
   * @param currentVel - Current velocity (2D: x, z)
   * @param obstacles - List of obstacles to avoid
   * @param goal - Goal position (2D: x, z)
   * @param maxSpeed - Maximum allowed speed
   * @returns Safe velocity vector (2D: x, z)
   */
  computeSafeVelocity(
    currentPos: Vector2D,
    currentVel: Vector2D,
    obstacles: Obstacle[],
    goal: Vector2D,
    maxSpeed: number
  ): Vector2D
}

// ============================================
// Inverse Kinematics Algorithm Interface
// ============================================
export interface InverseKinematicsAlgorithm {
  /**
   * Solve inverse kinematics to reach target position
   * 
   * @param targetPos - Target end-effector position
   * @param currentAngles - Current joint angles (radians)
   * @param linkLengths - Length of each link in the chain
   * @returns Joint angles to reach target
   */
  solveIK(
    targetPos: THREE.Vector3,
    currentAngles: number[],
    linkLengths: number[]
  ): number[]
}

// ============================================
// Computer Vision Algorithm Interface
// ============================================
export interface Detection {
  label: string
  confidence: number
  bbox: { x: number; y: number; width: number; height: number }
  position3D: THREE.Vector3
  distance: number
}

export interface ComputerVisionAlgorithm {
  /**
   * Process visual input and detect objects
   * 
   * @param cameraState - Camera position and orientation
   * @param sceneObjects - Objects in the scene
   * @param params - Algorithm parameters (threshold, range, etc.)
   * @returns List of detected objects
   */
  processVision(
    cameraState: {
      position: THREE.Vector3
      direction: THREE.Vector3
      up: THREE.Vector3
    },
    sceneObjects: Array<{
      position: THREE.Vector3
      label: string
      radius: number
    }>,
    params: { threshold: number; range: number }
  ): Detection[]
}

// ============================================
// Algorithm Metadata (returned by backend)
// ============================================
export interface AlgorithmMetadata {
  functionName: string  // Primary function name
  algorithmType: 'path_planning' | 'obstacle_avoidance' | 'inverse_kinematics' | 'computer_vision'
  parameterTypes: string[]  // Parameter type signatures
  returnType: string  // Return type signature
}
