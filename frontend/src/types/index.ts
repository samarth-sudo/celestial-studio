/**
 * Centralized Type Definitions for Celestial Studio
 *
 * This file contains all TypeScript interfaces and types used throughout
 * the application to ensure type safety and consistency.
 */

// ============================================================================
// Scene Configuration Types
// ============================================================================

export interface SceneConfig {
  environment: Environment
  robot: RobotConfig
  objects: SceneObject[]
  lighting?: LightingConfig
  camera?: CameraConfig
}

export interface Environment {
  floor: FloorConfig
  walls?: boolean
  wallHeight?: number
  wallColor?: string
  skybox?: string
}

export interface FloorConfig {
  size: [number, number]
  color?: string
  texture?: string
  gridSize?: number
  gridColor?: string
}

export interface LightingConfig {
  ambient?: {
    color: string
    intensity: number
  }
  directional?: {
    color: string
    intensity: number
    position: [number, number, number]
    castShadow: boolean
  }
  point?: Array<{
    color: string
    intensity: number
    position: [number, number, number]
    distance: number
  }>
}

export interface CameraConfig {
  position: [number, number, number]
  target: [number, number, number]
  fov?: number
}

// ============================================================================
// Robot Configuration Types
// ============================================================================

export interface RobotConfig {
  type: 'mobile_robot' | 'robotic_arm' | 'humanoid' | 'quadruped' | 'custom'
  name?: string
  links?: LinkConfig[]
  joints?: JointConfig[]
  kinematic_tree?: KinematicTree
  base_link?: string
  urdf?: string
}

export interface LinkConfig {
  name: string
  geometry: GeometryConfig
  material: MaterialConfig
  position: [number, number, number]
  rotation: [number, number, number]
  castShadow: boolean
  receiveShadow: boolean
}

export interface GeometryConfig {
  type: 'box' | 'cylinder' | 'sphere' | 'mesh'
  args: number[]
  filename?: string
  scale?: [number, number, number]
}

export interface MaterialConfig {
  color: string
  metalness: number
  roughness: number
  transparent?: boolean
  opacity?: number
}

export interface JointConfig {
  name: string
  type: 'fixed' | 'revolute' | 'prismatic' | 'continuous' | 'floating' | 'planar'
  parent_link: string
  child_link: string
  position: [number, number, number]
  rotation: [number, number, number]
  axis: [number, number, number]
  limits: {
    lower: number
    upper: number
    effort: number
    velocity: number
  }
  current_angle?: number
}

export interface KinematicTree {
  [linkName: string]: {
    children: {
      [childLink: string]: boolean
    }
  }
}

// ============================================================================
// Scene Objects
// ============================================================================

export interface SceneObject {
  id: string
  type: 'box' | 'sphere' | 'cylinder' | 'mesh' | 'obstacle'
  position: [number, number, number]
  rotation?: [number, number, number]
  scale?: [number, number, number]
  size?: [number, number, number]
  radius?: number
  height?: number
  color?: string
  metalness?: number
  roughness?: number
  castShadow?: boolean
  receiveShadow?: boolean
}

// ============================================================================
// Algorithm Types
// ============================================================================

export interface Algorithm {
  id: string
  name: string
  type: AlgorithmType
  code: string
  parameters: AlgorithmParameters
  description?: string
  complexity?: string
  author?: string
  created?: string
  modified?: string
}

export type AlgorithmType =
  | 'path_planning'
  | 'control'
  | 'perception'
  | 'manipulation'
  | 'navigation'
  | 'custom'

export interface AlgorithmParameters {
  [key: string]: AlgorithmParameter
}

export interface AlgorithmParameter {
  type: 'number' | 'string' | 'boolean' | 'select'
  value: number | string | boolean
  min?: number
  max?: number
  step?: number
  options?: string[]
  label: string
  description?: string
}

export interface AlgorithmExecutionResult {
  success: boolean
  output?: unknown
  error?: string
  executionTime?: number
  metrics?: Record<string, number>
}

// ============================================================================
// Simulation Types
// ============================================================================

export interface SimulationState {
  running: boolean
  paused: boolean
  time: number
  fps: number
  step: number
}

export interface SimulationConfig {
  duration: number
  fps: number
  recordVideo: boolean
  headless: boolean
}

export interface SimulationResult {
  success: boolean
  metrics?: SimulationMetrics
  videoPath?: string
  error?: string
}

export interface SimulationMetrics {
  frames: number
  steps: number
  fps_avg: number
  duration: number
  steps_per_sec?: number
  robot_type?: string
}

// ============================================================================
// Training Types
// ============================================================================

export interface TrainingConfig {
  task_name: string
  num_envs: number
  max_iterations: number
  algorithm: 'PPO' | 'SAC' | 'RSL'
  headless: boolean
}

export interface TrainingResult {
  success: boolean
  model_path?: string
  task?: string
  environment?: string
  algorithm?: string
  training_time?: number
  timesteps?: number
  error?: string
}

export interface TrainingProgress {
  current_iteration: number
  total_iterations: number
  elapsed_time: number
  metrics: Record<string, number>
  status: 'pending' | 'running' | 'completed' | 'failed'
}

// ============================================================================
// API Response Types
// ============================================================================

export interface APIResponse<T = unknown> {
  status: 'success' | 'error'
  data?: T
  error?: string
  message?: string
}

export interface HealthCheckResponse {
  status: 'healthy' | 'unhealthy'
  deployment: string
  gpu_available: boolean
}

export interface TaskInfo {
  id: string
  name: string
  description: string
  type: string
  robot?: string
  difficulty?: string
}

// ============================================================================
// Export/Download Types
// ============================================================================

export interface ExportConfig {
  export_format: 'react' | 'ros' | 'python' | 'algorithms'
  algorithms: Algorithm[]
  scene_config: SceneConfig
  robots: RobotConfig[]
  project_name: string
}

export interface ExportResult {
  status: 'success' | 'error'
  filename?: string
  download_url?: string
  error?: string
}

// ============================================================================
// Chat/Conversation Types
// ============================================================================

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  metadata?: Record<string, unknown>
}

export interface ConversationContext {
  messages: ChatMessage[]
  scene_config?: SceneConfig
  current_algorithm?: Algorithm
}

// ============================================================================
// URDF Types
// ============================================================================

export interface URDFParseResult {
  status: 'success' | 'error'
  filename?: string
  urdf_data?: URDFData
  scene_config?: SceneConfig
  error?: string
}

export interface URDFData {
  robot: RobotConfig
  links: LinkConfig[]
  joints: JointConfig[]
  materials: Record<string, MaterialConfig>
}

// ============================================================================
// Multi-Robot Types
// ============================================================================

export interface MultiRobotScene {
  robots: RobotInstance[]
  environment: Environment
  bounds?: SceneBounds
}

export interface RobotInstance {
  id: string
  name: string
  scene_config: SceneConfig
  position: [number, number, number]
  orientation: [number, number, number]
  metadata?: Record<string, unknown>
}

export interface SceneBounds {
  min: [number, number, number]
  max: [number, number, number]
  center: [number, number, number]
  size: [number, number, number]
}

// ============================================================================
// Utility Types
// ============================================================================

export type Vector3 = [number, number, number]
export type Quaternion = [number, number, number, number]
export type Color = string

export interface Transform {
  position: Vector3
  rotation: Vector3
  scale?: Vector3
}

// ============================================================================
// Component Props Types
// ============================================================================

export interface SimulatorProps {
  sceneConfig: SceneConfig | null
  algorithmManager?: AlgorithmManager
  onSceneUpdate?: (config: SceneConfig) => void
}

export interface RobotProps {
  sceneConfig: SceneConfig
  position?: Vector3
  scale?: number
}

export interface AlgorithmControlsProps {
  algorithmManager: AlgorithmManager
  onExecute?: (result: AlgorithmExecutionResult) => void
}

// ============================================================================
// Algorithm Manager Interface (for services)
// ============================================================================

export interface AlgorithmManager {
  algorithms: Map<string, Algorithm>
  currentAlgorithm: Algorithm | null

  loadAlgorithm: (id: string) => Promise<Algorithm | null>
  saveAlgorithm: (algorithm: Algorithm) => Promise<boolean>
  deleteAlgorithm: (id: string) => Promise<boolean>
  listAlgorithms: () => Promise<Algorithm[]>
  executeAlgorithm: (id: string, params?: Record<string, unknown>) => Promise<AlgorithmExecutionResult>
  generateAlgorithm: (description: string, type: AlgorithmType) => Promise<Algorithm | null>
  modifyAlgorithm: (id: string, modification: string) => Promise<Algorithm | null>
}
