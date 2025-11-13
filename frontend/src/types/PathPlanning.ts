import * as THREE from 'three'

// Obstacle types
export type ObstacleType = 'box' | 'cylinder' | 'sphere' | 'plant' | 'furniture' | 'wall' | 'custom-area'

export interface Obstacle {
  id: string
  type: ObstacleType
  position: THREE.Vector3
  size: THREE.Vector3 // width, height, depth
  radius?: number // for cylinders and spheres
  rotation?: THREE.Euler
  color?: string
  points?: THREE.Vector3[] // for custom areas (polygon)
}

// Scene markers
export interface PathMarker {
  id: string
  type: 'origin' | 'destination'
  position: THREE.Vector3
  color: string
}

// Path visualization
export interface PathSegment {
  start: THREE.Vector3
  end: THREE.Vector3
}

export interface ComputedPath {
  waypoints: THREE.Vector3[]
  segments: PathSegment[]
  length: number // total path length in meters
  estimatedTime: number // seconds
  isValid: boolean
  algorithmUsed: string
}

// Interaction modes
export type InteractionMode = 'none' | 'place-origin' | 'place-destination' | 'place-obstacle' | 'move-obstacle' | 'delete-obstacle'

// Scene state
export interface PathPlanningState {
  origin: THREE.Vector3 | null
  destination: THREE.Vector3 | null
  obstacles: Obstacle[]
  computedPath: ComputedPath | null
  interactionMode: InteractionMode
  selectedObstacle: string | null
  gridSize: number // cell size for A* grid
}

// Obstacle presets
export interface ObstaclePreset {
  type: ObstacleType
  name: string
  defaultSize: THREE.Vector3
  defaultRadius?: number
  icon: string
  description: string
}

// Scenario presets
export interface ScenarioPreset {
  id: string
  name: string
  description: string
  thumbnail?: string
  obstacles: Omit<Obstacle, 'id'>[]
  origin?: THREE.Vector3
  destination?: THREE.Vector3
}

// Path planning algorithm comparison
export interface AlgorithmComparison {
  algorithmName: string
  computationTime: number // milliseconds
  pathLength: number // meters
  waypointCount: number
  smoothness: number // 0-1 score
}
