/**
 * Algorithm Manager for Celestial Studio - FIXED VERSION
 * 
 * Key improvements:
 * 1. Uses standardized function names from AlgorithmInterface
 * 2. Proper THREE.js context injection
 * 3. Better error handling and reporting
 * 4. Stores function metadata from backend
 */

import * as THREE from 'three'
import axios from 'axios'
import config from '../config'
import type {
  PathPlanningAlgorithm,
  ObstacleAvoidanceAlgorithm,
  InverseKinematicsAlgorithm,
  ComputerVisionAlgorithm,
  AlgorithmMetadata,
  Obstacle,
  Vector2D
} from '../types/AlgorithmInterface'

export interface Algorithm {
  id: string
  name: string
  type: 'path_planning' | 'obstacle_avoidance' | 'inverse_kinematics' | 'computer_vision' | 'motion_control'
  code: string
  parameters: AlgorithmParameter[]
  complexity: string
  description: string
  compiledFunction?: any  // Compiled algorithm functions
  functionName?: string  // NEW: Primary function name from backend
  functionSignature?: AlgorithmMetadata  // NEW: Function metadata
}

export interface AlgorithmParameter {
  name: string
  type: 'number' | 'boolean' | 'string'
  value: string | number | boolean
  min?: number
  max?: number
  step?: number
  description: string
}

export interface AlgorithmState {
  position: THREE.Vector3
  velocity: THREE.Vector3
  rotation: THREE.Euler
  customData: Record<string, unknown>
}

/**
 * AlgorithmManager - Hot-swappable algorithm system with standardized interfaces
 */
export class AlgorithmManager {
  private algorithms: Map<string, Algorithm> = new Map()
  private activeAlgorithms: Map<string, Set<string>> = new Map()
  private robotStates: Map<string, AlgorithmState> = new Map()
  private readonly API_URL = config.backendUrl

  /**
   * Generate a new algorithm from natural language description
   */
  async generateAlgorithm(
    description: string,
    robotType: 'mobile' | 'arm' | 'drone',
    algorithmType: Algorithm['type'],
    userId?: string
  ): Promise<Algorithm> {
    try {
      console.log(`🔄 Generating ${algorithmType} algorithm...`)

      const response = await axios.post(`${this.API_URL}/api/generate-algorithm`, {
        description,
        robot_type: robotType,
        algorithm_type: algorithmType
      }, {
        timeout: 120000,
        headers: { 'Content-Type': 'application/json' }
      })

      console.log('✅ Received response from backend:', response.status)
      const {
        code,
        parameters,
        complexity,
        description: desc,
        function_name,  // NEW: From backend
        function_signature  // NEW: From backend
      } = response.data

      // Compile code with proper THREE.js context
      const compiledFunction = this.compileCode(code, function_name)

      const algorithm: Algorithm = {
        id: this.generateId(),
        name: function_name || this.extractAlgorithmName(code) || `${algorithmType}_${Date.now()}`,
        type: algorithmType,
        code,
        parameters: parameters || [],
        complexity: complexity || 'Unknown',
        description: desc || description,
        compiledFunction,
        functionName: function_name,  // NEW
        functionSignature: function_signature  // NEW
      }

      this.algorithms.set(algorithm.id, algorithm)

      if (userId) {
        await this.storeAlgorithmInContext(userId, algorithm)
      }

      console.log(`✅ Generated algorithm: ${algorithm.name}`)
      console.log(`   Function name: ${function_name}`)
      console.log(`   Compiled: ${compiledFunction ? 'Yes' : 'No'}`)
      
      return algorithm

    } catch (error: unknown) {
      console.error('❌ Algorithm generation failed:', error)

      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNABORTED') {
          throw new Error('Request timed out. Algorithm generation takes 10-20 seconds.')
        } else if (error.response) {
          throw new Error(`Server error: ${error.response.data?.detail || error.response.statusText}`)
        } else if (error.request) {
          throw new Error('No response from server. Is the backend running on port 8000?')
        }
      }

      throw new Error(`Failed to generate algorithm: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  /**
   * Compile TypeScript/JavaScript code with proper THREE.js context
   * FIXED VERSION: Uses proper sandboxing and function extraction
   */
  private compileCode(code: string, expectedFunctionName?: string): any | null {
    try {
      console.log(`🔨 Compiling code (expected function: ${expectedFunctionName})...`)

      // Remove imports (not supported in Function constructor)
      const cleanCode = code.replace(/import\s+.*\s+from\s+['"].*['"]/g, '')

      // Create sandbox with THREE.js context
      const sandbox = {
        THREE,
        Vector3: THREE.Vector3,
        Vector2: THREE.Vector2,
        Euler: THREE.Euler,
        Math: Math,
        console: console  // Allow logging for debugging
      }

      // Wrap code to return an object with all functions
      const wrappedCode = `
        ${cleanCode}
        
        // Return object with all declared functions
        return {
          ${expectedFunctionName || 'main'}: typeof ${expectedFunctionName} !== 'undefined' ? ${expectedFunctionName} : null
        }
      `

      // Create function with sandbox context
      const sandboxKeys = Object.keys(sandbox)
      const sandboxValues = Object.values(sandbox)
      
      const compiledFn = new Function(...sandboxKeys, wrappedCode)
      const result = compiledFn(...sandboxValues)

      if (expectedFunctionName && !result[expectedFunctionName]) {
        console.warn(`⚠️ Expected function '${expectedFunctionName}' not found in compiled code`)
        return null
      }

      console.log(`✅ Code compiled successfully`)
      console.log(`   Available functions:`, Object.keys(result))
      
      return result

    } catch (error: unknown) {
      console.error('❌ Code compilation failed:', error instanceof Error ? error.message : String(error))
      console.log('Code preview:', code.substring(0, 200) + '...')
      return null
    }
  }

  /**
   * Apply algorithm to a robot
   */
  applyAlgorithm(robotId: string, algorithmId: string): void {
    const algorithm = this.algorithms.get(algorithmId)
    if (!algorithm) {
      throw new Error(`Algorithm ${algorithmId} not found`)
    }

    if (!this.activeAlgorithms.has(robotId)) {
      this.activeAlgorithms.set(robotId, new Set<string>())
    }

    this.activeAlgorithms.get(robotId)!.add(algorithmId)

    console.log(`🔄 Applied algorithm ${algorithm.name} to robot ${robotId}`)
  }

  /**
   * Remove algorithm from a robot
   */
  removeAlgorithm(robotId: string, algorithmId: string): void {
    const activeSet = this.activeAlgorithms.get(robotId)
    if (!activeSet) return

    activeSet.delete(algorithmId)
    if (activeSet.size === 0) {
      this.activeAlgorithms.delete(robotId)
    }
  }

  /**
   * Get all active algorithms for a robot
   */
  getActiveAlgorithms(robotId: string): Algorithm[] {
    const activeSet = this.activeAlgorithms.get(robotId)
    if (!activeSet || activeSet.size === 0) return []

    return Array.from(activeSet)
      .map(id => this.algorithms.get(id))
      .filter((algo): algo is Algorithm => algo !== undefined)
  }

  /**
   * Get active algorithms of a specific type for a robot
   */
  getAlgorithmsByType(robotId: string, type: Algorithm['type']): Algorithm[] {
    return this.getActiveAlgorithms(robotId).filter(algo => algo.type === type)
  }

  /**
   * Execute path planning algorithm using standardized interface
   */
  executePathPlanning(
    algorithmId: string,
    start: THREE.Vector3,
    goal: THREE.Vector3,
    obstacles: Obstacle[]
  ): THREE.Vector3[] {
    const algorithm = this.algorithms.get(algorithmId)
    if (!algorithm) {
      throw new Error(`Algorithm ${algorithmId} not found`)
    }

    if (algorithm.type !== 'path_planning') {
      throw new Error(`Algorithm ${algorithmId} is not a path planning algorithm`)
    }

    if (!algorithm.compiledFunction) {
      throw new Error(`Algorithm ${algorithmId} is not compiled`)
    }

    const functionName = algorithm.functionName || 'findPath'

    try {
      const func = algorithm.compiledFunction[functionName]
      if (typeof func !== 'function') {
        throw new Error(`Function '${functionName}' not found in algorithm`)
      }

      const result = func(start, goal, obstacles)

      if (!Array.isArray(result)) {
        throw new Error(`Expected array of waypoints, got ${typeof result}`)
      }

      return result

    } catch (error: unknown) {
      console.error(`❌ Path planning execution error:`, error)
      throw new Error(`Failed to execute path planning: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  /**
   * Execute obstacle avoidance algorithm using standardized interface
   */
  executeObstacleAvoidance(
    algorithmId: string,
    currentPos: Vector2D,
    currentVel: Vector2D,
    obstacles: Obstacle[],
    goal: Vector2D,
    maxSpeed: number
  ): Vector2D {
    const algorithm = this.algorithms.get(algorithmId)
    if (!algorithm) {
      throw new Error(`Algorithm ${algorithmId} not found`)
    }

    if (algorithm.type !== 'obstacle_avoidance') {
      throw new Error(`Algorithm ${algorithmId} is not an obstacle avoidance algorithm`)
    }

    if (!algorithm.compiledFunction) {
      throw new Error(`Algorithm ${algorithmId} is not compiled`)
    }

    const functionName = algorithm.functionName || 'computeSafeVelocity'

    try {
      const func = algorithm.compiledFunction[functionName]
      if (typeof func !== 'function') {
        throw new Error(`Function '${functionName}' not found in algorithm`)
      }

      const result = func(currentPos, currentVel, obstacles, goal, maxSpeed)

      if (typeof result !== 'object' || typeof result.x !== 'number' || typeof result.z !== 'number') {
        throw new Error(`Expected {x, z} velocity vector, got ${JSON.stringify(result)}`)
      }

      return result

    } catch (error: unknown) {
      console.error(`❌ Obstacle avoidance execution error:`, error)
      throw new Error(`Failed to execute obstacle avoidance: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  /**
   * Get algorithm by ID
   */
  getAlgorithm(algorithmId: string): Algorithm | undefined {
    return this.algorithms.get(algorithmId)
  }

  /**
   * Get all loaded algorithms
   */
  getAllAlgorithms(): Algorithm[] {
    return Array.from(this.algorithms.values())
  }

  /**
   * Save robot state
   */
  saveRobotState(robotId: string, state: AlgorithmState): void {
    this.robotStates.set(robotId, state)
  }

  /**
   * Restore robot state
   */
  restoreRobotState(robotId: string): AlgorithmState | null {
    return this.robotStates.get(robotId) || null
  }

  // ========== Private Helper Methods ==========

  private extractAlgorithmName(code: string): string | null {
    const functionMatch = code.match(/function\s+(\w+)/)
    if (functionMatch) return functionMatch[1]

    const constMatch = code.match(/const\s+(\w+)\s*=/)
    if (constMatch) return constMatch[1]

    return null
  }

  private generateId(): string {
    return `algo_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  private async storeAlgorithmInContext(userId: string, algorithm: Algorithm): Promise<void> {
    try {
      const algorithmData = {
        id: algorithm.id,
        name: algorithm.name,
        type: algorithm.type,
        code: algorithm.code,
        parameters: algorithm.parameters,
        complexity: algorithm.complexity,
        description: algorithm.description,
        functionName: algorithm.functionName
      }

      await axios.post(`${this.API_URL}/api/chat/store-algorithm`, {
        userId,
        algorithm: algorithmData
      })

      console.log(`📝 Stored algorithm in conversation context`)
    } catch (error) {
      console.warn('Failed to store algorithm in context:', error)
    }
  }
}

// Singleton instance
let managerInstance: AlgorithmManager | null = null

export function getAlgorithmManager(): AlgorithmManager {
  if (!managerInstance) {
    managerInstance = new AlgorithmManager()
  }
  return managerInstance
}
