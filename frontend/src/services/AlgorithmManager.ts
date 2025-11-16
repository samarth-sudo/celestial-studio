/**
 * Algorithm Manager for Celestial Studio
 *
 * Manages hot-swappable algorithms for robots in the simulation.
 * Enables real-time code injection and live parameter tuning without restart.
 */

import * as THREE from 'three'
import axios from 'axios'
import config from '../config'

export interface Algorithm {
  id: string
  name: string
  type: 'path_planning' | 'obstacle_avoidance' | 'inverse_kinematics' | 'computer_vision' | 'motion_control'
  code: string
  parameters: AlgorithmParameter[]
  complexity: string
  description: string
  compiledFunction?: Function
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
 * AlgorithmManager - Hot-swappable algorithm system
 */
export class AlgorithmManager {
  private algorithms: Map<string, Algorithm> = new Map()
  private activeAlgorithms: Map<string, Set<string>> = new Map() // robotId -> Set of algorithmIds
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
      console.log(`Description: "${description}"`)
      console.log(`Robot type: ${robotType}`)

      const response = await axios.post(`${this.API_URL}/api/generate-algorithm`, {
        description,
        robot_type: robotType,
        algorithm_type: algorithmType
      }, {
        timeout: 60000, // 60 second timeout (generation can take 10-20 seconds)
        headers: {
          'Content-Type': 'application/json'
        }
      })

      console.log('✅ Received response from backend:', response.status)
      const { code, parameters, complexity, description: desc } = response.data

      // Try to compile code (optional for Computer Vision algorithms)
      const compiledFunction = this.compileCode(code)

      // Create algorithm object
      const algorithm: Algorithm = {
        id: this.generateId(),
        name: this.extractAlgorithmName(code) || `${algorithmType}_${Date.now()}`,
        type: algorithmType,
        code,
        parameters: parameters || [],
        complexity: complexity || 'Unknown',
        description: desc || description,
        compiledFunction: compiledFunction || undefined
      }

      // Store algorithm
      this.algorithms.set(algorithm.id, algorithm)

      // Store in conversation context for export
      if (userId) {
        await this.storeAlgorithmInContext(userId, algorithm)
      }

      console.log(`✅ Generated algorithm: ${algorithm.name}`)
      console.log(`Code length: ${code.length} chars`)
      console.log(`Parameters: ${parameters?.length || 0}`)
      console.log(`Compiled: ${compiledFunction ? 'Yes' : 'No (display only)'}`)
      return algorithm

    } catch (error: unknown) {
      console.error('❌ Algorithm generation failed:', error)

      if (axios.isAxiosError(error)) {
        console.error('Error details:', {
          message: error.message,
          response: error.response?.data,
          status: error.response?.status,
          config: {
            url: error.config?.url,
            method: error.config?.method,
            data: error.config?.data
          }
        })

        if (error.code === 'ECONNABORTED') {
          throw new Error('Request timed out. Algorithm generation takes 10-20 seconds, please wait...')
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
   * Modify an existing algorithm
   */
  async modifyAlgorithm(
    algorithmId: string,
    modificationRequest: string
  ): Promise<Algorithm> {
    const currentAlgorithm = this.algorithms.get(algorithmId)
    if (!currentAlgorithm) {
      throw new Error(`Algorithm ${algorithmId} not found`)
    }

    try {
      const response = await axios.post(`${this.API_URL}/api/generate-algorithm`, {
        description: modificationRequest,
        robot_type: 'mobile', // Will be improved to track per-algorithm
        algorithm_type: currentAlgorithm.type,
        current_code: currentAlgorithm.code,
        modification_request: modificationRequest
      })

      const { code, parameters, complexity } = response.data

      // Update algorithm
      const modifiedAlgorithm: Algorithm = {
        ...currentAlgorithm,
        code,
        parameters: parameters || currentAlgorithm.parameters,
        complexity: complexity || currentAlgorithm.complexity,
        description: `${currentAlgorithm.description} (Modified: ${modificationRequest})`,
        compiledFunction: this.compileCode(code)
      }

      this.algorithms.set(algorithmId, modifiedAlgorithm)

      console.log(`✅ Modified algorithm: ${modifiedAlgorithm.name}`)
      return modifiedAlgorithm

    } catch (error) {
      console.error('❌ Algorithm modification failed:', error)
      throw new Error(`Failed to modify algorithm: ${error}`)
    }
  }

  /**
   * Apply algorithm to a robot (adds to active algorithms)
   */
  applyAlgorithm(robotId: string, algorithmId: string): void {
    const algorithm = this.algorithms.get(algorithmId)
    if (!algorithm) {
      throw new Error(`Algorithm ${algorithmId} not found`)
    }

    // Get or create active algorithms set for this robot
    if (!this.activeAlgorithms.has(robotId)) {
      this.activeAlgorithms.set(robotId, new Set<string>())
    }

    // Add algorithm to active set
    this.activeAlgorithms.get(robotId)!.add(algorithmId)

    console.log(`🔄 Applied algorithm ${algorithm.name} to robot ${robotId}`)
    console.log(`   Total active algorithms: ${this.activeAlgorithms.get(robotId)!.size}`)
  }

  /**
   * Remove algorithm from a robot
   */
  removeAlgorithm(robotId: string, algorithmId: string): void {
    const activeSet = this.activeAlgorithms.get(robotId)
    if (!activeSet) {
      console.warn(`No active algorithms for robot ${robotId}`)
      return
    }

    const removed = activeSet.delete(algorithmId)
    if (removed) {
      const algorithm = this.algorithms.get(algorithmId)
      console.log(`🗑️ Removed algorithm ${algorithm?.name || algorithmId} from robot ${robotId}`)
      console.log(`   Remaining active algorithms: ${activeSet.size}`)
    }

    // Clean up empty sets
    if (activeSet.size === 0) {
      this.activeAlgorithms.delete(robotId)
    }
  }

  /**
   * Get active algorithm for a robot (returns first one for backwards compatibility)
   */
  getActiveAlgorithm(robotId: string): Algorithm | null {
    const activeSet = this.activeAlgorithms.get(robotId)
    if (!activeSet || activeSet.size === 0) return null

    const firstId = Array.from(activeSet)[0]
    return this.algorithms.get(firstId) || null
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
   * Check if robot has an active algorithm of a specific type
   */
  hasAlgorithmType(robotId: string, type: Algorithm['type']): boolean {
    return this.getAlgorithmsByType(robotId, type).length > 0
  }

  /**
   * Update algorithm parameter value (live tuning)
   */
  updateParameter(algorithmId: string, paramName: string, value: string | number | boolean): void {
    const algorithm = this.algorithms.get(algorithmId)
    if (!algorithm) {
      throw new Error(`Algorithm ${algorithmId} not found`)
    }

    const param = algorithm.parameters.find(p => p.name === paramName)
    if (!param) {
      throw new Error(`Parameter ${paramName} not found in algorithm`)
    }

    // Update parameter value
    param.value = value

    // Re-compile code with new parameter value
    const modifiedCode = this.injectParameterValues(algorithm.code, algorithm.parameters)
    algorithm.compiledFunction = this.compileCode(modifiedCode)

    console.log(`🎛️ Updated ${paramName} = ${value} in ${algorithm.name}`)
  }

  /**
   * Execute algorithm function
   */
  executeAlgorithm(
    algorithmId: string,
    functionName: string,
    ...args: unknown[]
  ): unknown {
    const algorithm = this.algorithms.get(algorithmId)
    if (!algorithm || !algorithm.compiledFunction) {
      throw new Error(`Algorithm ${algorithmId} not compiled`)
    }

    try {
      // Execute in sandboxed context
      return algorithm.compiledFunction(functionName, ...args)
    } catch (error) {
      console.error(`❌ Algorithm execution error:`, error)
      throw error
    }
  }

  /**
   * Execute a path planning algorithm
   */
  executePathPlanning(
    algorithmId: string,
    origin: THREE.Vector3,
    destination: THREE.Vector3,
    obstacles: Array<{ position: THREE.Vector3; radius: number }>
  ): THREE.Vector3[] {
    const algorithm = this.algorithms.get(algorithmId)
    if (!algorithm) {
      throw new Error(`Algorithm ${algorithmId} not found`)
    }

    if (algorithm.type !== 'path_planning') {
      throw new Error(`Algorithm ${algorithmId} is not a path planning algorithm`)
    }

    if (!algorithm.compiledFunction) {
      throw new Error(`Algorithm ${algorithmId} is not compiled (display only)`)
    }

    try {
      // Try common function names for path planning
      const functionNames = ['findPath', 'planPath', 'computePath', 'calculatePath']

      for (const funcName of functionNames) {
        try {
          const result = this.executeAlgorithm(algorithmId, funcName, origin, destination, obstacles)

          // Validate result is an array of Vector3
          if (Array.isArray(result) && result.length > 0) {
            return result
          }
        } catch {
          // Try next function name
          continue
        }
      }

      throw new Error('No valid path planning function found in algorithm')
    } catch (error: unknown) {
      console.error(`❌ Path planning execution error:`, error)
      throw new Error(`Failed to execute path planning: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  /**
   * Get algorithm by ID
   */
  getAlgorithm(algorithmId: string): Algorithm | undefined {
    return this.algorithms.get(algorithmId)
  }

  /**
   * Save robot state (for state preservation during hot-swap)
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

  /**
   * List all available algorithms
   */
  async listAvailableAlgorithms(): Promise<any[]> {
    try {
      const response = await axios.get(`${this.API_URL}/api/algorithms`)
      return response.data.algorithms || []
    } catch (error) {
      console.error('❌ Failed to fetch algorithms:', error)
      return []
    }
  }

  /**
   * Get all loaded algorithms
   */
  getAllAlgorithms(): Algorithm[] {
    return Array.from(this.algorithms.values())
  }

  // ========== Private Helper Methods ==========

  /**
   * Compile TypeScript/JavaScript code to executable function
   */
  private compileCode(code: string): Function | null {
    try {
      // Remove imports (not supported in Function constructor)
      const cleanCode = code.replace(/import\s+.*\s+from\s+['"].*['"]/g, '')

      // Remove TypeScript type annotations that might cause issues
      const jsCode = cleanCode
        .replace(/:\s*\w+(\[\])?(\s*\|\s*\w+)*\s*(?=[,;=)\]])/g, '') // Remove type annotations
        .replace(/interface\s+\w+\s*\{[^}]*\}/g, '') // Remove interfaces
        .replace(/type\s+\w+\s*=\s*[^;]+;/g, '') // Remove type aliases

      // Wrap code in a function that collects all declared functions
      const wrappedCode = `
        ${jsCode}

        // Collect all functions into a safe registry
        const functionRegistry = {};

        // Use this pattern to safely extract function names without eval
        const functionPattern = /function\\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\\s*\\(/g;
        const codeStr = ${JSON.stringify(jsCode)};
        let match;

        while ((match = functionPattern.exec(codeStr)) !== null) {
          const fnName = match[1];
          try {
            // Use indirect evaluation through this context
            if (typeof this[fnName] === 'function') {
              functionRegistry[fnName] = this[fnName];
            }
          } catch (e) {
            // Function might not be accessible, skip it
          }
        }

        // Return executor function with safe function registry
        return function(functionName, ...args) {
          if (functionRegistry[functionName] && typeof functionRegistry[functionName] === 'function') {
            return functionRegistry[functionName](...args);
          }
          throw new Error('Function ' + functionName + ' not found in registry');
        }
      `

      // Create function in isolated scope with THREE.js available
      const compiledFn = new Function('THREE', wrappedCode)

      // Execute with THREE.js injected
      return compiledFn.call(global || window || {}, THREE)

    } catch (error: unknown) {
      console.warn('⚠️ Code compilation skipped (will work for display only):', error instanceof Error ? error.message : String(error))
      console.log('Code preview:', code.substring(0, 200) + '...')
      // Return null instead of throwing - algorithm can still be stored and displayed
      return null
    }
  }

  /**
   * Inject parameter values into code
   */
  private injectParameterValues(code: string, parameters: AlgorithmParameter[]): string {
    let modifiedCode = code

    parameters.forEach(param => {
      // Replace const declarations with new values
      const pattern = new RegExp(
        `const\\s+${param.name}\\s*[:=]\\s*[^;\\n]+`,
        'g'
      )
      modifiedCode = modifiedCode.replace(
        pattern,
        `const ${param.name} = ${JSON.stringify(param.value)}`
      )
    })

    return modifiedCode
  }

  /**
   * Extract function name from code
   */
  private extractAlgorithmName(code: string): string | null {
    // Try to find function name
    const functionMatch = code.match(/function\s+(\w+)/)
    if (functionMatch) return functionMatch[1]

    // Try to find const function name
    const constMatch = code.match(/const\s+(\w+)\s*=/)
    if (constMatch) return constMatch[1]

    return null
  }

  /**
   * Generate unique algorithm ID
   */
  private generateId(): string {
    return `algo_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  /**
   * Store algorithm in conversation context for export
   */
  private async storeAlgorithmInContext(userId: string, algorithm: Algorithm): Promise<void> {
    try {
      // Prepare algorithm data for storage (exclude compiled function)
      const algorithmData = {
        id: algorithm.id,
        name: algorithm.name,
        type: algorithm.type,
        code: algorithm.code,
        parameters: algorithm.parameters,
        complexity: algorithm.complexity,
        description: algorithm.description
      }

      await axios.post(`${this.API_URL}/api/chat/store-algorithm`, {
        userId,
        algorithm: algorithmData
      })

      console.log(`📝 Stored algorithm in conversation context for user ${userId}`)
    } catch (error) {
      console.warn('Failed to store algorithm in context:', error)
      // Non-critical error, don't throw
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
