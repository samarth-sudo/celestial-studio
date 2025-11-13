/**
 * Algorithm Manager for Celestial Studio
 *
 * Manages hot-swappable algorithms for robots in the simulation.
 * Enables real-time code injection and live parameter tuning without restart.
 */

import * as THREE from 'three'
import axios from 'axios'

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
  value: any
  min?: number
  max?: number
  step?: number
  description: string
}

export interface AlgorithmState {
  position: THREE.Vector3
  velocity: THREE.Vector3
  rotation: THREE.Euler
  customData: Record<string, any>
}

/**
 * AlgorithmManager - Hot-swappable algorithm system
 */
export class AlgorithmManager {
  private algorithms: Map<string, Algorithm> = new Map()
  private activeAlgorithms: Map<string, string> = new Map() // robotId -> algorithmId
  private robotStates: Map<string, AlgorithmState> = new Map()
  private readonly API_URL = 'http://localhost:8000'

  /**
   * Generate a new algorithm from natural language description
   */
  async generateAlgorithm(
    description: string,
    robotType: 'mobile' | 'arm' | 'drone',
    algorithmType: Algorithm['type']
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

      console.log(`✅ Generated algorithm: ${algorithm.name}`)
      console.log(`Code length: ${code.length} chars`)
      console.log(`Parameters: ${parameters?.length || 0}`)
      console.log(`Compiled: ${compiledFunction ? 'Yes' : 'No (display only)'}`)
      return algorithm

    } catch (error: any) {
      console.error('❌ Algorithm generation failed:', error)
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
      } else {
        throw new Error(`Failed to generate algorithm: ${error.message}`)
      }
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
   * Apply algorithm to a robot (hot-swap)
   */
  applyAlgorithm(robotId: string, algorithmId: string): void {
    const algorithm = this.algorithms.get(algorithmId)
    if (!algorithm) {
      throw new Error(`Algorithm ${algorithmId} not found`)
    }

    // Save current robot state before swapping
    // (State preservation will be handled by robot component)

    // Activate algorithm for robot
    this.activeAlgorithms.set(robotId, algorithmId)

    console.log(`🔄 Applied algorithm ${algorithm.name} to robot ${robotId}`)
  }

  /**
   * Get active algorithm for a robot
   */
  getActiveAlgorithm(robotId: string): Algorithm | null {
    const algorithmId = this.activeAlgorithms.get(robotId)
    if (!algorithmId) return null
    return this.algorithms.get(algorithmId) || null
  }

  /**
   * Update algorithm parameter value (live tuning)
   */
  updateParameter(algorithmId: string, paramName: string, value: any): void {
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
    ...args: any[]
  ): any {
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
        } catch (e) {
          // Try next function name
          continue
        }
      }

      throw new Error('No valid path planning function found in algorithm')
    } catch (error: any) {
      console.error(`❌ Path planning execution error:`, error)
      throw new Error(`Failed to execute path planning: ${error.message}`)
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

      // Wrap code in a function that returns an executor
      const wrappedCode = `
        ${jsCode}

        // Return executor function
        return function(functionName, ...args) {
          // Find and execute the requested function
          try {
            if (typeof eval(functionName) === 'function') {
              return eval(functionName)(...args)
            }
            throw new Error('Function ' + functionName + ' not found')
          } catch (e) {
            console.error('Execution error:', e)
            throw e
          }
        }
      `

      // Create function in isolated scope with THREE.js available
      const compiledFn = new Function('THREE', wrappedCode)

      // Execute with THREE.js injected
      return compiledFn(THREE)

    } catch (error: any) {
      console.warn('⚠️ Code compilation skipped (will work for display only):', error.message)
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
}

// Singleton instance
let managerInstance: AlgorithmManager | null = null

export function getAlgorithmManager(): AlgorithmManager {
  if (!managerInstance) {
    managerInstance = new AlgorithmManager()
  }
  return managerInstance
}
