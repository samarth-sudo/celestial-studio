/**
 * Robot Service
 *
 * Service for interacting with the multi-robot API backend.
 * Supports 6 robot types: manipulator, mobile, quadruped, humanoid, aerial, mobile_manipulator
 */

import axios from 'axios'

const API_URL = 'http://localhost:8000'

export type RobotType =
  | 'manipulator'
  | 'mobile'
  | 'quadruped'
  | 'humanoid'
  | 'aerial'
  | 'mobile_manipulator'

export interface RobotInfo {
  name: string
  type: RobotType
  urdf_path: string
  fixed_base: boolean
  num_actuator_groups: number
  ee_link_name?: string
  base_link_name?: string
  total_mass?: number
  max_linear_velocity: number
  max_angular_velocity: number
}

export interface SimulationRequest {
  task_type: string
  duration: number
  use_gui: boolean
  params: Record<string, any>
}

export interface SimulationResponse {
  success: boolean
  robot_name: string
  task_type: string
  steps: number
  total_reward: number
  final_state: Record<string, any>
  info: Record<string, any>
}

export interface Controller {
  name: string
  type: RobotType
  description: string
}

export interface RobotExample {
  description: string
  example_code: string
  compatible_tasks: string[]
}

/**
 * RobotService - API client for multi-robot system
 */
export class RobotService {
  /**
   * Get list of all available robots
   */
  static async listAllRobots(): Promise<string[]> {
    try {
      const response = await axios.get(`${API_URL}/api/robots/list`)
      return response.data
    } catch (error) {
      console.error('Failed to list robots:', error)
      throw error
    }
  }

  /**
   * Get list of robots of a specific type
   */
  static async listRobotsByType(robotType: RobotType): Promise<string[]> {
    try {
      const response = await axios.get(`${API_URL}/api/robots/list/${robotType}`)
      return response.data
    } catch (error) {
      console.error(`Failed to list ${robotType} robots:`, error)
      throw error
    }
  }

  /**
   * Get detailed information about a specific robot
   */
  static async getRobotInfo(robotName: string): Promise<RobotInfo> {
    try {
      const response = await axios.get(`${API_URL}/api/robots/${robotName}/info`)
      return response.data
    } catch (error) {
      console.error(`Failed to get info for robot ${robotName}:`, error)
      throw error
    }
  }

  /**
   * Run a simulation for a specific robot
   */
  static async simulateRobot(
    robotName: string,
    request: SimulationRequest
  ): Promise<SimulationResponse> {
    try {
      const response = await axios.post(
        `${API_URL}/api/robots/${robotName}/simulate`,
        request
      )
      return response.data
    } catch (error) {
      console.error(`Failed to simulate robot ${robotName}:`, error)
      throw error
    }
  }

  /**
   * Get list of all available tasks
   */
  static async listAllTasks(): Promise<string[]> {
    try {
      const response = await axios.get(`${API_URL}/api/robots/tasks/list`)
      return response.data
    } catch (error) {
      console.error('Failed to list tasks:', error)
      throw error
    }
  }

  /**
   * Get tasks compatible with a robot type
   */
  static async getCompatibleTasks(robotType: RobotType): Promise<string[]> {
    try {
      const response = await axios.get(`${API_URL}/api/robots/tasks/${robotType}/compatible`)
      return response.data
    } catch (error) {
      console.error(`Failed to get compatible tasks for ${robotType}:`, error)
      throw error
    }
  }

  /**
   * Get list of all available controllers
   */
  static async listControllers(): Promise<Controller[]> {
    try {
      const response = await axios.get(`${API_URL}/api/robots/controllers/list`)
      return response.data
    } catch (error) {
      console.error('Failed to list controllers:', error)
      throw error
    }
  }

  /**
   * Get example code for a robot type
   */
  static async getExample(robotType: RobotType): Promise<RobotExample> {
    try {
      const response = await axios.get(`${API_URL}/api/robots/examples/${robotType}`)
      return response.data
    } catch (error) {
      console.error(`Failed to get example for ${robotType}:`, error)
      throw error
    }
  }

  /**
   * Get robot type icon (for UI)
   */
  static getRobotTypeIcon(robotType: RobotType): string {
    const icons: Record<RobotType, string> = {
      manipulator: '🦾',
      mobile: '🤖',
      quadruped: '🐕',
      humanoid: '🚶',
      aerial: '🚁',
      mobile_manipulator: '🦿'
    }
    return icons[robotType] || '🤖'
  }

  /**
   * Get robot type display name
   */
  static getRobotTypeDisplayName(robotType: RobotType): string {
    const names: Record<RobotType, string> = {
      manipulator: 'Robotic Arm',
      mobile: 'Mobile Robot',
      quadruped: 'Quadruped',
      humanoid: 'Humanoid',
      aerial: 'Drone',
      mobile_manipulator: 'Mobile Manipulator'
    }
    return names[robotType] || robotType
  }

  /**
   * Get task type icon (for UI)
   */
  static getTaskTypeIcon(taskType: string): string {
    const icons: Record<string, string> = {
      reach: '🎯',
      lift: '⬆️',
      navigation: '🧭',
      walking: '🦿',
      flight: '✈️'
    }
    return icons[taskType] || '📋'
  }

  /**
   * Get task type display name
   */
  static getTaskTypeDisplayName(taskType: string): string {
    const names: Record<string, string> = {
      reach: 'Reach Target',
      lift: 'Lift Object',
      navigation: 'Navigate Waypoints',
      walking: 'Forward Walking',
      flight: '3D Flight'
    }
    return names[taskType] || taskType
  }
}

// Export singleton (optional)
export default RobotService
