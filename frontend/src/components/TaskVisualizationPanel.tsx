import { useState, useEffect } from 'react'
import RobotService, { RobotType } from '../services/RobotService'
import './TaskVisualizationPanel.css'

interface TaskMetrics {
  success_rate: number
  average_reward: number
  average_steps: number
  total_episodes: number
}

interface TaskState {
  robot_name: string
  robot_type: RobotType
  task_name: string
  task_type: string
  status: 'idle' | 'running' | 'completed' | 'failed'
  current_step: number
  total_steps: number
  current_reward: number
  observations: Record<string, any>
  info: Record<string, any>
}

interface TaskVisualizationPanelProps {
  robotName: string
  robotType: RobotType
  taskName: string
  onClose?: () => void
}

export default function TaskVisualizationPanel({
  robotName,
  robotType,
  taskName,
  onClose
}: TaskVisualizationPanelProps) {
  const [taskState, setTaskState] = useState<TaskState>({
    robot_name: robotName,
    robot_type: robotType,
    task_name: taskName,
    task_type: '',
    status: 'idle',
    current_step: 0,
    total_steps: 1000,
    current_reward: 0,
    observations: {},
    info: {}
  })
  const [metrics, setMetrics] = useState<TaskMetrics>({
    success_rate: 0,
    average_reward: 0,
    average_steps: 0,
    total_episodes: 0
  })
  const [isRunning, setIsRunning] = useState(false)

  // Get task type icon and display name
  const taskIcon = RobotService.getTaskTypeIcon(taskName)
  const taskDisplayName = RobotService.getTaskTypeDisplayName(taskName)

  // Start simulation
  const handleStart = async () => {
    setIsRunning(true)
    setTaskState(prev => ({ ...prev, status: 'running' }))

    try {
      // Call simulation API
      const result = await RobotService.simulateRobot(robotName, {
        task_type: taskName,
        duration: 10.0,
        use_gui: false,
        params: {}
      })

      // Update state with results
      setTaskState(prev => ({
        ...prev,
        status: result.success ? 'completed' : 'failed',
        current_step: result.steps,
        current_reward: result.total_reward,
        info: result.info
      }))

      // Update metrics
      setMetrics(prev => {
        const newEpisodes = prev.total_episodes + 1
        const newSuccessRate = result.success
          ? (prev.success_rate * prev.total_episodes + 1) / newEpisodes
          : (prev.success_rate * prev.total_episodes) / newEpisodes
        const newAvgReward =
          (prev.average_reward * prev.total_episodes + result.total_reward) / newEpisodes
        const newAvgSteps =
          (prev.average_steps * prev.total_episodes + result.steps) / newEpisodes

        return {
          success_rate: newSuccessRate,
          average_reward: newAvgReward,
          average_steps: newAvgSteps,
          total_episodes: newEpisodes
        }
      })

      setIsRunning(false)
    } catch (error) {
      console.error('Simulation failed:', error)
      setTaskState(prev => ({ ...prev, status: 'failed' }))
      setIsRunning(false)
    }
  }

  // Reset simulation
  const handleReset = () => {
    setTaskState({
      robot_name: robotName,
      robot_type: robotType,
      task_name: taskName,
      task_type: '',
      status: 'idle',
      current_step: 0,
      total_steps: 1000,
      current_reward: 0,
      observations: {},
      info: {}
    })
  }

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return '#3b82f6'
      case 'completed': return '#22c55e'
      case 'failed': return '#ef4444'
      default: return '#64748b'
    }
  }

  return (
    <div className="task-visualization-panel">
      {/* Header */}
      <div className="task-viz-header">
        <div className="header-info">
          <h2>
            {taskIcon} {taskDisplayName}
          </h2>
          <p>
            {RobotService.getRobotTypeIcon(robotType)} {robotName}
          </p>
        </div>
        {onClose && (
          <button onClick={onClose} className="close-btn">×</button>
        )}
      </div>

      {/* Status Bar */}
      <div className="status-bar" style={{ borderLeftColor: getStatusColor(taskState.status) }}>
        <div className="status-item">
          <span className="status-label">Status:</span>
          <span className="status-value" style={{ color: getStatusColor(taskState.status) }}>
            {taskState.status.toUpperCase()}
          </span>
        </div>
        <div className="status-item">
          <span className="status-label">Step:</span>
          <span className="status-value">
            {taskState.current_step} / {taskState.total_steps}
          </span>
        </div>
        <div className="status-item">
          <span className="status-label">Reward:</span>
          <span className="status-value">
            {taskState.current_reward.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="progress-section">
        <div className="progress-label">
          Progress: {((taskState.current_step / taskState.total_steps) * 100).toFixed(1)}%
        </div>
        <div className="progress-bar-container">
          <div
            className="progress-bar-fill"
            style={{
              width: `${(taskState.current_step / taskState.total_steps) * 100}%`,
              backgroundColor: getStatusColor(taskState.status)
            }}
          />
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="metrics-section">
        <h3>📊 Performance Metrics</h3>
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-icon">✅</div>
            <div className="metric-value">{(metrics.success_rate * 100).toFixed(1)}%</div>
            <div className="metric-label">Success Rate</div>
          </div>
          <div className="metric-card">
            <div className="metric-icon">🏆</div>
            <div className="metric-value">{metrics.average_reward.toFixed(2)}</div>
            <div className="metric-label">Avg Reward</div>
          </div>
          <div className="metric-card">
            <div className="metric-icon">⏱️</div>
            <div className="metric-value">{Math.round(metrics.average_steps)}</div>
            <div className="metric-label">Avg Steps</div>
          </div>
          <div className="metric-card">
            <div className="metric-icon">🔄</div>
            <div className="metric-value">{metrics.total_episodes}</div>
            <div className="metric-label">Episodes</div>
          </div>
        </div>
      </div>

      {/* Task Info */}
      {Object.keys(taskState.info).length > 0 && (
        <div className="info-section">
          <h3>ℹ️ Task Info</h3>
          <div className="info-grid">
            {Object.entries(taskState.info).map(([key, value]) => (
              <div key={key} className="info-item">
                <span className="info-key">{key}:</span>
                <span className="info-value">
                  {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Control Buttons */}
      <div className="controls-section">
        <button
          onClick={handleStart}
          disabled={isRunning || taskState.status === 'running'}
          className="control-btn start-btn"
        >
          {isRunning ? '⏳ Running...' : '▶️ Start Simulation'}
        </button>
        <button
          onClick={handleReset}
          disabled={isRunning}
          className="control-btn reset-btn"
        >
          🔄 Reset
        </button>
      </div>
    </div>
  )
}
