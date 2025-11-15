import { useState, useEffect } from 'react'
import RobotService, { RobotType } from '../services/RobotService'
import './TaskSelector.css'

interface TaskInfo {
  name: string
  type: string
  description: string
  compatibleRobots: RobotType[]
}

interface TaskSelectorProps {
  robotType?: RobotType
  onTaskSelect: (taskName: string, taskInfo: TaskInfo) => void
  onClose?: () => void
}

export default function TaskSelector({ robotType, onTaskSelect, onClose }: TaskSelectorProps) {
  const [tasks, setTasks] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<string>('all')
  const [selectedTask, setSelectedTask] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadTasks()
  }, [robotType])

  const loadTasks = async () => {
    setLoading(true)
    setError(null)
    try {
      let taskList: string[]
      if (robotType) {
        // Load tasks compatible with specific robot type
        taskList = await RobotService.getCompatibleTasks(robotType)
      } else {
        // Load all tasks
        taskList = await RobotService.listAllTasks()
      }
      setTasks(taskList)
      setLoading(false)
    } catch (err) {
      console.error('Failed to load tasks:', err)
      setError('Failed to load tasks')
      setLoading(false)
    }
  }

  // Get task info
  const getTaskInfo = (taskName: string): TaskInfo => {
    const taskMap: Record<string, TaskInfo> = {
      reach: {
        name: 'Reach Target',
        type: 'manipulation',
        description: 'Move end-effector to target pose',
        compatibleRobots: ['manipulator', 'mobile_manipulator']
      },
      lift: {
        name: 'Lift Object',
        type: 'manipulation',
        description: 'Pick and lift object to target height',
        compatibleRobots: ['manipulator', 'mobile_manipulator']
      },
      navigation: {
        name: 'Navigate Waypoints',
        type: 'navigation',
        description: 'Follow waypoints to reach destination',
        compatibleRobots: ['mobile', 'quadruped', 'mobile_manipulator']
      },
      walking: {
        name: 'Forward Walking',
        type: 'locomotion',
        description: 'Walk forward maintaining balance',
        compatibleRobots: ['quadruped', 'humanoid']
      },
      flight: {
        name: '3D Flight',
        type: 'locomotion',
        description: 'Navigate 3D waypoints in air',
        compatibleRobots: ['aerial']
      }
    }
    return taskMap[taskName] || {
      name: taskName,
      type: 'unknown',
      description: taskName,
      compatibleRobots: []
    }
  }

  const filteredTasks = tasks.filter(taskName => {
    if (filter === 'all') return true
    const info = getTaskInfo(taskName)
    return info.type === filter
  })

  const handleSelect = (taskName: string) => {
    setSelectedTask(taskName)
  }

  const handleConfirm = () => {
    if (selectedTask) {
      const taskInfo = getTaskInfo(selectedTask)
      onTaskSelect(selectedTask, taskInfo)
    }
  }

  const getTaskTypeIcon = (type: string) => {
    switch (type) {
      case 'manipulation': return '🦾'
      case 'navigation': return '🧭'
      case 'locomotion': return '🦿'
      case 'control': return '🎮'
      default: return '🤖'
    }
  }

  return (
    <div className="task-selector">
      <div className="task-selector-header">
        <h2>📋 Select Task</h2>
        {robotType && (
          <p>Tasks compatible with {RobotService.getRobotTypeDisplayName(robotType)}</p>
        )}
        {onClose && (
          <button onClick={onClose} className="close-btn">×</button>
        )}
      </div>

      <div className="task-filters">
        <button
          className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
          onClick={() => setFilter('all')}
        >
          All Tasks
        </button>
        <button
          className={`filter-btn ${filter === 'manipulation' ? 'active' : ''}`}
          onClick={() => setFilter('manipulation')}
        >
          🦾 Manipulation
        </button>
        <button
          className={`filter-btn ${filter === 'navigation' ? 'active' : ''}`}
          onClick={() => setFilter('navigation')}
        >
          🧭 Navigation
        </button>
        <button
          className={`filter-btn ${filter === 'locomotion' ? 'active' : ''}`}
          onClick={() => setFilter('locomotion')}
        >
          🦿 Locomotion
        </button>
      </div>

      {loading ? (
        <div className="task-loading">Loading tasks...</div>
      ) : error ? (
        <div className="task-error">❌ {error}</div>
      ) : filteredTasks.length === 0 ? (
        <div className="task-empty">No tasks available</div>
      ) : (
        <>
          <div className="task-list">
            {filteredTasks.map(taskName => {
              const taskInfo = getTaskInfo(taskName)
              return (
                <div
                  key={taskName}
                  className={`task-card ${selectedTask === taskName ? 'selected' : ''}`}
                  onClick={() => handleSelect(taskName)}
                >
                  <div className="task-card-header">
                    <span className="task-icon">
                      {RobotService.getTaskTypeIcon(taskName)}
                    </span>
                    <div className="task-info">
                      <h3>{taskInfo.name}</h3>
                      <p className="task-description">{taskInfo.description}</p>
                    </div>
                  </div>

                  <div className="task-card-footer">
                    <span className="task-type">{taskInfo.type}</span>
                    <span className="task-robots">
                      {taskInfo.compatibleRobots.map(rt =>
                        RobotService.getRobotTypeIcon(rt)
                      ).join(' ')}
                    </span>
                  </div>
                </div>
              )
            })}
          </div>

          {selectedTask && (
            <div className="task-selector-footer">
              <div className="selected-task-info">
                <strong>Selected:</strong> {getTaskInfo(selectedTask).name}
              </div>
              <button onClick={handleConfirm} className="confirm-btn">
                Use This Task
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
