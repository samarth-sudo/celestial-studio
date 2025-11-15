import { useState, useEffect } from 'react'
import RobotService, { RobotType, RobotInfo } from '../services/RobotService'
import './RobotTypeSelector.css'

interface RobotTypeSelectorProps {
  onRobotSelect: (robotName: string, robotInfo: RobotInfo) => void
  onClose?: () => void
}

export default function RobotTypeSelector({ onRobotSelect, onClose }: RobotTypeSelectorProps) {
  const [selectedType, setSelectedType] = useState<RobotType>('manipulator')
  const [robots, setRobots] = useState<string[]>([])
  const [selectedRobot, setSelectedRobot] = useState<string | null>(null)
  const [robotInfo, setRobotInfo] = useState<RobotInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Robot types available
  const robotTypes: RobotType[] = [
    'manipulator',
    'mobile',
    'quadruped',
    'humanoid',
    'aerial',
    'mobile_manipulator'
  ]

  // Load robots when type changes
  useEffect(() => {
    loadRobots(selectedType)
  }, [selectedType])

  // Load robot info when robot is selected
  useEffect(() => {
    if (selectedRobot) {
      loadRobotInfo(selectedRobot)
    } else {
      setRobotInfo(null)
    }
  }, [selectedRobot])

  const loadRobots = async (type: RobotType) => {
    setLoading(true)
    setError(null)
    try {
      const robotList = await RobotService.listRobotsByType(type)
      setRobots(robotList)
      setSelectedRobot(null)
      setRobotInfo(null)
    } catch (err) {
      console.error('Failed to load robots:', err)
      setError('Failed to load robots')
      setRobots([])
    } finally {
      setLoading(false)
    }
  }

  const loadRobotInfo = async (robotName: string) => {
    try {
      const info = await RobotService.getRobotInfo(robotName)
      setRobotInfo(info)
    } catch (err) {
      console.error('Failed to load robot info:', err)
      setError('Failed to load robot details')
    }
  }

  const handleRobotSelect = (robotName: string) => {
    setSelectedRobot(robotName)
  }

  const handleConfirm = () => {
    if (selectedRobot && robotInfo) {
      onRobotSelect(selectedRobot, robotInfo)
    }
  }

  return (
    <div className="robot-type-selector">
      <div className="selector-header">
        <h2>🤖 Select Robot</h2>
        <p>Choose from 6 different robot types</p>
        {onClose && (
          <button onClick={onClose} className="close-btn">×</button>
        )}
      </div>

      {/* Robot Type Filter */}
      <div className="robot-type-grid">
        {robotTypes.map(type => (
          <button
            key={type}
            className={`robot-type-card ${selectedType === type ? 'active' : ''}`}
            onClick={() => setSelectedType(type)}
          >
            <div className="robot-type-icon">
              {RobotService.getRobotTypeIcon(type)}
            </div>
            <div className="robot-type-name">
              {RobotService.getRobotTypeDisplayName(type)}
            </div>
          </button>
        ))}
      </div>

      {/* Robot List */}
      <div className="robot-list-section">
        <h3>
          {RobotService.getRobotTypeIcon(selectedType)}{' '}
          {RobotService.getRobotTypeDisplayName(selectedType)} Robots
        </h3>

        {loading ? (
          <div className="loading-message">Loading robots...</div>
        ) : error ? (
          <div className="error-message">❌ {error}</div>
        ) : robots.length === 0 ? (
          <div className="empty-message">
            No {selectedType} robots available
          </div>
        ) : (
          <div className="robot-cards">
            {robots.map(robotName => (
              <div
                key={robotName}
                className={`robot-card ${selectedRobot === robotName ? 'selected' : ''}`}
                onClick={() => handleRobotSelect(robotName)}
              >
                <div className="robot-card-icon">
                  {RobotService.getRobotTypeIcon(selectedType)}
                </div>
                <div className="robot-card-name">{robotName}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Robot Details */}
      {robotInfo && (
        <div className="robot-details">
          <h3>Robot Details</h3>
          <div className="details-grid">
            <div className="detail-item">
              <span className="detail-label">Name:</span>
              <span className="detail-value">{robotInfo.name}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Type:</span>
              <span className="detail-value">
                {RobotService.getRobotTypeIcon(robotInfo.type)}{' '}
                {RobotService.getRobotTypeDisplayName(robotInfo.type)}
              </span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Fixed Base:</span>
              <span className="detail-value">{robotInfo.fixed_base ? 'Yes' : 'No'}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Actuator Groups:</span>
              <span className="detail-value">{robotInfo.num_actuator_groups}</span>
            </div>
            {robotInfo.ee_link_name && (
              <div className="detail-item">
                <span className="detail-label">End-Effector:</span>
                <span className="detail-value">{robotInfo.ee_link_name}</span>
              </div>
            )}
            {robotInfo.total_mass && (
              <div className="detail-item">
                <span className="detail-label">Total Mass:</span>
                <span className="detail-value">{robotInfo.total_mass.toFixed(2)} kg</span>
              </div>
            )}
            <div className="detail-item">
              <span className="detail-label">Max Linear Velocity:</span>
              <span className="detail-value">{robotInfo.max_linear_velocity.toFixed(2)} m/s</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Max Angular Velocity:</span>
              <span className="detail-value">{robotInfo.max_angular_velocity.toFixed(2)} rad/s</span>
            </div>
          </div>
        </div>
      )}

      {/* Confirm Button */}
      {selectedRobot && robotInfo && (
        <div className="selector-footer">
          <button onClick={handleConfirm} className="confirm-btn">
            Use {robotInfo.name}
          </button>
        </div>
      )}
    </div>
  )
}
