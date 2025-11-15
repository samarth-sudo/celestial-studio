import type { ComputedPath } from '../types/PathPlanning'

interface PathExecutionPanelProps {
  isVisible: boolean
  onClose: () => void
  path: ComputedPath | null
  isPlaying: boolean
  onPlay: () => void
  onPause: () => void
  onStop: () => void
  currentWaypoint?: number
  totalWaypoints?: number
  distanceRemaining?: number
  status?: 'idle' | 'moving' | 'completed'
}

export default function PathExecutionPanel({
  isVisible,
  onClose,
  path,
  isPlaying,
  onPlay,
  onPause,
  onStop,
  currentWaypoint = 0,
  totalWaypoints = 0,
  distanceRemaining = 0,
  status = 'idle'
}: PathExecutionPanelProps) {
  if (!isVisible) return null

  const hasPath = path && path.isValid
  const progressPercent = totalWaypoints > 0
    ? Math.round((currentWaypoint / totalWaypoints) * 100)
    : 0

  const getStatusColor = () => {
    switch (status) {
      case 'moving':
        return '#00ff00'
      case 'completed':
        return '#00aaff'
      default:
        return '#888'
    }
  }

  const getStatusText = () => {
    switch (status) {
      case 'moving':
        return 'Moving'
      case 'completed':
        return 'Completed'
      default:
        return 'Idle'
    }
  }

  return (
    <div className="path-execution-window">
      <div className="window-header">
        <h4>🎯 Path Execution</h4>
        <button className="window-close" onClick={onClose} title="Close panel">
          ×
        </button>
      </div>

      <div className="execution-content">
        {!hasPath ? (
          <div className="no-path-message">
            <p>⚠️ No path computed</p>
            <p className="hint">Place origin and destination markers to generate a path</p>
          </div>
        ) : (
          <>
            {/* Status Indicator */}
            <div className="status-row">
              <div className="status-indicator">
                <div
                  className="status-light"
                  style={{ backgroundColor: getStatusColor() }}
                />
                <span className="status-text">{getStatusText()}</span>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="progress-section">
              <div className="progress-label">
                <span>Progress</span>
                <span className="progress-percent">{progressPercent}%</span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
            </div>

            {/* Stats Grid */}
            <div className="execution-stats">
              <div className="stat-box">
                <div className="stat-label">Waypoint</div>
                <div className="stat-value">
                  {currentWaypoint} / {totalWaypoints}
                </div>
              </div>

              <div className="stat-box">
                <div className="stat-label">Remaining</div>
                <div className="stat-value">
                  {distanceRemaining.toFixed(1)} m
                </div>
              </div>

              <div className="stat-box">
                <div className="stat-label">Total Length</div>
                <div className="stat-value">
                  {path.length.toFixed(1)} m
                </div>
              </div>

              <div className="stat-box">
                <div className="stat-label">Est. Time</div>
                <div className="stat-value">
                  {path.estimatedTime.toFixed(1)} s
                </div>
              </div>
            </div>

            {/* Playback Controls */}
            <div className="execution-controls">
              {!isPlaying ? (
                <button
                  className="control-btn play-btn"
                  onClick={onPlay}
                  disabled={status === 'completed'}
                  title="Start path execution"
                >
                  ▶️ Play
                </button>
              ) : (
                <button
                  className="control-btn pause-btn"
                  onClick={onPause}
                  title="Pause execution"
                >
                  ⏸️ Pause
                </button>
              )}

              <button
                className="control-btn stop-btn"
                onClick={onStop}
                disabled={status === 'idle'}
                title="Stop and reset"
              >
                ⏹️ Stop
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
