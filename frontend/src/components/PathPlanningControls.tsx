import { useState } from 'react'
import type { InteractionMode, ObstacleType, ComputedPath } from '../types/PathPlanning'
import './PathPlanningControls.css'

interface PathPlanningControlsProps {
  mode: InteractionMode
  onModeChange: (mode: InteractionMode) => void
  onClearAll?: () => void
  onClearPath?: () => void
  currentObstacleType: ObstacleType
  onObstacleTypeChange: (type: ObstacleType) => void
  computedPath: ComputedPath | null
  hasOrigin: boolean
  hasDestination: boolean
}

export default function PathPlanningControls({
  mode,
  onModeChange,
  onClearAll,
  onClearPath,
  currentObstacleType,
  onObstacleTypeChange,
  computedPath,
  hasOrigin,
  hasDestination
}: PathPlanningControlsProps) {
  const [showObstacleTypes, setShowObstacleTypes] = useState(false)
  const [showCoordinateInput, setShowCoordinateInput] = useState(false)

  const obstacleTypes: { type: ObstacleType; label: string; icon: string }[] = [
    { type: 'box', label: 'Box', icon: '📦' },
    { type: 'cylinder', label: 'Cylinder', icon: '🗜️' },
    { type: 'sphere', label: 'Sphere', icon: '⚽' },
    { type: 'plant', label: 'Plant', icon: '🌱' },
    { type: 'wall', label: 'Wall', icon: '🧱' },
    { type: 'custom-area', label: 'Custom Area', icon: '🎨' }
  ]

  return (
    <div className="path-planning-controls">
      <div className="controls-header">
        <h4>🎯 Path Planning Controls</h4>
        <button
          className="toggle-btn"
          onClick={() => setShowCoordinateInput(!showCoordinateInput)}
          title="Toggle coordinate input"
        >
          {showCoordinateInput ? '📍' : '⌨️'}
        </button>
      </div>

      {/* Main control buttons */}
      <div className="control-buttons">
        <button
          className={`control-btn ${mode === 'place-origin' ? 'active' : ''} ${hasOrigin ? 'has-value' : ''}`}
          onClick={() => onModeChange(mode === 'place-origin' ? 'none' : 'place-origin')}
        >
          <span className="btn-icon">🟢</span>
          <span className="btn-label">Set Origin</span>
          {hasOrigin && <span className="checkmark">✓</span>}
        </button>

        <button
          className={`control-btn ${mode === 'place-destination' ? 'active' : ''} ${hasDestination ? 'has-value' : ''}`}
          onClick={() => onModeChange(mode === 'place-destination' ? 'none' : 'place-destination')}
        >
          <span className="btn-icon">🔴</span>
          <span className="btn-label">Set Destination</span>
          {hasDestination && <span className="checkmark">✓</span>}
        </button>

        <div className="obstacle-btn-group">
          <button
            className={`control-btn ${mode === 'place-obstacle' ? 'active' : ''}`}
            onClick={() => {
              if (mode === 'place-obstacle') {
                onModeChange('none')
                setShowObstacleTypes(false)
              } else {
                onModeChange('place-obstacle')
                setShowObstacleTypes(true)
              }
            }}
          >
            <span className="btn-icon">🚧</span>
            <span className="btn-label">Place Obstacle</span>
          </button>

          {showObstacleTypes && (
            <div className="obstacle-types-dropdown">
              {obstacleTypes.map(({ type, label, icon }) => (
                <button
                  key={type}
                  className={`obstacle-type-btn ${currentObstacleType === type ? 'selected' : ''}`}
                  onClick={() => onObstacleTypeChange(type)}
                >
                  <span>{icon}</span>
                  <span>{label}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        <button
          className={`control-btn danger ${mode === 'delete-obstacle' ? 'active' : ''}`}
          onClick={() => onModeChange(mode === 'delete-obstacle' ? 'none' : 'delete-obstacle')}
        >
          <span className="btn-icon">🗑️</span>
          <span className="btn-label">Delete Obstacle</span>
        </button>
      </div>

      {/* Coordinate input (optional) */}
      {showCoordinateInput && (
        <div className="coordinate-input">
          <h5>Manual Coordinates</h5>
          <div className="coord-fields">
            <input type="number" placeholder="X" step="0.1" />
            <input type="number" placeholder="Z" step="0.1" />
            <button className="apply-btn">Apply</button>
          </div>
        </div>
      )}

      {/* Path info */}
      {computedPath && (
        <div className="path-info">
          <h5>📊 Path Information</h5>
          <div className="path-stats">
            <div className="stat">
              <span className="stat-label">Status:</span>
              <span className={`stat-value ${computedPath.isValid ? 'valid' : 'invalid'}`}>
                {computedPath.isValid ? '✓ Valid' : '✗ No path found'}
              </span>
            </div>
            {computedPath.isValid && (
              <>
                <div className="stat">
                  <span className="stat-label">Length:</span>
                  <span className="stat-value">{computedPath.length.toFixed(2)}m</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Waypoints:</span>
                  <span className="stat-value">{computedPath.waypoints.length}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Est. Time:</span>
                  <span className="stat-value">{computedPath.estimatedTime.toFixed(1)}s</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Algorithm:</span>
                  <span className="stat-value">{computedPath.algorithmUsed}</span>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div className="action-buttons">
        {onClearPath && computedPath && (
          <button className="action-btn secondary" onClick={onClearPath}>
            Clear Path
          </button>
        )}
        {onClearAll && (
          <button className="action-btn danger" onClick={onClearAll}>
            Clear All
          </button>
        )}
      </div>

      {/* Mode indicator */}
      {mode !== 'none' && (
        <div className="mode-indicator">
          <div className="mode-indicator-content">
            {mode === 'place-origin' && '🟢 Click on ground to set origin'}
            {mode === 'place-destination' && '🔴 Click on ground to set destination'}
            {mode === 'place-obstacle' && `🚧 Click to place ${currentObstacleType}`}
            {mode === 'delete-obstacle' && '🗑️ Click obstacle to delete'}
          </div>
          <button className="cancel-mode-btn" onClick={() => onModeChange('none')}>
            Cancel
          </button>
        </div>
      )}
    </div>
  )
}
