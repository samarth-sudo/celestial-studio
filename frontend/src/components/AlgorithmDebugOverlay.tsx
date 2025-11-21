/**
 * Algorithm Debug Overlay
 *
 * Displays real-time performance metrics and algorithm status for mobile robots
 */

import { useEffect, useState } from 'react'
import './AlgorithmDebugOverlay.css'

interface AlgorithmDebugOverlayProps {
  robotId: string
  position: [number, number, number]
  rotation: [number, number, number]
  currentWaypoint: number
  totalWaypoints: number
  algorithmActive: boolean
  algorithmName?: string
  fps?: number
}

export default function AlgorithmDebugOverlay({
  robotId,
  position,
  rotation,
  currentWaypoint,
  totalWaypoints,
  algorithmActive,
  algorithmName,
  fps
}: AlgorithmDebugOverlayProps) {
  const [isExpanded, setIsExpanded] = useState(true)

  return (
    <div className="algorithm-debug-overlay">
      <div className="debug-header" onClick={() => setIsExpanded(!isExpanded)}>
        <span className="debug-title">🤖 {robotId} Debug</span>
        <span className="debug-toggle">{isExpanded ? '▼' : '▶'}</span>
      </div>

      {isExpanded && (
        <div className="debug-content">
          {/* Algorithm Status */}
          <div className="debug-section">
            <h4>Algorithm Status</h4>
            <div className="debug-row">
              <span className="label">Active:</span>
              <span className={`status ${algorithmActive ? 'active' : 'inactive'}`}>
                {algorithmActive ? '✅ Running' : '⏸ Inactive'}
              </span>
            </div>
            {algorithmName && (
              <div className="debug-row">
                <span className="label">Name:</span>
                <span className="value">{algorithmName}</span>
              </div>
            )}
          </div>

          {/* Position & Rotation */}
          <div className="debug-section">
            <h4>Transform</h4>
            <div className="debug-row">
              <span className="label">Position:</span>
              <span className="value monospace">
                ({position[0].toFixed(2)}, {position[1].toFixed(2)}, {position[2].toFixed(2)})
              </span>
            </div>
            <div className="debug-row">
              <span className="label">Rotation:</span>
              <span className="value monospace">
                ({rotation[0].toFixed(2)}, {rotation[1].toFixed(2)}, {rotation[2].toFixed(2)})
              </span>
            </div>
          </div>

          {/* Waypoint Progress */}
          <div className="debug-section">
            <h4>Navigation</h4>
            <div className="debug-row">
              <span className="label">Waypoint:</span>
              <span className="value">
                {currentWaypoint + 1} / {totalWaypoints}
              </span>
            </div>
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${((currentWaypoint + 1) / totalWaypoints) * 100}%`
                }}
              />
            </div>
          </div>

          {/* Performance Metrics */}
          {fps !== undefined && (
            <div className="debug-section">
              <h4>Performance</h4>
              <div className="debug-row">
                <span className="label">FPS:</span>
                <span className={`value ${fps < 30 ? 'warning' : fps < 55 ? 'caution' : 'good'}`}>
                  {fps.toFixed(0)}
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
