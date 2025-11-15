/**
 * Parameter Panel Component
 *
 * Live parameter tuning for active algorithms
 */

import { useState, useEffect } from 'react'
import { getAlgorithmManager, type Algorithm, type AlgorithmParameter } from '../services/AlgorithmManager'
import './ParameterPanel.css'

interface ParameterPanelProps {
  algorithmId: string | null
  onParameterChange?: (paramName: string, value: any) => void
}

export default function ParameterPanel({
  algorithmId,
  onParameterChange
}: ParameterPanelProps) {
  const [algorithm, setAlgorithm] = useState<Algorithm | null>(null)
  const [parameters, setParameters] = useState<AlgorithmParameter[]>([])

  const manager = getAlgorithmManager()

  useEffect(() => {
    if (!algorithmId) {
      setAlgorithm(null)
      setParameters([])
      return
    }

    // Load algorithm
    const allAlgorithms = manager.getAllAlgorithms()
    const algo = allAlgorithms.find(a => a.id === algorithmId)

    if (algo) {
      setAlgorithm(algo)
      setParameters(algo.parameters)
    }
  }, [algorithmId, manager])

  const handleParameterChange = (paramName: string, value: any) => {
    try {
      if (!algorithmId) return

      // Update parameter in manager
      manager.updateParameter(algorithmId, paramName, value)

      // Update local state
      setParameters(prev =>
        prev.map(p =>
          p.name === paramName ? { ...p, value } : p
        )
      )

      // Notify parent
      if (onParameterChange) {
        onParameterChange(paramName, value)
      }

      console.log(`🎛️ Updated ${paramName} = ${value}`)
    } catch (error) {
      console.error('❌ Failed to update parameter:', error)
    }
  }

  if (!algorithm) {
    return (
      <div className="parameter-panel empty">
        <p>Select an algorithm to tune parameters</p>
      </div>
    )
  }

  if (parameters.length === 0) {
    return (
      <div className="parameter-panel empty">
        <p>This algorithm has no tunable parameters</p>
      </div>
    )
  }

  return (
    <div className="parameter-panel">
      <div className="panel-header">
        <h3>🎛️ Live Parameter Tuning</h3>
        <span className="algorithm-name">{algorithm.name}</span>
      </div>

      <div className="parameters-list">
        {parameters.map(param => (
          <div key={param.name} className="parameter-control">
            <div className="parameter-header">
              <label>{param.description || param.name}</label>
              <span className="parameter-value">
                {typeof param.value === 'number'
                  ? param.value.toFixed(2)
                  : String(param.value)}
              </span>
            </div>

            {param.type === 'number' && (
              <div className="slider-container">
                <input
                  type="range"
                  min={param.min ?? 0}
                  max={param.max ?? 100}
                  step={param.step ?? 0.1}
                  value={param.value}
                  onChange={(e) => handleParameterChange(param.name, parseFloat(e.target.value))}
                  className="parameter-slider"
                />
                <div className="slider-labels">
                  <span>{param.min ?? 0}</span>
                  <span>{param.max ?? 100}</span>
                </div>
              </div>
            )}

            {param.type === 'boolean' && (
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={param.value}
                  onChange={(e) => handleParameterChange(param.name, e.target.checked)}
                />
                <span className="toggle-slider"></span>
              </label>
            )}

            {param.type === 'string' && (
              <input
                type="text"
                value={param.value}
                onChange={(e) => handleParameterChange(param.name, e.target.value)}
                className="parameter-input"
              />
            )}
          </div>
        ))}
      </div>

      <div className="panel-footer">
        <p className="hint">
          💡 Adjust parameters in real-time to see immediate effects
        </p>
      </div>
    </div>
  )
}
