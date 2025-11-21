/**
 * Algorithm Controls Component
 *
 * UI for managing and applying algorithms to robots
 */

import { useState, useEffect, useCallback } from 'react'
import { getAlgorithmManager, type Algorithm, type AlgorithmParameter } from '../services/AlgorithmManager'
import type { AlgorithmType } from '../types'
import AlgorithmLibrary from './AlgorithmLibrary'
import ParameterPanel from './ParameterPanel'
import GenerationProgress from './GenerationProgress'
import { useToast } from '../hooks/useToast'
import axios from 'axios'
import './AlgorithmControls.css'

interface AlgorithmControlsProps {
  robotId: string
  robotType: 'mobile' | 'arm' | 'drone'
  onAlgorithmApplied?: (algorithm: Algorithm) => void
  onTestInSandbox?: (algorithmId: string) => void
}

interface AlgorithmTemplate {
  name: string
  type: AlgorithmType
  description: string
  code_template: string
  complexity: string
  parameters: AlgorithmParameter[]
}

export default function AlgorithmControls({
  robotId,
  robotType,
  onAlgorithmApplied,
  onTestInSandbox
}: AlgorithmControlsProps) {
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([])
  const [activeAlgorithmIds, setActiveAlgorithmIds] = useState<Set<string>>(new Set())
  const [isGenerating, setIsGenerating] = useState(false)
  const [description, setDescription] = useState('')
  const [algorithmType, setAlgorithmType] = useState<Algorithm['type']>('path_planning')
  const [showGenerator, setShowGenerator] = useState(false)
  const [expandedCodeId, setExpandedCodeId] = useState<string | null>(null)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [showLibrary, setShowLibrary] = useState(false)
  const [expandedParametersId, setExpandedParametersId] = useState<string | null>(null)

  const manager = getAlgorithmManager()
  const { showToast } = useToast()

  const loadAlgorithms = useCallback(() => {
    const allAlgorithms = manager.getAllAlgorithms()
    setAlgorithms(allAlgorithms)

    // Get all active algorithms for this robot
    const activeAlgos = manager.getActiveAlgorithms(robotId)
    const activeIds = new Set(activeAlgos.map(a => a.id))
    setActiveAlgorithmIds(activeIds)
  }, [manager, robotId])

  useEffect(() => {
    // Load algorithms
    loadAlgorithms()
  }, [loadAlgorithms])

  const handleGenerate = async () => {
    if (!description.trim()) return

    setIsGenerating(true)
    try {
      console.log('🚀 Starting algorithm generation...')
      const algorithm = await manager.generateAlgorithm(
        description,
        robotType,
        algorithmType
      )

      setAlgorithms(prev => [...prev, algorithm])
      setDescription('')
      setShowGenerator(false)

      console.log('✅ Algorithm generated:', algorithm.name)

      // Success toast
      showToast({
        type: 'success',
        title: 'Algorithm Generated',
        message: `"${algorithm.name}" is ready to use`
      })
    } catch (error: unknown) {
      console.error('❌ Generation failed:', error)

      // Smart error handling with specific toast messages
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNABORTED') {
          showToast({
            type: 'error',
            title: 'Generation Timeout',
            message: 'Algorithm generation took too long (>60s). The AI may be overloaded.',
            action: {
              label: 'Try Again',
              onClick: () => handleGenerate()
            }
          })
        } else if (error.response?.status === 500) {
          showToast({
            type: 'error',
            title: 'AI Service Error',
            message: error.response?.data?.detail || 'The AI code generator encountered an error.',
            action: {
              label: 'Copy Error',
              onClick: () => navigator.clipboard.writeText(JSON.stringify(error.response?.data))
            }
          })
        } else if (!error.response) {
          showToast({
            type: 'error',
            title: 'Backend Offline',
            message: `Cannot connect to backend server. Is it running?`,
            action: {
              label: 'Retry',
              onClick: () => handleGenerate()
            }
          })
        } else {
          showToast({
            type: 'error',
            title: 'Generation Failed',
            message: error.message || 'Unknown error occurred'
          })
        }
      } else {
        const errorMessage = error instanceof Error ? error.message : 'Failed to generate algorithm'
        showToast({
          type: 'error',
          title: 'Generation Failed',
          message: errorMessage
        })
      }
    } finally {
      setIsGenerating(false)
    }
  }

  const handleApply = (algorithmId: string) => {
    try {
      const algorithm = algorithms.find(a => a.id === algorithmId)
      if (!algorithm) return

      manager.applyAlgorithm(robotId, algorithmId)

      // Update active algorithms set
      setActiveAlgorithmIds(prev => new Set([...prev, algorithmId]))

      if (onAlgorithmApplied) {
        onAlgorithmApplied(algorithm)
      }

      console.log(`🔄 Applied ${algorithm.name} to robot`)

      // Success toast
      showToast({
        type: 'success',
        title: 'Algorithm Applied',
        message: `"${algorithm.name}" is now active on ${robotId}`
      })
    } catch (error) {
      console.error('❌ Failed to apply algorithm:', error)
      const message = error instanceof Error ? error.message : 'Unknown error occurred'
      showToast({
        type: 'error',
        title: 'Cannot Apply Algorithm',
        message: `Failed to activate the algorithm. ${message}`
      })
    }
  }

  const handleRemove = (algorithmId: string) => {
    try {
      const algorithm = algorithms.find(a => a.id === algorithmId)
      if (!algorithm) return

      manager.removeAlgorithm(robotId, algorithmId)

      // Update active algorithms set
      setActiveAlgorithmIds(prev => {
        const newSet = new Set(prev)
        newSet.delete(algorithmId)
        return newSet
      })

      console.log(`🗑️ Removed ${algorithm.name} from robot`)

      // Success toast
      showToast({
        type: 'success',
        title: 'Algorithm Removed',
        message: `"${algorithm.name}" deactivated from ${robotId}`
      })
    } catch (error) {
      console.error('❌ Failed to remove algorithm:', error)
      const message = error instanceof Error ? error.message : 'Unknown error occurred'
      showToast({
        type: 'error',
        title: 'Cannot Remove Algorithm',
        message: `Failed to deactivate the algorithm. ${message}`
      })
    }
  }

  const toggleCodeView = (algorithmId: string) => {
    setExpandedCodeId(expandedCodeId === algorithmId ? null : algorithmId)
  }

  const toggleParametersView = (algorithmId: string) => {
    setExpandedParametersId(expandedParametersId === algorithmId ? null : algorithmId)
  }

  const handleParameterChange = (algorithmId: string, paramName: string, value: any) => {
    console.log(`🎛️ Parameter changed: ${paramName} = ${value} for ${algorithmId}`)

    // Success toast for parameter changes
    showToast({
      type: 'info',
      title: 'Parameter Updated',
      message: `${paramName} = ${typeof value === 'number' ? value.toFixed(2) : value}`,
      duration: 2000
    })
  }

  const copyCode = async (code: string, algorithmId: string) => {
    try {
      await navigator.clipboard.writeText(code)
      setCopiedId(algorithmId)
      setTimeout(() => setCopiedId(null), 2000)
    } catch (error) {
      console.error('Failed to copy code:', error)
    }
  }

  const getAlgorithmTypeColor = (type: Algorithm['type']) => {
    const colors = {
      path_planning: '#3498db',
      obstacle_avoidance: '#e74c3c',
      inverse_kinematics: '#9b59b6',
      computer_vision: '#f39c12',
      motion_control: '#2ecc71'
    }
    return colors[type] || '#95a5a6'
  }

  const handleUseTemplate = async (template: AlgorithmTemplate) => {
    try {
      // Create algorithm from template
      const algorithm: Algorithm = {
        id: `template-${Date.now()}`,
        name: template.name,
        type: template.type,
        description: template.description,
        code: template.code_template,
        complexity: template.complexity,
        parameters: template.parameters,
        metadata: {
          robotType: robotType,
          generatedAt: new Date().toISOString(),
          isTemplate: true
        }
      }

      // Add to algorithm manager
      manager.addAlgorithm(algorithm)
      setAlgorithms(prev => [...prev, algorithm])
      setShowLibrary(false)

      console.log('✅ Template added:', template.name)

      // Success toast
      showToast({
        type: 'success',
        title: 'Template Added',
        message: `"${template.name}" is ready to use`,
        action: {
          label: 'Apply Now',
          onClick: () => handleApply(algorithm.id)
        }
      })
    } catch (error) {
      console.error('❌ Failed to use template:', error)
      const message = error instanceof Error ? error.message : 'Unknown error occurred'
      showToast({
        type: 'error',
        title: 'Cannot Use Template',
        message: `Failed to add template. ${message}`
      })
    }
  }

  const handleCustomizeTemplate = (template: AlgorithmTemplate) => {
    // Pre-fill the generator with template info
    setAlgorithmType(template.type)
    setDescription(`Based on ${template.name}: ${template.description}`)
    setShowLibrary(false)
    setShowGenerator(true)
  }

  return (
    <div className="algorithm-controls">
      <div className="controls-header">
        <h3>🧠 Algorithm Controls</h3>
        <div className="header-buttons">
          <button
            className="browse-btn"
            onClick={() => setShowLibrary(true)}
          >
            📚 Browse Templates
          </button>
          <button
            className="generate-btn"
            onClick={() => setShowGenerator(!showGenerator)}
          >
            {showGenerator ? '✕ Close' : '+ Generate New'}
          </button>
        </div>
      </div>

      {showGenerator && (
        <div className="algorithm-generator">
          <h4>Generate Algorithm</h4>

          <div className="form-group">
            <label>Algorithm Type:</label>
            <select
              value={algorithmType}
              onChange={(e) => setAlgorithmType(e.target.value as Algorithm['type'])}
              disabled={isGenerating}
            >
              <option value="path_planning">Path Planning</option>
              <option value="obstacle_avoidance">Obstacle Avoidance</option>
              <option value="inverse_kinematics">Inverse Kinematics</option>
              <option value="computer_vision">Computer Vision</option>
              <option value="motion_control">Motion Control</option>
            </select>
          </div>

          <div className="form-group">
            <label>Description:</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe the algorithm behavior...&#10;&#10;Examples:&#10;- Navigate to goal while avoiding obstacles using DWA&#10;- Find optimal path using A* algorithm&#10;- Reach target position with FABRIK IK solver"
              rows={4}
              disabled={isGenerating}
            />
          </div>

          <button
            className="generate-algorithm-btn"
            onClick={handleGenerate}
            disabled={isGenerating || !description.trim()}
          >
            {isGenerating ? '⏳ Generating...' : '🚀 Generate Algorithm'}
          </button>

          <GenerationProgress
            isGenerating={isGenerating}
            onCancel={() => setIsGenerating(false)}
          />
        </div>
      )}

      <div className="algorithms-list">
        <h4>Available Algorithms ({algorithms.length})</h4>

        {algorithms.length === 0 ? (
          <p className="empty-state">No algorithms yet. Generate one to get started!</p>
        ) : (
          algorithms.map(algo => {
            const isActive = activeAlgorithmIds.has(algo.id)
            return (
              <div
                key={algo.id}
                className={`algorithm-card ${isActive ? 'active' : ''}`}
              >
                <div className="algorithm-header">
                  <h5>{algo.name}</h5>
                  <span
                    className="algorithm-type-badge"
                    style={{ backgroundColor: getAlgorithmTypeColor(algo.type) }}
                  >
                    {algo.type.replace('_', ' ')}
                  </span>
                </div>

                <p className="algorithm-description">{algo.description}</p>

                <div className="algorithm-meta">
                  <span className="complexity">{algo.complexity}</span>
                  <span className="param-count">
                    {algo.parameters.length} parameters
                  </span>
                </div>

                <div className="algorithm-buttons">
                  <button
                    className={`apply-btn ${isActive ? 'applied' : ''}`}
                    onClick={() => handleApply(algo.id)}
                    disabled={isActive}
                  >
                    {isActive ? '✓ Active' : 'Apply to Robot'}
                  </button>
                  {isActive && (
                    <button
                      className="remove-btn"
                      onClick={() => handleRemove(algo.id)}
                    >
                      Remove
                    </button>
                  )}
                </div>

              <div className="algorithm-actions">
                <button
                  className="view-code-btn"
                  onClick={() => toggleCodeView(algo.id)}
                >
                  {expandedCodeId === algo.id ? '▲ Hide Code' : '▼ View Code'}
                </button>

                {algo.parameters.length > 0 && (
                  <button
                    className="tune-params-btn"
                    onClick={() => toggleParametersView(algo.id)}
                  >
                    {expandedParametersId === algo.id ? '▲ Hide Parameters' : '🎛️ Tune Parameters'}
                  </button>
                )}

                {algo.type === 'path_planning' && onTestInSandbox && (
                  <button
                    className="test-sandbox-btn"
                    onClick={() => onTestInSandbox(algo.id)}
                  >
                    🎯 Test in Path Sandbox
                  </button>
                )}
              </div>

              {expandedCodeId === algo.id && (
                <div className="code-section">
                  <div className="code-header">
                    <span>Algorithm Implementation</span>
                    <button
                      className="copy-code-btn"
                      onClick={() => copyCode(algo.code, algo.id)}
                    >
                      {copiedId === algo.id ? '✓ Copied!' : '📋 Copy'}
                    </button>
                  </div>
                  <pre className="code-block">
                    <code>{algo.code}</code>
                  </pre>
                </div>
              )}

              {expandedParametersId === algo.id && (
                <div className="parameters-section">
                  <ParameterPanel
                    algorithmId={algo.id}
                    onParameterChange={(paramName, value) => handleParameterChange(algo.id, paramName, value)}
                  />
                </div>
              )}
            </div>
          )})
        )}
      </div>

      <AlgorithmLibrary
        isOpen={showLibrary}
        onClose={() => setShowLibrary(false)}
        onUseTemplate={handleUseTemplate}
        onCustomize={handleCustomizeTemplate}
      />
    </div>
  )
}
