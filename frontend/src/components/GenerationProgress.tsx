/**
 * Generation Progress Component
 *
 * Animated progress indicator for AI algorithm generation
 */

import { useState, useEffect } from 'react'
import './GenerationProgress.css'

interface GenerationProgressProps {
  isGenerating: boolean
  onCancel?: () => void
}

export default function GenerationProgress({ isGenerating, onCancel }: GenerationProgressProps) {
  const [elapsedTime, setElapsedTime] = useState(0)
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    if (!isGenerating) {
      setElapsedTime(0)
      setProgress(0)
      return
    }

    // Update elapsed time every 100ms
    const timeInterval = setInterval(() => {
      setElapsedTime(prev => prev + 0.1)
    }, 100)

    // Simulate progress (asymptotic growth - never reaches 100%)
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        // Logarithmic growth that slows down over time
        if (prev < 30) return prev + 2
        if (prev < 60) return prev + 0.8
        if (prev < 85) return prev + 0.3
        return prev + 0.1
      })
    }, 200)

    return () => {
      clearInterval(timeInterval)
      clearInterval(progressInterval)
    }
  }, [isGenerating])

  if (!isGenerating) return null

  const formatTime = (seconds: number) => {
    if (seconds < 60) {
      return `${seconds.toFixed(1)}s`
    }
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}m ${secs}s`
  }

  return (
    <div className="generation-progress">
      <div className="progress-header">
        <div className="progress-title">
          <div className="spinner" />
          <span>Generating Algorithm...</span>
        </div>
        <div className="progress-time">{formatTime(elapsedTime)}</div>
      </div>

      <div className="progress-bar-container">
        <div
          className="progress-bar-fill"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div className="progress-footer">
        <p className="progress-status">
          {progress < 20 && '🤖 Analyzing requirements...'}
          {progress >= 20 && progress < 50 && '🧠 Generating algorithm logic...'}
          {progress >= 50 && progress < 80 && '⚙️ Optimizing code structure...'}
          {progress >= 80 && '✨ Finalizing implementation...'}
        </p>

        {onCancel && (
          <button className="cancel-btn" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>
    </div>
  )
}
