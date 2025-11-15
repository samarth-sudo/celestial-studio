import { useState } from 'react'
import axios from 'axios'
import IsaacLabVideoPlayer from './IsaacLabVideoPlayer'
import './IsaacSimulationPanel.css'

interface SimulationMetrics {
  frames_simulated?: number
  average_fps?: number
  physics_steps?: number
}

interface SimulationResult {
  success: boolean
  metrics?: SimulationMetrics
  video_path?: string
  error?: string
}

interface IsaacSimulationPanelProps {
  sceneConfig: any
  onClose?: () => void
}

export default function IsaacSimulationPanel({ sceneConfig, onClose }: IsaacSimulationPanelProps) {
  const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'error'>('idle')
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<SimulationResult | null>(null)
  const [streamSessionId, setStreamSessionId] = useState<string | null>(null)
  const [videoSource, setVideoSource] = useState<string | null>(null)

  // Simulation parameters (could be configurable)
  const [duration, setDuration] = useState(10)
  const [recordVideo, setRecordVideo] = useState(true)
  const [fps, setFps] = useState(30)

  const runSimulation = async () => {
    setStatus('running')
    setProgress(0)
    setResult(null)

    try {
      console.log('Starting Isaac Lab simulation...')

      // Simulate progress (since Modal blocks, we can't get real-time progress)
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 5, 95))
      }, 1000)

      const response = await axios.post('http://localhost:8000/api/isaac-lab/simulate', {
        scene_config: sceneConfig,
        duration,
        record_video: recordVideo,
        headless: true,
        fps
      })

      clearInterval(progressInterval)
      setProgress(100)

      console.log('Simulation completed:', response.data)

      setResult(response.data.result)
      setStatus('completed')

      // If video was recorded, set up streaming
      if (response.data.video_path) {
        setVideoSource(response.data.video_path)
        // Create WebRTC session for streaming
        await createStreamingSession(response.data.video_path)
      }

    } catch (error: any) {
      console.error('Simulation error:', error)
      setStatus('error')
      setResult({
        success: false,
        error: error.response?.data?.detail || error.message || 'Simulation failed'
      })
    }
  }

  const createStreamingSession = async (videoPath: string) => {
    try {
      const response = await axios.post('http://localhost:8000/api/webrtc/session', {
        metadata: {
          type: 'isaac_simulation',
          video_path: videoPath
        }
      })

      console.log('WebRTC session created:', response.data.session_id)
      setStreamSessionId(response.data.session_id)
    } catch (error) {
      console.error('Failed to create streaming session:', error)
    }
  }

  const reset = () => {
    setStatus('idle')
    setProgress(0)
    setResult(null)
    setStreamSessionId(null)
    setVideoSource(null)
  }

  return (
    <div className="isaac-simulation-panel">
      <div className="panel-header">
        <h2>Isaac Lab GPU Simulation</h2>
        {onClose && (
          <button onClick={onClose} className="close-button">×</button>
        )}
      </div>

      <div className="panel-content">
        {/* Configuration Section */}
        {status === 'idle' && (
          <div className="config-section">
            <h3>Simulation Configuration</h3>

            <div className="config-grid">
              <div className="config-item">
                <label>Duration (seconds)</label>
                <input
                  type="number"
                  value={duration}
                  onChange={(e) => setDuration(Number(e.target.value))}
                  min={5}
                  max={60}
                />
              </div>

              <div className="config-item">
                <label>FPS</label>
                <select value={fps} onChange={(e) => setFps(Number(e.target.value))}>
                  <option value={30}>30</option>
                  <option value={60}>60</option>
                </select>
              </div>

              <div className="config-item checkbox">
                <label>
                  <input
                    type="checkbox"
                    checked={recordVideo}
                    onChange={(e) => setRecordVideo(e.target.checked)}
                  />
                  Record Video
                </label>
              </div>
            </div>

            <div className="info-box">
              <h4>What you'll get:</h4>
              <ul>
                <li>GPU-accelerated physics simulation</li>
                <li>Photorealistic rendering</li>
                {recordVideo && <li>Recorded video of simulation</li>}
                <li>Performance metrics and analysis</li>
              </ul>

              <div className="cost-estimate">
                <strong>Estimated cost:</strong> ${((duration / 10) * 0.15).toFixed(2)} - ${((duration / 10) * 0.50).toFixed(2)}
                <span className="cost-note">(Modal A10G GPU pricing)</span>
              </div>
            </div>

            <button onClick={runSimulation} className="run-button">
              Run Isaac Lab Simulation
            </button>
          </div>
        )}

        {/* Running State */}
        {status === 'running' && (
          <div className="running-section">
            <div className="progress-container">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
              <p className="progress-text">{progress}% - Running on Modal GPU...</p>
            </div>

            <div className="running-info">
              <p>Your simulation is running on NVIDIA A10G GPU</p>
              <p>Duration: {duration}s | FPS: {fps}</p>
              <p className="running-note">This may take a few minutes...</p>
            </div>
          </div>
        )}

        {/* Completed State */}
        {status === 'completed' && result?.success && (
          <div className="completed-section">
            <div className="success-banner">
              Simulation Completed Successfully!
            </div>

            {result.metrics && (
              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-value">{result.metrics.frames_simulated}</div>
                  <div className="metric-label">Frames Simulated</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value">{result.metrics.average_fps?.toFixed(1)}</div>
                  <div className="metric-label">Average FPS</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value">{result.metrics.physics_steps}</div>
                  <div className="metric-label">Physics Steps</div>
                </div>
              </div>
            )}

            {streamSessionId && (
              <div className="video-section">
                <h3>Recorded Video</h3>
                <IsaacLabVideoPlayer
                  sessionId={streamSessionId}
                  videoSource={videoSource || undefined}
                  onError={(error) => console.error('Video player error:', error)}
                />
              </div>
            )}

            <div className="action-buttons">
              <button onClick={reset} className="secondary-button">
                Run Again
              </button>
              {onClose && (
                <button onClick={onClose} className="primary-button">
                  Close
                </button>
              )}
            </div>
          </div>
        )}

        {/* Error State */}
        {status === 'error' && (
          <div className="error-section">
            <div className="error-banner">
              Simulation Failed
            </div>

            <div className="error-message">
              {result?.error || 'Unknown error occurred'}
            </div>

            <div className="action-buttons">
              <button onClick={reset} className="secondary-button">
                Try Again
              </button>
              {onClose && (
                <button onClick={onClose} className="primary-button">
                  Close
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
