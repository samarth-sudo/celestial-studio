import { useState, useEffect } from 'react'
import GenesisViewer from './GenesisViewer'
import SceneEditor, { type EditorTool } from './SceneEditor'
import AlgorithmControls from './AlgorithmControls'
import IsaacSimulationPanel from './IsaacSimulationPanel'
import './Simulator.css'

type SimulationMode = 'genesis' | 'isaac'

interface SimulatorProps {
  sceneConfig?: any
  onSceneChange?: (sceneConfig: any) => void
}

export default function Simulator({ sceneConfig, onSceneChange }: SimulatorProps) {
  const [mode, setMode] = useState<SimulationMode>('genesis')
  const [showIsaacPanel, setShowIsaacPanel] = useState(false)
  const [isSimulationRunning, setIsSimulationRunning] = useState(false)
  const [genesisWsUrl, setGenesisWsUrl] = useState<string>('')
  const [showAlgorithmControls, setShowAlgorithmControls] = useState(false)

  // Genesis simulation state
  const [genesisSceneId, setGenesisSceneId] = useState<string | null>(null)

  // Determine if we have content to render
  const hasContent = sceneConfig !== undefined

  const handleModeSwitch = (newMode: SimulationMode) => {
    setMode(newMode)
    if (newMode === 'isaac') {
      setShowIsaacPanel(true)
    }
  }

  const handleStartGenesisSimulation = async () => {
    try {
      // Initialize Genesis scene
      const response = await fetch('http://localhost:8000/api/genesis/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          backend: 'auto',  // Auto-detect Metal/CUDA/CPU
          show_viewer: false
        })
      })

      const data = await response.json()

      if (data.status === 'initialized') {
        // Add robot based on scene config
        const robotType = sceneConfig?.robot?.type || 'mobile_robot'
        const robotMap = {
          'mobile_robot': 'mobile',
          'robotic_arm': 'arm',
          'quadcopter': 'drone'
        }

        const addRobotResponse = await fetch('http://localhost:8000/api/genesis/robot/add', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            robot_type: robotMap[robotType as keyof typeof robotMap] || 'mobile',
            position: [0, 0, 1],
            robot_id: 'robot-1'
          })
        })

        const robotData = await addRobotResponse.json()

        // Build the scene
        const buildResponse = await fetch('http://localhost:8000/api/genesis/scene/build', {
          method: 'POST'
        })

        // Start simulation
        const controlResponse = await fetch('http://localhost:8000/api/genesis/control', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'start'
          })
        })

        // Set WebSocket URL for Genesis viewer
        setGenesisWsUrl('ws://localhost:8000/api/genesis/ws')
        setIsSimulationRunning(true)
        console.log('✅ Genesis simulation started')
      }
    } catch (error) {
      console.error('❌ Failed to start Genesis simulation:', error)
    }
  }

  const handleStopGenesisSimulation = async () => {
    try {
      await fetch('http://localhost:8000/api/genesis/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'stop'
        })
      })

      setIsSimulationRunning(false)
      setGenesisWsUrl('')
      console.log('🛑 Genesis simulation stopped')
    } catch (error) {
      console.error('❌ Failed to stop simulation:', error)
    }
  }

  const handleResetGenesisSimulation = async () => {
    try {
      await fetch('http://localhost:8000/api/genesis/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'reset'
        })
      })

      console.log('🔄 Genesis simulation reset')
    } catch (error) {
      console.error('❌ Failed to reset simulation:', error)
    }
  }

  return (
    <div className="simulator">
      {/* Mode Switcher */}
      {hasContent && (
        <div className="mode-switcher">
          <button
            className={`mode-button ${mode === 'genesis' ? 'active' : ''}`}
            onClick={() => handleModeSwitch('genesis')}
            title="GPU-accelerated Genesis simulation (physics-accurate)"
          >
            <span className="mode-icon">🎯</span>
            Genesis Sim
          </button>
          <button
            className={`mode-button ${mode === 'isaac' ? 'active' : ''}`}
            onClick={() => handleModeSwitch('isaac')}
            title="GPU-accelerated Isaac Lab simulation (RL training)"
          >
            <span className="mode-icon">🤖</span>
            Isaac Lab
          </button>
        </div>
      )}

      {/* Simulation Controls for Genesis */}
      {hasContent && mode === 'genesis' && (
        <div className="playback-controls">
          {!isSimulationRunning ? (
            <button
              className="playback-button"
              onClick={handleStartGenesisSimulation}
              title="Start Genesis simulation"
            >
              ▶️ Start Simulation
            </button>
          ) : (
            <>
              <button
                className="playback-button"
                onClick={handleStopGenesisSimulation}
                title="Stop simulation"
              >
                ⏹️ Stop
              </button>
              <button
                className="playback-button"
                onClick={handleResetGenesisSimulation}
                title="Reset simulation"
              >
                🔄 Reset
              </button>
            </>
          )}
          <button
            className="playback-button"
            onClick={() => setShowAlgorithmControls(!showAlgorithmControls)}
            title="Generate or manage algorithms"
          >
            {showAlgorithmControls ? '✖ Close Algorithms' : '⚡ Algorithms'}
          </button>
        </div>
      )}

      {/* Algorithm Controls Panel */}
      {showAlgorithmControls && sceneConfig && (
        <div className="algorithm-controls-overlay">
          <div className="algorithm-controls-panel">
            <div className="panel-header">
              <h3>Algorithm Generator</h3>
              <button className="close-btn" onClick={() => setShowAlgorithmControls(false)}>
                ✖
              </button>
            </div>
            <AlgorithmControls
              robotId="robot-1"
              robotType={sceneConfig.robot?.type === 'mobile_robot' ? 'mobile' : sceneConfig.robot?.type === 'robotic_arm' ? 'arm' : 'drone'}
              onAlgorithmApplied={(algorithm) => {
                console.log('Algorithm applied:', algorithm)
              }}
            />
          </div>
        </div>
      )}

      {!hasContent ? (
        <div className="simulator-empty">
          <div className="empty-message">
            <h2>🎮 Genesis Simulator</h2>
            <p>Generate a robot simulation to see Genesis in action!</p>
            <p className="tech-note">
              <strong>Powered by Genesis Engine:</strong> GPU-accelerated physics simulation
              with 1000x faster rendering than client-side engines.
            </p>
          </div>
        </div>
      ) : mode === 'isaac' && showIsaacPanel ? (
        <IsaacSimulationPanel
          sceneConfig={sceneConfig}
          onClose={() => setShowIsaacPanel(false)}
        />
      ) : mode === 'genesis' ? (
        <div className="genesis-container">
          {isSimulationRunning && genesisWsUrl ? (
            <GenesisViewer wsUrl={genesisWsUrl} />
          ) : (
            <div className="genesis-placeholder">
              <div className="placeholder-content">
                <h3>🚀 Ready to Simulate</h3>
                <p>Click "Start Simulation" to run your robot in Genesis</p>
                <div className="robot-info">
                  <strong>Robot Type:</strong> {sceneConfig?.robot?.type || 'mobile_robot'}
                </div>
              </div>
            </div>
          )}
        </div>
      ) : null}
    </div>
  )
}
