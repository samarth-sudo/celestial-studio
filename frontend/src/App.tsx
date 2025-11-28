import { useState, useEffect } from 'react'
import type { SceneConfig } from './types'
import LandingPage from './components/LandingPage'
import ConversationalChat from './components/ConversationalChat'
import Simulator from './components/Simulator'
import GenesisViewer from './components/GenesisViewer'
import CameraViewTest from './pages/CameraViewTest'
import TeleopTest from './pages/TeleopTest'
import ToastNotification from './components/ToastNotification'
import './App.css'

type RendererType = 'genesis' | 'threejs'

function App() {
  const [showLanding, setShowLanding] = useState(true)
  const [initialPrompt, setInitialPrompt] = useState('')
  const [sceneConfig, setSceneConfig] = useState<SceneConfig | null>(null)
  const [isTestMode, setIsTestMode] = useState(false)
  const [testMode, setTestMode] = useState<string | null>(null)

  // Renderer selection: Genesis (default) or Three.js (fallback)
  const [renderer, setRenderer] = useState<RendererType>(() => {
    // Check localStorage for saved preference
    const saved = localStorage.getItem('celestial-renderer')
    return (saved as RendererType) || 'genesis'
  })

  const [genesisAvailable, setGenesisAvailable] = useState(true)

  // Check URL for test mode
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const testParam = params.get('test')
    if (testParam === 'camera' || testParam === 'teleop') {
      setIsTestMode(true)
      setTestMode(testParam)
    }

    // Check for renderer override in URL
    const rendererParam = params.get('renderer')
    if (rendererParam === 'threejs' || rendererParam === 'genesis') {
      setRenderer(rendererParam as RendererType)
    }
  }, [])

  // Check Genesis availability
  useEffect(() => {
    const checkGenesis = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/genesis/status')
        const data = await response.json()
        setGenesisAvailable(data.available)

        // If Genesis not available, fallback to Three.js
        if (!data.available && renderer === 'genesis') {
          console.warn('⚠️ Genesis not available, falling back to Three.js')
          setRenderer('threejs')
        }
      } catch (error) {
        console.warn('⚠️ Could not check Genesis status, using Three.js')
        setGenesisAvailable(false)
        if (renderer === 'genesis') {
          setRenderer('threejs')
        }
      }
    }

    if (!showLanding) {
      checkGenesis()
    }
  }, [showLanding, renderer])

  // Save renderer preference
  useEffect(() => {
    localStorage.setItem('celestial-renderer', renderer)
  }, [renderer])

  const toggleRenderer = () => {
    setRenderer(prev => {
      if (prev === 'genesis') {
        return 'threejs'
      } else {
        // Only switch to Genesis if it's available
        return genesisAvailable ? 'genesis' : 'threejs'
      }
    })
  }

  // Generate a unique user ID for conversational chat
  const userId = 'user-' + Math.random().toString(36).substr(2, 9)

  const handleStart = (prompt: string) => {
    setInitialPrompt(prompt)
    setShowLanding(false)
  }

  const handleSimulationGenerated = (sceneConfig: SceneConfig) => {
    setSceneConfig(sceneConfig)
    console.log('🎬 Simulation generated:', sceneConfig)
  }

  // Render test mode
  if (isTestMode) {
    if (testMode === 'camera') {
      return <CameraViewTest />
    } else if (testMode === 'teleop') {
      return <TeleopTest />
    }
  }

  if (showLanding) {
    return <LandingPage onStart={handleStart} />
  }

  return (
    <div className="app">
      {/* Global Toast Notifications */}
      <ToastNotification />

      {/* Renderer Toggle */}
      <div className="renderer-toggle">
        <button
          onClick={toggleRenderer}
          className={`toggle-btn ${renderer === 'genesis' ? 'genesis-active' : 'threejs-active'}`}
          title={genesisAvailable ? 'Switch renderer' : 'Genesis not available'}
          disabled={!genesisAvailable && renderer === 'threejs'}
        >
          {renderer === 'genesis' ? '🚀 Genesis' : '🎨 Three.js'}
          {!genesisAvailable && ' (Genesis unavailable)'}
        </button>
        <span className="renderer-badge">
          {renderer === 'genesis' ? 'Python + GPU Physics' : 'Browser Rendering'}
        </span>
      </div>

      <div className="app-content">
        <div className="left-panel">
          <ConversationalChat
            onSimulationGenerated={handleSimulationGenerated}
            userId={userId}
            initialPrompt={initialPrompt}
          />
        </div>

        <div className="right-panel">
          {renderer === 'genesis' ? (
            <GenesisViewer
              backendUrl="http://localhost:8000"
              quality="medium"
              showDebug={true}
              autoStart={true}
            />
          ) : (
            <Simulator
              sceneConfig={sceneConfig}
            />
          )}
        </div>
      </div>
    </div>
  )
}

export default App
