import { useState, useEffect } from 'react'
import type { SceneConfig } from './types'
import LandingPage from './components/LandingPage'
import ConversationalChat from './components/ConversationalChat'
import GenesisViewer from './components/GenesisViewer'
import TeleopTest from './pages/TeleopTest'
import ToastNotification from './components/ToastNotification'
import './App.css'

function App() {
  const [showLanding, setShowLanding] = useState(true)
  const [initialPrompt, setInitialPrompt] = useState('')
  const [sceneConfig, setSceneConfig] = useState<SceneConfig | null>(null)
  const [isTestMode, setIsTestMode] = useState(false)
  const [testMode, setTestMode] = useState<string | null>(null)

  // Check URL for test mode
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    const testParam = params.get('test')
    if (testParam === 'teleop') {
      setIsTestMode(true)
      setTestMode(testParam)
    }
  }, [])

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
  if (isTestMode && testMode === 'teleop') {
    return <TeleopTest />
  }

  if (showLanding) {
    return <LandingPage onStart={handleStart} />
  }

  return (
    <div className="app">
      {/* Global Toast Notifications */}
      <ToastNotification />

      <div className="app-content">
        <div className="left-panel">
          <ConversationalChat
            onSimulationGenerated={handleSimulationGenerated}
            userId={userId}
            initialPrompt={initialPrompt}
          />
        </div>

        <div className="right-panel">
          <GenesisViewer
            backendUrl="http://localhost:8000"
            quality="medium"
            showDebug={true}
            autoStart={true}
          />
        </div>
      </div>
    </div>
  )
}

export default App
