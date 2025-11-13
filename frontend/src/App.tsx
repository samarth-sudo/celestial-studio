import { useState } from 'react'
import LandingPage from './components/LandingPage'
import ConversationalChat from './components/ConversationalChat'
import Simulator from './components/Simulator'
import './App.css'

function App() {
  const [showLanding, setShowLanding] = useState(true)
  const [initialPrompt, setInitialPrompt] = useState('')
  const [sceneConfig, setSceneConfig] = useState<any>(null)

  // Generate a unique user ID for conversational chat
  const userId = 'user-' + Math.random().toString(36).substr(2, 9)

  const handleStart = (prompt: string) => {
    setInitialPrompt(prompt)
    setShowLanding(false)
  }

  const handleSimulationGenerated = (sceneConfig: any) => {
    setSceneConfig(sceneConfig)
    console.log('🎬 Simulation generated:', sceneConfig)
  }

  if (showLanding) {
    return <LandingPage onStart={handleStart} />
  }

  return (
    <div className="app">
      <div className="app-content">
        <div className="left-panel">
          <ConversationalChat
            onSimulationGenerated={handleSimulationGenerated}
            userId={userId}
            initialPrompt={initialPrompt}
          />
        </div>

        <div className="right-panel">
          <Simulator
            sceneConfig={sceneConfig}
          />
        </div>
      </div>
    </div>
  )
}

export default App
