import { useState, useEffect } from 'react'
import './LandingPage.css'

interface LandingPageProps {
  onStart: (prompt: string) => void
}

const EXAMPLE_PROMPTS = [
  "I want a mobile robot in a warehouse",
  "Create a robotic arm that picks boxes",
  "Make a drone for outdoor inspection"
]

const REVOLVING_WORDS = ['Chat','Simulate','Build']

export default function LandingPage({ onStart }: LandingPageProps) {
  const [input, setInput] = useState('')
  const [showSplitView, setShowSplitView] = useState(false)
  const [leftPanelWidth, setLeftPanelWidth] = useState(50) // percentage
  const [isDragging, setIsDragging] = useState(false)
  const [currentWordIndex, setCurrentWordIndex] = useState(0)

  // Revolving text animation
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentWordIndex((prev) => (prev + 1) % REVOLVING_WORDS.length)
    }, 2000) // Change word every 2 seconds

    return () => clearInterval(interval)
  }, [])

  const handleMouseDown = () => {
    setIsDragging(true)
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return

    const container = e.currentTarget as HTMLElement
    const containerRect = container.getBoundingClientRect()
    const newWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100

    // Constrain between 20% and 80%
    if (newWidth >= 20 && newWidth <= 80) {
      setLeftPanelWidth(newWidth)
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim()) {
      setShowSplitView(true)
      // Small delay to show split view before transitioning
      setTimeout(() => {
        onStart(input.trim())
      }, 100)
    }
  }

  const handleExampleClick = (prompt: string) => {
    onStart(prompt)
  }

  // Initial Hero View with Dusk Gradient
  if (!showSplitView) {
    return (
      <div className="landing-page">
        <div className="landing-hero">
          <div className="hero-content">
            <h1 className="hero-headline">
              <span className="revolving-word">{REVOLVING_WORDS[currentWordIndex]}</span> with AI
            </h1>
            <form onSubmit={handleSubmit} className="hero-form">
              <div className="input-container">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Describe your simulation..."
                  className="landing-input"
                />
                <button
                  type="submit"
                  className="landing-submit-btn"
                  disabled={!input.trim()}
                >
                  →
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    )
  }

  // Split Panel View
  return (
    <div className="landing-page-modern">
      <div
        className="landing-split-layout"
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: isDragging ? 'col-resize' : 'default' }}
      >
        {/* Left Panel - Simulation Generator */}
        <div
          className="landing-left-panel"
          style={{ width: `${leftPanelWidth}%` }}
        >
          <div className="panel-header">
            <div className="header-text">
              <h2>Celestial Studio</h2>
            </div>
          </div>

          <div className="panel-content">
            <h1 className="create-heading">Create Your Simulation</h1>
            <p className="create-subtext">
              Tell me what kind of robotics simulation you want to create!
            </p>

            <div className="try-asking-section">
              <h3>TRY ASKING:</h3>
              <div className="example-buttons">
                {EXAMPLE_PROMPTS.map((prompt, index) => (
                  <button
                    key={index}
                    className="example-button"
                    onClick={() => handleExampleClick(prompt)}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="panel-input-form">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Describe your simulation..."
              className="panel-input"
              autoFocus
            />
            <button
              type="submit"
              className="panel-submit-btn"
              disabled={!input.trim()}
            >
              →
            </button>
          </form>
        </div>

        {/* Resize Handle */}
        <div
          className="resize-handle"
          onMouseDown={handleMouseDown}
        />

        {/* Right Panel - 3D Simulator Preview */}
        <div
          className="landing-right-panel"
          style={{ width: `${100 - leftPanelWidth}%` }}
        >
          <div className="simulator-preview">
            <h2>3D Simulator</h2>
            <p>Generate a robot or create a simulation to see it in action!</p>
          </div>
        </div>
      </div>
    </div>
  )
}
