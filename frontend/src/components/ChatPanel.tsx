import { useState, useEffect, useRef } from 'react'
import Editor from '@monaco-editor/react'
import axios from 'axios'
import './ChatPanel.css'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

interface ChatPanelProps {
  onCodeGenerated: (code: string, robotType: 'mobile' | 'arm' | 'drone') => void
  onSimulate: () => void
  hasCode: boolean
  initialPrompt?: string | null
}

const ROBOT_EXAMPLES = [
  { type: 'mobile' as const, text: 'Build a 4-wheel mobile warehouse robot' },
  { type: 'arm' as const, text: 'Create a 6-DOF robotic arm for pick and place' },
  { type: 'drone' as const, text: 'Make a quadcopter drone for inspection' },
]

export default function ChatPanel({ onCodeGenerated, onSimulate, hasCode, initialPrompt }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [generatedCode, setGeneratedCode] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const hasRunInitialPrompt = useRef(false)

  // Auto-submit initial prompt from landing page
  useEffect(() => {
    if (initialPrompt && !hasRunInitialPrompt.current) {
      hasRunInitialPrompt.current = true
      handleGenerate(initialPrompt)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const detectRobotType = (text: string): 'mobile' | 'arm' | 'drone' => {
    const lowerText = text.toLowerCase()
    if (lowerText.includes('mobile') || lowerText.includes('wheel') || lowerText.includes('navigate')) {
      return 'mobile'
    } else if (lowerText.includes('arm') || lowerText.includes('manipulator') || lowerText.includes('pick')) {
      return 'arm'
    } else if (lowerText.includes('drone') || lowerText.includes('quadcopter') || lowerText.includes('fly')) {
      return 'drone'
    }
    return 'mobile'
  }

  const handleGenerate = async (prompt: string) => {
    if (!prompt.trim() || isGenerating) return

    const robotType = detectRobotType(prompt)

    setMessages(prev => [...prev, { role: 'user', content: prompt }])
    setInput('')
    setIsGenerating(true)

    try {
      const response = await axios.post('http://localhost:8000/api/generate', {
        prompt,
        robot_type: robotType
      })

      const code = response.data.code
      setGeneratedCode(code)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Generated ${robotType} robot code! Click "Run Simulation" to see it in action.`
      }])
      onCodeGenerated(code, robotType)
    } catch (error) {
      console.error('Generation error:', error)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, there was an error generating the code. Please try again.'
      }])
    } finally {
      setIsGenerating(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleGenerate(input)
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <h2>👋 Welcome to Robotics Demo</h2>
            <p>Describe the robot you want to build, or try one of these:</p>
            <div className="examples">
              {ROBOT_EXAMPLES.map((example, i) => (
                <button
                  key={i}
                  className="example-btn"
                  onClick={() => handleGenerate(example.text)}
                >
                  {example.text}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div className="message-content">{msg.content}</div>
            </div>
          ))
        )}
        {isGenerating && (
          <div className="message assistant">
            <div className="message-content">Generating code...</div>
          </div>
        )}
      </div>

      {generatedCode && (
        <div className="code-viewer">
          <div className="code-header">
            <span>Generated Code</span>
            <button
              className="simulate-btn"
              onClick={onSimulate}
              disabled={!hasCode}
            >
              ▶ Run Simulation
            </button>
          </div>
          <Editor
            height="300px"
            language="python"
            value={generatedCode}
            theme="vs-dark"
            options={{
              readOnly: true,
              minimap: { enabled: false },
              fontSize: 12,
            }}
          />
        </div>
      )}

      <div className="chat-input">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Describe your robot..."
          disabled={isGenerating}
        />
        <button
          onClick={() => handleGenerate(input)}
          disabled={isGenerating || !input.trim()}
        >
          Generate
        </button>
      </div>
    </div>
  )
}
