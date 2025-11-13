import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import './ConversationalChat.css'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

interface ChatResponse {
  type: 'clarification_needed' | 'simulation_ready' | 'chat'
  message: string
  questions?: Array<{
    text: string
    options: string[]
    field: string
  }>
  simulation?: any
  requirements?: any
}

interface ConversationalChatProps {
  onSimulationGenerated: (sceneConfig: any) => void
  userId: string
  initialPrompt?: string
}

export default function ConversationalChat({ onSimulationGenerated, userId, initialPrompt }: ConversationalChatProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [currentQuestions, setCurrentQuestions] = useState<any[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const hasProcessedInitialPrompt = useRef(false)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Handle initial prompt from landing page
  useEffect(() => {
    if (initialPrompt && !hasProcessedInitialPrompt.current) {
      hasProcessedInitialPrompt.current = true
      handleSend(initialPrompt)
    }
  }, [initialPrompt])

  const handleSend = async (message: string) => {
    if (!message.trim() || isProcessing) return

    // Add user message to chat
    setMessages(prev => [...prev, { role: 'user', content: message }])
    setInput('')
    setIsProcessing(true)
    setCurrentQuestions([])

    try {
      const response = await axios.post<ChatResponse>('http://localhost:8000/api/chat', {
        userId,
        message
      })

      const data = response.data

      // Add assistant message to chat
      setMessages(prev => [...prev, { role: 'assistant', content: data.message }])

      if (data.type === 'clarification_needed') {
        // Store questions for potential UI enhancement
        setCurrentQuestions(data.questions || [])
      } else if (data.type === 'simulation_ready') {
        // Simulation is ready! Pass it to parent
        if (data.simulation) {
          onSimulationGenerated(data.simulation)
        }
      }
    } catch (error) {
      console.error('Chat error:', error)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, there was an error processing your request. Please try again.'
      }])
    } finally {
      setIsProcessing(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend(input)
    }
  }

  const handleQuickResponse = (option: string) => {
    handleSend(option)
  }

  return (
    <div className="conversational-chat">
      <div className="chat-header">
        <h3>🤖 Simulation Generator</h3>
        <p>Describe your robot simulation in natural language</p>
      </div>

      <div className="chat-messages-container">
        {messages.length === 0 ? (
          <div className="welcome-screen">
            <div className="welcome-icon">🎭</div>
            <h2>Create Your Simulation</h2>
            <p>Tell me what kind of robotics simulation you want to create!</p>
            <div className="example-prompts">
              <h4>Try asking:</h4>
              <button
                className="example-prompt"
                onClick={() => handleSend('I want a mobile robot in a warehouse')}
              >
                "I want a mobile robot in a warehouse"
              </button>
              <button
                className="example-prompt"
                onClick={() => handleSend('Create a robotic arm that picks boxes')}
              >
                "Create a robotic arm that picks boxes"
              </button>
              <button
                className="example-prompt"
                onClick={() => handleSend('Make a drone for outdoor inspection')}
              >
                "Make a drone for outdoor inspection"
              </button>
            </div>
          </div>
        ) : (
          <div className="messages-list">
            {messages.map((msg, i) => (
              <div key={i} className={`chat-message ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? '👤' : '🤖'}
                </div>
                <div className="message-bubble">
                  <div className="message-text">{msg.content}</div>
                </div>
              </div>
            ))}
            {isProcessing && (
              <div className="chat-message assistant">
                <div className="message-avatar">🤖</div>
                <div className="message-bubble">
                  <div className="message-text typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {currentQuestions.length > 0 && (
        <div className="quick-responses">
          {currentQuestions[0]?.options?.slice(0, 3).map((option: string, i: number) => (
            <button
              key={i}
              className="quick-response-btn"
              onClick={() => handleQuickResponse(option)}
              disabled={isProcessing}
            >
              {option}
            </button>
          ))}
        </div>
      )}

      <div className="chat-input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Describe your simulation..."
          disabled={isProcessing}
          className="chat-input"
        />
        <button
          onClick={() => handleSend(input)}
          disabled={isProcessing || !input.trim()}
          className="send-button"
        >
          {isProcessing ? '⏳' : '→'}
        </button>
      </div>
    </div>
  )
}
