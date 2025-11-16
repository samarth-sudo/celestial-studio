import { useState, useEffect, useRef, useCallback } from 'react'
import axios from 'axios'
import type { SceneConfig } from '../types'
import config from '../config'
import './ConversationalChat.css'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

interface ChatQuestion {
  text: string
  options: string[]
  field: string
}

interface ChatResponse {
  type: 'clarification_needed' | 'simulation_ready' | 'chat' | 'export_ready'
  message: string
  questions?: ChatQuestion[]
  simulation?: SceneConfig
  requirements?: Record<string, unknown>
  export_data?: {
    filename: string
    format: string
    download_url: string
  }
}

interface ConversationalChatProps {
  onSimulationGenerated: (sceneConfig: SceneConfig) => void
  userId: string
  initialPrompt?: string
}

export default function ConversationalChat({ onSimulationGenerated, userId, initialPrompt }: ConversationalChatProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [currentQuestions, setCurrentQuestions] = useState<ChatQuestion[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const hasProcessedInitialPrompt = useRef(false)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Type adapter to convert backend scene config to frontend SceneConfig type
  const adaptSceneConfig = (backendConfig: any): SceneConfig => {
    console.log('📦 Adapting backend scene config:', JSON.stringify(backendConfig, null, 2))

    try {
      // Backend sends robot with: {type, model, dimensions, position, wheels, hasGripper, ...}
      // Frontend expects: {type, name?, links?, joints?, kinematic_tree?, base_link?, urdf?}

      const adaptedConfig: SceneConfig = {
        environment: backendConfig.environment || {},
        robot: {
          type: backendConfig.robot?.type || 'mobile_robot',
          name: backendConfig.robot?.model || backendConfig.robot?.type || 'robot',
          // Keep the backend fields for compatibility with Simulator component
          ...(backendConfig.robot || {})
        },
        objects: backendConfig.objects || [],
        lighting: backendConfig.lighting || {},
        camera: backendConfig.camera || { position: [10, 10, 10], lookAt: [0, 0, 0], fov: 50 }
      }

      console.log('✅ Adapted scene config:', JSON.stringify(adaptedConfig, null, 2))
      return adaptedConfig
    } catch (error) {
      console.error('❌ Error adapting scene config:', error)
      throw error
    }
  }

  const handleSend = useCallback(async (message: string) => {
    if (!message.trim() || isProcessing) return

    // Add user message to chat
    setMessages(prev => [...prev, { role: 'user', content: message }])
    setInput('')
    setIsProcessing(true)
    setCurrentQuestions([])

    try {
      console.log('📤 Sending message to backend:', message)

      const response = await axios.post<ChatResponse>(`${config.backendUrl}/api/chat`, {
        userId,
        message
      })

      const data = response.data
      console.log('📥 Received response:', {
        type: data.type,
        hasSimulation: !!data.simulation,
        message: data.message.substring(0, 100) + '...'
      })

      // Add assistant message to chat
      setMessages(prev => [...prev, { role: 'assistant', content: data.message }])

      if (data.type === 'clarification_needed') {
        // Store questions for potential UI enhancement
        setCurrentQuestions(data.questions || [])
      } else if (data.type === 'simulation_ready') {
        // Simulation is ready! Adapt and pass it to parent
        if (data.simulation) {
          console.log('🎯 Processing simulation_ready response')
          try {
            const adaptedConfig = adaptSceneConfig(data.simulation)
            console.log('🚀 Calling onSimulationGenerated with adapted config')
            onSimulationGenerated(adaptedConfig)
            console.log('✅ Simulation generation callback completed')

            // Sync the scene with backend conversation context
            console.log('🔄 Syncing scene with backend context')
            await axios.post(`${config.backendUrl}/api/chat/sync-scene`, {
              userId,
              sceneConfig: data.simulation
            })
            console.log('✅ Scene synced with backend context')
          } catch (adaptError) {
            console.error('❌ Error in simulation adaptation/generation:', adaptError)
            if (adaptError instanceof Error) {
              console.error('Error message:', adaptError.message)
              console.error('Error stack:', adaptError.stack)
            }
            throw adaptError
          }
        } else {
          console.warn('⚠️ simulation_ready response but no simulation data provided')
        }
      } else if (data.type === 'export_ready') {
        // Export is ready! Trigger download
        if (data.export_data?.download_url) {
          await triggerDownload(data.export_data.download_url, data.export_data.filename)
        }
      }
    } catch (error) {
      console.error('❌ Chat error:', error)
      if (error instanceof Error) {
        console.error('Error message:', error.message)
        console.error('Error stack:', error.stack)
      }
      if (axios.isAxiosError(error)) {
        console.error('Axios error details:', {
          status: error.response?.status,
          statusText: error.response?.statusText,
          data: error.response?.data
        })
      }
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, there was an error processing your request. Please try again.'
      }])
    } finally {
      setIsProcessing(false)
    }
  }, [isProcessing, userId, onSimulationGenerated])

  // Handle initial prompt from landing page
  useEffect(() => {
    if (initialPrompt && !hasProcessedInitialPrompt.current) {
      hasProcessedInitialPrompt.current = true
      handleSend(initialPrompt)
    }
  }, [initialPrompt, handleSend])

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend(input)
    }
  }

  const handleQuickResponse = (option: string) => {
    handleSend(option)
  }

  const triggerDownload = async (downloadUrl: string, filename: string) => {
    try {
      console.log(`📦 Downloading: ${filename}`)

      // Fetch the file as blob
      const response = await axios.get(`${config.backendUrl}${downloadUrl}`, {
        responseType: 'blob'
      })

      // Create blob URL and trigger download
      const blob = new Blob([response.data], { type: 'application/zip' })
      const url = window.URL.createObjectURL(blob)

      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()

      // Cleanup
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)

      console.log(`✅ Download complete: ${filename}`)
    } catch (error) {
      console.error('Download error:', error)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '❌ Sorry, there was an error downloading the file. Please try again.'
      }])
    }
  }

  return (
    <div className="conversational-chat">
      <div className="chat-header">
        <h3>Celestial Studio</h3>
        
      </div>

      <div className="chat-messages-container">
        {messages.length === 0 ? (
          <div className="welcome-screen">
            <h2>Create Your Simulation</h2>
            <p>Tell me what kind of robotics simulation you want to create</p>

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
              <button
                className="example-prompt"
                onClick={() => handleSend('Build a custom 2-DOF robot arm')}
              >
                "Build a custom 2-DOF robot arm"
              </button>
              <button
                className="example-prompt"
                onClick={() => handleSend('Add multiple robots to my scene')}
              >
                "Add multiple robots to my scene"
              </button>
            </div>
          </div>
        ) : (
          <div className="messages-list">
            {messages.map((msg, i) => (
              <div key={i} className={`chat-message ${msg.role}`}>
                <div className="message-avatar">
                  {msg.role === 'user' ? 'U' : 'AI'}
                </div>
                <div className="message-bubble">
                  <div className="message-text">{msg.content}</div>
                </div>
              </div>
            ))}
            {isProcessing && (
              <div className="chat-message assistant">
                <div className="message-avatar">AI</div>
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
          {isProcessing ? '...' : 'Send'}
        </button>
      </div>
    </div>
  )
}
