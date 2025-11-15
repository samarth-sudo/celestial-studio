import { useState, useEffect, useRef, useCallback } from 'react'
import axios from 'axios'
import './IsaacLabVideoPlayer.css'

interface IsaacLabVideoPlayerProps {
  sessionId?: string
  videoSource?: string
  onConnectionStateChange?: (state: string) => void
  onError?: (error: string) => void
}

export default function IsaacLabVideoPlayer({
  sessionId,
  videoSource,
  onConnectionStateChange,
  onError
}: IsaacLabVideoPlayerProps) {
  const [connectionState, setConnectionState] = useState<string>('disconnected')
  const [error, setError] = useState<string | null>(null)
  const [isConnecting, setIsConnecting] = useState(false)

  const videoRef = useRef<HTMLVideoElement>(null)
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null)

  const disconnect = useCallback(() => {
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close()
      peerConnectionRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    setConnectionState('disconnected')
  }, [])

  const connectWebRTC = useCallback(async () => {
    if (!sessionId) {
      const err = 'No session ID provided'
      setError(err)
      onError?.(err)
      return
    }

    setIsConnecting(true)
    setError(null)

    try {
      console.log('Connecting to WebRTC session:', sessionId)

      // Create peer connection
      const pc = new RTCPeerConnection({
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' }
        ]
      })

      peerConnectionRef.current = pc

      // Handle connection state changes
      pc.onconnectionstatechange = () => {
        console.log('Connection state:', pc.connectionState)
        setConnectionState(pc.connectionState)
        onConnectionStateChange?.(pc.connectionState)

        if (pc.connectionState === 'failed') {
          setError('WebRTC connection failed')
          onError?.('WebRTC connection failed')
        }
      }

      // Handle ICE connection state
      pc.oniceconnectionstatechange = () => {
        console.log('ICE connection state:', pc.iceConnectionState)
      }

      // Handle incoming tracks (video stream)
      pc.ontrack = (event) => {
        console.log('Received remote track:', event.track.kind)

        if (event.track.kind === 'video' && videoRef.current) {
          videoRef.current.srcObject = event.streams[0]
          console.log('Video stream attached to video element')
        }
      }

      // Add transceiver for receiving video
      pc.addTransceiver('video', { direction: 'recvonly' })

      // Create offer
      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)

      console.log('Created offer, sending to backend...')

      // Send offer to backend
      const response = await axios.post('http://localhost:8000/api/webrtc/offer', {
        session_id: sessionId,
        offer: {
          sdp: pc.localDescription?.sdp,
          type: pc.localDescription?.type
        },
        video_source: videoSource
      })

      if (!response.data.success) {
        throw new Error(response.data.error || 'Failed to handle offer')
      }

      // Set remote description (answer from server)
      const answer = response.data.answer
      await pc.setRemoteDescription(new RTCSessionDescription(answer))

      console.log('WebRTC connection established')
      setIsConnecting(false)

    } catch (err: any) {
      console.error('WebRTC connection error:', err)
      const errorMsg = err.response?.data?.detail || err.message || 'Connection failed'
      setError(errorMsg)
      onError?.(errorMsg)
      setIsConnecting(false)
      setConnectionState('failed')
    }
  }, [sessionId, videoSource, onError, onConnectionStateChange])

  useEffect(() => {
    if (sessionId && !isConnecting) {
      connectWebRTC()
    }

    return () => {
      disconnect()
    }
  }, [sessionId, isConnecting, connectWebRTC, disconnect])

  const retry = () => {
    disconnect()
    setError(null)
    if (sessionId) {
      connectWebRTC()
    }
  }

  return (
    <div className="isaac-video-player">
      <div className="video-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="isaac-video"
        />

        {isConnecting && (
          <div className="video-overlay">
            <div className="loading-spinner" />
            <p>Connecting to Isaac Lab stream...</p>
          </div>
        )}

        {error && (
          <div className="video-overlay error">
            <div className="error-content">
              <h3>Connection Error</h3>
              <p>{error}</p>
              <button onClick={retry} className="retry-button">
                Retry Connection
              </button>
            </div>
          </div>
        )}

        {connectionState === 'disconnected' && !isConnecting && !error && (
          <div className="video-overlay">
            <p>No active stream</p>
          </div>
        )}
      </div>

      <div className="connection-status">
        <div className={`status-indicator ${connectionState}`} />
        <span className="status-text">
          {connectionState === 'connected' && 'Streaming'}
          {connectionState === 'connecting' && 'Connecting...'}
          {connectionState === 'disconnected' && 'Disconnected'}
          {connectionState === 'failed' && 'Failed'}
        </span>
      </div>
    </div>
  )
}
