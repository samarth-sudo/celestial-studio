import { useEffect, useRef, useState, useCallback } from 'react'
import * as THREE from 'three'
import axios from 'axios'
import config from '../config'
import './CameraView.css'

interface Detection {
  label: string
  confidence: number
  bbox: { x: number; y: number; width: number; height: number }
  position3D?: THREE.Vector3
  distance?: number
}

interface TrackingTrail {
  positions: { x: number; y: number; timestamp: number }[]
  label: string
  color: string
}

interface VisionAnalysisResponse {
  description: string
  objects_identified: Array<{
    label: string
    confidence: number
    description: string
    position_description: string
  }>
  scene_understanding: string
  spatial_relationships: string[]
  model_used: string
}

interface CameraViewProps {
  scene: THREE.Scene
  robotPosition?: THREE.Vector3
  robotRotation?: THREE.Euler
  objects: Array<{
    position: THREE.Vector3
    label: string
    radius: number
    color?: string
  }>
  onDetectionsUpdate?: (detections: Detection[]) => void
}

export default function CameraView({
  scene,
  robotPosition = new THREE.Vector3(0, 1, 0),
  robotRotation = new THREE.Euler(0, 0, 0),
  objects,
  onDetectionsUpdate
}: CameraViewProps) {
  // Refs for Three.js rendering
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const renderTargetRef = useRef<THREE.WebGLRenderTarget | null>(null)
  const animationFrameRef = useRef<number | null>(null)

  // State for detections and vision analysis
  const [detections, setDetections] = useState<Detection[]>([])
  const [trackingTrails, setTrackingTrails] = useState<Map<string, TrackingTrail>>(new Map())
  const [visionAnalysis, setVisionAnalysis] = useState<VisionAnalysisResponse | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [fps, setFps] = useState(0)
  const [detectionCount, setDetectionCount] = useState(0)
  const [avgConfidence, setAvgConfidence] = useState(0)

  // Frame counter for throttling vision API calls
  const frameCountRef = useRef(0)
  const lastAnalysisTimeRef = useRef(0)
  const fpsCounterRef = useRef({ frames: 0, lastTime: Date.now() })

  // Initialize Three.js camera and renderer
  useEffect(() => {
    if (!containerRef.current || !canvasRef.current) return

    // Create camera (robot's perspective)
    const camera = new THREE.PerspectiveCamera(
      75, // FOV
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    )
    cameraRef.current = camera

    // Create renderer
    const renderer = new THREE.WebGLRenderer({
      canvas: canvasRef.current,
      antialias: true,
      alpha: false
    })
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight)
    renderer.setPixelRatio(window.devicePixelRatio)
    rendererRef.current = renderer

    // Create render target for capturing frames
    const renderTarget = new THREE.WebGLRenderTarget(
      containerRef.current.clientWidth,
      containerRef.current.clientHeight,
      {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format: THREE.RGBAFormat
      }
    )
    renderTargetRef.current = renderTarget

    // Handle window resize
    const handleResize = () => {
      if (!containerRef.current || !camera || !renderer || !renderTarget) return

      const width = containerRef.current.clientWidth
      const height = containerRef.current.clientHeight

      camera.aspect = width / height
      camera.updateProjectionMatrix()

      renderer.setSize(width, height)
      renderTarget.setSize(width, height)

      if (overlayCanvasRef.current) {
        overlayCanvasRef.current.width = width
        overlayCanvasRef.current.height = height
      }
    }

    window.addEventListener('resize', handleResize)
    handleResize()

    return () => {
      window.removeEventListener('resize', handleResize)
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      renderer.dispose()
      renderTarget.dispose()
    }
  }, [])

  // Update camera position/rotation to match robot
  useEffect(() => {
    if (!cameraRef.current) return

    const camera = cameraRef.current
    camera.position.copy(robotPosition)
    camera.rotation.copy(robotRotation)
    camera.updateMatrixWorld()
  }, [robotPosition, robotRotation])

  // Generate detections from objects
  const generateDetections = useCallback((): Detection[] => {
    if (!cameraRef.current) return []

    const camera = cameraRef.current
    const detections: Detection[] = []

    objects.forEach(obj => {
      // Project 3D position to 2D screen space
      const position = obj.position.clone()
      const distance = camera.position.distanceTo(position)

      // Check if object is in front of camera
      const cameraDirection = new THREE.Vector3()
      camera.getWorldDirection(cameraDirection)
      const toObject = position.clone().sub(camera.position).normalize()
      const dot = cameraDirection.dot(toObject)

      if (dot > 0.3 && distance < 50) { // Object is in view
        // Project to screen coordinates
        const projected = position.clone().project(camera)

        // Convert to canvas coordinates
        const canvas = canvasRef.current
        if (!canvas) return

        const x = ((projected.x + 1) / 2) * canvas.width
        const y = ((-projected.y + 1) / 2) * canvas.height

        // Estimate bbox size based on object radius and distance
        const apparentSize = (obj.radius / distance) * 500
        const width = Math.max(20, Math.min(200, apparentSize))
        const height = width

        // Calculate confidence based on distance and apparent size
        const distanceConfidence = Math.max(0.3, 1 - distance / 50)
        const sizeConfidence = Math.min(1, apparentSize / 50)
        const confidence = (distanceConfidence + sizeConfidence) / 2

        detections.push({
          label: obj.label || 'object',
          confidence: Math.round(confidence * 100) / 100,
          bbox: {
            x: x - width / 2,
            y: y - height / 2,
            width,
            height
          },
          position3D: position,
          distance: Math.round(distance * 10) / 10
        })
      }
    })

    return detections
  }, [objects])

  // Update tracking trails
  const updateTrackingTrails = useCallback((currentDetections: Detection[]) => {
    const now = Date.now()
    const newTrails = new Map(trackingTrails)

    currentDetections.forEach(detection => {
      const key = detection.label
      const trail = newTrails.get(key)

      if (trail) {
        // Add new position to existing trail
        trail.positions.push({
          x: detection.bbox.x + detection.bbox.width / 2,
          y: detection.bbox.y + detection.bbox.height / 2,
          timestamp: now
        })

        // Keep only last 30 positions (2 seconds at 15 FPS)
        if (trail.positions.length > 30) {
          trail.positions = trail.positions.slice(-30)
        }
      } else {
        // Create new trail
        const color = `hsl(${Math.random() * 360}, 70%, 50%)`
        newTrails.set(key, {
          positions: [{
            x: detection.bbox.x + detection.bbox.width / 2,
            y: detection.bbox.y + detection.bbox.height / 2,
            timestamp: now
          }],
          label: key,
          color
        })
      }
    })

    // Remove old trails (no update for 3 seconds)
    const trailTimeout = 3000
    newTrails.forEach((trail, key) => {
      const lastPosition = trail.positions[trail.positions.length - 1]
      if (now - lastPosition.timestamp > trailTimeout) {
        newTrails.delete(key)
      }
    })

    setTrackingTrails(newTrails)
  }, [trackingTrails])

  // Draw detection overlays on canvas
  const drawDetectionOverlay = useCallback((detections: Detection[]) => {
    const canvas = overlayCanvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw tracking trails first (behind bboxes)
    trackingTrails.forEach(trail => {
      if (trail.positions.length < 2) return

      ctx.beginPath()
      ctx.strokeStyle = trail.color
      ctx.lineWidth = 2

      trail.positions.forEach((pos, i) => {
        if (i === 0) {
          ctx.moveTo(pos.x, pos.y)
        } else {
          ctx.lineTo(pos.x, pos.y)
        }
      })

      ctx.stroke()

      // Draw trail points
      trail.positions.forEach((pos, i) => {
        const alpha = i / trail.positions.length
        ctx.fillStyle = trail.color.replace(')', `, ${alpha})`)
        ctx.beginPath()
        ctx.arc(pos.x, pos.y, 3, 0, Math.PI * 2)
        ctx.fill()
      })
    })

    // Draw bounding boxes and labels
    detections.forEach(detection => {
      const { bbox, label, confidence, distance } = detection

      // Color based on confidence
      let color = '#00ff00' // Green for high confidence
      if (confidence < 0.7) color = '#ffff00' // Yellow for medium
      if (confidence < 0.5) color = '#ff0000' // Red for low

      // Draw bounding box
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height)

      // Draw label background
      const labelText = `${label} (${(confidence * 100).toFixed(0)}%)`
      const distanceText = distance ? `${distance}m` : ''
      ctx.font = '14px monospace'
      const labelWidth = Math.max(
        ctx.measureText(labelText).width,
        ctx.measureText(distanceText).width
      ) + 8

      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
      ctx.fillRect(bbox.x, bbox.y - 40, labelWidth, 38)

      // Draw label text
      ctx.fillStyle = color
      ctx.fillText(labelText, bbox.x + 4, bbox.y - 22)

      if (distance) {
        ctx.fillStyle = '#ffffff'
        ctx.fillText(distanceText, bbox.x + 4, bbox.y - 6)
      }

      // Draw confidence bar
      const barWidth = bbox.width
      const barHeight = 4
      ctx.fillStyle = 'rgba(0, 0, 0, 0.5)'
      ctx.fillRect(bbox.x, bbox.y + bbox.height + 2, barWidth, barHeight)
      ctx.fillStyle = color
      ctx.fillRect(bbox.x, bbox.y + bbox.height + 2, barWidth * confidence, barHeight)
    })
  }, [trackingTrails])

  // Call vision API for scene understanding
  const analyzeWithVision = useCallback(async () => {
    if (!canvasRef.current || isAnalyzing) return

    const now = Date.now()
    // Throttle to max 1 call every 5 seconds
    if (now - lastAnalysisTimeRef.current < 5000) return

    setIsAnalyzing(true)
    lastAnalysisTimeRef.current = now

    try {
      // Capture current frame as base64
      const canvas = canvasRef.current
      const imageData = canvas.toDataURL('image/png')

      // Call vision analysis API
      const response = await axios.post<VisionAnalysisResponse>(
        `${config.backendUrl}/api/vision/analyze`,
        {
          image_data: imageData,
          prompt: 'Describe the robotics simulation scene. Identify all objects, their positions, and spatial relationships. Focus on objects that would be relevant for robot navigation and interaction.'
        }
      )

      setVisionAnalysis(response.data)
      console.log('Vision analysis:', response.data)
    } catch (error) {
      console.error('Vision analysis error:', error)
    } finally {
      setIsAnalyzing(false)
    }
  }, [isAnalyzing])

  // Main render loop
  useEffect(() => {
    const animate = () => {
      if (!rendererRef.current || !cameraRef.current || !renderTargetRef.current) {
        animationFrameRef.current = requestAnimationFrame(animate)
        return
      }

      const renderer = rendererRef.current
      const camera = cameraRef.current

      // Render scene to canvas
      renderer.render(scene, camera)

      // Generate detections from current view
      const currentDetections = generateDetections()
      setDetections(currentDetections)
      setDetectionCount(currentDetections.length)

      // Calculate average confidence
      if (currentDetections.length > 0) {
        const avg = currentDetections.reduce((sum, d) => sum + d.confidence, 0) / currentDetections.length
        setAvgConfidence(Math.round(avg * 100) / 100)
      } else {
        setAvgConfidence(0)
      }

      // Update tracking trails
      updateTrackingTrails(currentDetections)

      // Draw detection overlay
      drawDetectionOverlay(currentDetections)

      // Notify parent of detections
      if (onDetectionsUpdate) {
        onDetectionsUpdate(currentDetections)
      }

      // Calculate FPS
      fpsCounterRef.current.frames++
      const now = Date.now()
      if (now - fpsCounterRef.current.lastTime >= 1000) {
        setFps(fpsCounterRef.current.frames)
        fpsCounterRef.current.frames = 0
        fpsCounterRef.current.lastTime = now
      }

      // Call vision API every 90 frames (~5 seconds at 15 FPS)
      frameCountRef.current++
      if (frameCountRef.current >= 90) {
        analyzeWithVision()
        frameCountRef.current = 0
      }

      animationFrameRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [scene, generateDetections, updateTrackingTrails, drawDetectionOverlay, analyzeWithVision, onDetectionsUpdate])

  return (
    <div className="camera-view" ref={containerRef}>
      <canvas ref={canvasRef} className="camera-canvas" />
      <canvas ref={overlayCanvasRef} className="overlay-canvas" />

      {/* Statistics Panel */}
      <div className="camera-stats">
        <div className="stat">
          <span className="stat-label">FPS:</span>
          <span className="stat-value">{fps}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Detections:</span>
          <span className="stat-value">{detectionCount}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Avg Conf:</span>
          <span className="stat-value">{(avgConfidence * 100).toFixed(0)}%</span>
        </div>
        {isAnalyzing && (
          <div className="stat analyzing">
            <span className="stat-label">🔍 Analyzing...</span>
          </div>
        )}
      </div>

      {/* Vision Analysis Panel */}
      {visionAnalysis && (
        <div className="vision-analysis-panel">
          <h4>AI Vision Analysis ({visionAnalysis.model_used})</h4>
          <div className="analysis-content">
            <p className="scene-description">{visionAnalysis.scene_understanding}</p>

            {visionAnalysis.objects_identified.length > 0 && (
              <div className="objects-list">
                <strong>Detected:</strong>
                {visionAnalysis.objects_identified.map((obj, i) => (
                  <span key={i} className="object-tag">
                    {obj.label} ({(obj.confidence * 100).toFixed(0)}%)
                  </span>
                ))}
              </div>
            )}

            {visionAnalysis.spatial_relationships.length > 0 && (
              <div className="spatial-info">
                <strong>Spatial:</strong>
                <ul>
                  {visionAnalysis.spatial_relationships.map((rel, i) => (
                    <li key={i}>{rel}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
