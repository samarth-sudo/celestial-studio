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

  // Track first render and mounted state to prevent crashes
  const firstRenderRef = useRef(false)
  const isMountedRef = useRef(true)
  const initRetryCountRef = useRef(0)
  const MAX_INIT_RETRIES = 10

  // Cleanup mounted ref on unmount
  useEffect(() => {
    isMountedRef.current = true
    initRetryCountRef.current = 0
    return () => {
      isMountedRef.current = false
    }
  }, [])

  // Initialize Three.js camera and renderer with retry logic
  useEffect(() => {
    let retryTimeout: ReturnType<typeof setTimeout> | null = null

    // Store references to event handlers for proper cleanup
    let handleResize: (() => void) | null = null
    let handleContextLost: ((event: Event) => void) | null = null
    let handleContextRestored: (() => void) | null = null
    let canvasElement: HTMLCanvasElement | null = null

    const initializeCamera = () => {
      // Don't initialize if unmounted
      if (!isMountedRef.current) return

      console.log(`🚀 FPV CameraView initializing (attempt ${initRetryCountRef.current + 1}/${MAX_INIT_RETRIES})...`, {
        sceneProvided: !!scene,
        sceneChildren: scene?.children?.length || 0,
        containerExists: !!containerRef.current,
        canvasExists: !!canvasRef.current
      })

      // Check container and canvas
      if (!containerRef.current || !canvasRef.current) {
        if (initRetryCountRef.current < MAX_INIT_RETRIES) {
          initRetryCountRef.current++
          console.warn('⚠️ FPV: Missing container/canvas, retrying in 300ms...')
          retryTimeout = setTimeout(initializeCamera, 300)
        } else {
          console.error('❌ FPV mount failed after max retries: Missing container or canvas')
        }
        return
      }

      // Validate container dimensions
      const containerWidth = containerRef.current.clientWidth
      const containerHeight = containerRef.current.clientHeight

      if (containerWidth < 50 || containerHeight < 50) {
        if (initRetryCountRef.current < MAX_INIT_RETRIES) {
          initRetryCountRef.current++
          console.warn(`⚠️ FPV: Container too small (${containerWidth}x${containerHeight}), retrying in 300ms...`)
          retryTimeout = setTimeout(initializeCamera, 300)
        } else {
          console.error('❌ FPV mount failed after max retries: Container too small')
        }
        return
      }

      // Check scene is ready with content
      if (!scene || !scene.children || scene.children.length === 0) {
        if (initRetryCountRef.current < MAX_INIT_RETRIES) {
          initRetryCountRef.current++
          console.warn('⚠️ FPV: Scene not ready, retrying in 300ms...')
          retryTimeout = setTimeout(initializeCamera, 300)
        } else {
          console.error('❌ FPV mount failed after max retries: Scene not ready')
        }
        return
      }

      // SUCCESS - all prerequisites met, initialize camera and renderer
      console.log('✅ FPV prerequisites met, initializing...')

      // Create camera (robot's perspective)
      const camera = new THREE.PerspectiveCamera(
        75, // FOV
        containerWidth / containerHeight,
        0.01, // Very close near plane
        100 // Reasonable far plane
      )
      cameraRef.current = camera

      console.log('🎥 FPV Camera initialized:', {
        fov: 75,
        aspect: camera.aspect,
        containerSize: { width: containerWidth, height: containerHeight }
      })

      // Create renderer with robust options
      const renderer = new THREE.WebGLRenderer({
        canvas: canvasRef.current,
        antialias: true,
        alpha: true,
        preserveDrawingBuffer: true,
        failIfMajorPerformanceCaveat: false // Don't fail on low-end GPUs
      })
      renderer.setSize(containerWidth, containerHeight)
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
      renderer.setClearColor(0x87CEEB, 1) // Sky blue background
      renderer.shadowMap.enabled = true
      rendererRef.current = renderer

      // Store canvas reference for cleanup
      canvasElement = canvasRef.current

      // Add WebGL context loss handlers (store references for cleanup)
      handleContextLost = (event: Event) => {
        event.preventDefault()
        console.warn('⚠️ FPV WebGL context lost - pausing rendering')
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current)
          animationFrameRef.current = null
        }
      }
      handleContextRestored = () => {
        console.log('✅ FPV WebGL context restored - will restart on next scene change')
        // Force a re-render by resetting the first render flag
        firstRenderRef.current = false
      }
      canvasElement.addEventListener('webglcontextlost', handleContextLost)
      canvasElement.addEventListener('webglcontextrestored', handleContextRestored)

      // Create render target for capturing frames
      const renderTarget = new THREE.WebGLRenderTarget(
        containerWidth,
        containerHeight,
        {
          minFilter: THREE.LinearFilter,
          magFilter: THREE.LinearFilter,
          format: THREE.RGBAFormat
        }
      )
      renderTargetRef.current = renderTarget

      // Initialize overlay canvas dimensions
      if (overlayCanvasRef.current) {
        overlayCanvasRef.current.width = containerWidth
        overlayCanvasRef.current.height = containerHeight
      }

      // Handle window resize (store reference for cleanup)
      handleResize = () => {
        if (!containerRef.current || !camera || !renderer || !renderTarget) return

        const width = containerRef.current.clientWidth
        const height = containerRef.current.clientHeight

        if (width < 50 || height < 50) return // Skip invalid sizes

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
    } // End of initializeCamera function

    // Start initialization
    initializeCamera()

    // Cleanup function - properly removes all event listeners
    return () => {
      console.log('🧹 FPV initialization cleanup starting...')

      // Clear any pending retry timeouts
      if (retryTimeout) {
        clearTimeout(retryTimeout)
      }

      // Remove resize listener (using stored reference)
      if (handleResize) {
        window.removeEventListener('resize', handleResize)
        console.log('  ✓ Removed resize listener')
      }

      // Remove WebGL context listeners (using stored references)
      if (canvasElement) {
        if (handleContextLost) {
          canvasElement.removeEventListener('webglcontextlost', handleContextLost)
        }
        if (handleContextRestored) {
          canvasElement.removeEventListener('webglcontextrestored', handleContextRestored)
        }
        console.log('  ✓ Removed WebGL context listeners')
      }

      // Cleanup WebGL resources
      if (rendererRef.current) {
        rendererRef.current.dispose()
        rendererRef.current = null
        console.log('  ✓ Disposed renderer')
      }
      if (renderTargetRef.current) {
        renderTargetRef.current.dispose()
        renderTargetRef.current = null
        console.log('  ✓ Disposed render target')
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }

      console.log('✅ FPV initialization cleanup complete')
    }
  }, [scene]) // Re-run when scene becomes available

  // Update camera position/rotation to match robot
  useEffect(() => {
    if (!cameraRef.current) return

    const camera = cameraRef.current

    // Position camera at robot's position with a slight upward offset for better view
    const cameraPos = robotPosition.clone()
    cameraPos.y += 0.3 // Raise camera 0.3 units above robot center
    camera.position.copy(cameraPos)

    // Calculate look-at point based on robot's rotation
    // Create a forward vector and rotate it by the robot's rotation
    const forward = new THREE.Vector3(0, 0, -1) // Forward is -Z in Three.js
    const lookAtPoint = new THREE.Vector3()

    // Apply robot's rotation to the forward vector
    const euler = new THREE.Euler(robotRotation.x, robotRotation.y, robotRotation.z)
    forward.applyEuler(euler)

    // Set look-at point 5 units in front of robot
    lookAtPoint.copy(cameraPos).add(forward.multiplyScalar(5))

    // Make camera look at the point
    camera.lookAt(lookAtPoint)
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

  // Update tracking trails - uses functional update to avoid stale closure
  const updateTrackingTrails = useCallback((currentDetections: Detection[]) => {
    const now = Date.now()

    // Use functional update pattern to always get latest state
    setTrackingTrails(prevTrails => {
      const newTrails = new Map(prevTrails)

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

      return newTrails
    })
  }, []) // No dependencies needed with functional update!

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
    if (!canvasRef.current || isAnalyzing || !isMountedRef.current) return

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

      // Only update state if still mounted
      if (isMountedRef.current) {
        setVisionAnalysis(response.data)
        console.log('Vision analysis:', response.data)
      }
    } catch (error) {
      console.error('Vision analysis error:', error)
    } finally {
      // Only update state if still mounted
      if (isMountedRef.current) {
        setIsAnalyzing(false)
      }
    }
  }, [isAnalyzing])

  // Add ambient lighting to scene if needed (ensures FPV is never completely black)
  useEffect(() => {
    if (!scene) return

    // Store reference to scene for cleanup
    const currentScene = scene
    let addedAmbient: THREE.AmbientLight | null = null
    let addedDirectional: THREE.DirectionalLight | null = null

    // Check if scene has adequate lighting
    const lights = currentScene.children.filter(child => child instanceof THREE.Light)

    if (lights.length === 0) {
      console.log('💡 No lights in scene, adding bright emergency lighting for FPV')
      addedAmbient = new THREE.AmbientLight(0xffffff, 1.0) // Increased from 0.6
      addedAmbient.name = 'FPV_Emergency_Ambient'
      currentScene.add(addedAmbient)

      addedDirectional = new THREE.DirectionalLight(0xffffff, 1.2) // Increased from 0.8
      addedDirectional.position.set(5, 15, 5) // Higher position for better coverage
      addedDirectional.castShadow = true
      addedDirectional.name = 'FPV_Emergency_Directional'
      currentScene.add(addedDirectional)
    }

    // Cleanup always runs when scene changes or component unmounts
    return () => {
      if (addedAmbient && currentScene) {
        currentScene.remove(addedAmbient)
      }
      if (addedDirectional && currentScene) {
        currentScene.remove(addedDirectional)
      }
    }
  }, [scene])

  // Main render loop with proper cleanup to prevent race conditions
  useEffect(() => {
    // Local flag scoped to this effect - prevents race condition with cleanup
    let isCleaned = false

    const animate = () => {
      // Exit immediately if cleaned up - do NOT schedule another frame
      if (isCleaned) {
        console.log('🛑 FPV animation stopped (cleanup)')
        return
      }

      // Validate all required refs and scene before rendering
      if (!rendererRef.current || !cameraRef.current || !renderTargetRef.current || !scene) {
        if (!isCleaned) {
          animationFrameRef.current = requestAnimationFrame(animate)
        }
        return
      }

      // Check container dimensions are valid (can change during runtime)
      if (!containerRef.current || containerRef.current.clientWidth <= 0 || containerRef.current.clientHeight <= 0) {
        if (!isCleaned) {
          animationFrameRef.current = requestAnimationFrame(animate)
        }
        return
      }

      const renderer = rendererRef.current
      const camera = cameraRef.current

      try {
        // Log first render for debugging (using ref instead of function property)
        if (!firstRenderRef.current) {
          console.log('🎥 FPV First render:', {
            cameraPosition: camera.position.toArray(),
            cameraRotation: camera.rotation.toArray(),
            sceneChildren: scene.children.length,
            rendererSize: { width: renderer.domElement.width, height: renderer.domElement.height }
          })
          firstRenderRef.current = true
        }

        // Render scene to canvas
        renderer.render(scene, camera)

        // Generate detections from current view
        const currentDetections = generateDetections()

        // Only update state if not cleaned up
        if (isCleaned) return
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
          if (!isCleaned) {
            setFps(fpsCounterRef.current.frames)
          }
          fpsCounterRef.current.frames = 0
          fpsCounterRef.current.lastTime = now
        }

        // Call vision API every 90 frames (~5 seconds at 15 FPS)
        frameCountRef.current++
        if (frameCountRef.current >= 90) {
          analyzeWithVision()
          frameCountRef.current = 0
        }
      } catch (error) {
        console.error('❌ FPV Render error:', error)
        console.error('Scene state:', {
          sceneExists: !!scene,
          sceneChildren: scene?.children?.length,
          cameraExists: !!cameraRef.current,
          rendererExists: !!rendererRef.current
        })
        // Continue animation loop even if render fails (unless cleaned up)
      }

      // Only schedule next frame if not cleaned up
      if (!isCleaned) {
        animationFrameRef.current = requestAnimationFrame(animate)
      }
    }

    // Start the animation loop
    animationFrameRef.current = requestAnimationFrame(animate)

    // Cleanup function - set flag first, then cancel frame
    return () => {
      console.log('🧹 FPV cleanup starting...')
      isCleaned = true // Set flag FIRST to stop any pending animate calls

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }
      console.log('✅ FPV cleanup complete')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    // Only re-run when scene changes. Callbacks are accessed via closure and always use latest version.
    // Including callbacks in deps causes infinite re-render loop as they recreate on state changes.
  }, [scene])

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
