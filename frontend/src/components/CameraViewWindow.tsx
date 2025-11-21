import { useState, useEffect } from 'react'
import Draggable from 'react-draggable'
import * as THREE from 'three'
import CameraView from './CameraView'

interface CameraViewWindowProps {
  isVisible: boolean
  onClose: () => void
  robotPosition?: [number, number, number]
  robotRotation?: [number, number, number]
  sceneConfig?: any
  editableObjects?: any[]
  scene: THREE.Scene | null
}

export default function CameraViewWindow({
  isVisible,
  onClose,
  robotPosition = [0, 0.5, 0],
  robotRotation = [0, 0, 0],
  sceneConfig,
  editableObjects = [],
  scene
}: CameraViewWindowProps) {
  const [isMinimized, setIsMinimized] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [position, setPosition] = useState(() => {
    const saved = localStorage.getItem('cameraViewPosition')
    return saved ? JSON.parse(saved) : { x: 0, y: 0 }
  })

  // Persist position to localStorage
  useEffect(() => {
    localStorage.setItem('cameraViewPosition', JSON.stringify(position))
  }, [position])

  if (!isVisible) return null

  // Show loading message if scene isn't ready yet
  // Check both scene existence AND that it has been properly initialized with content
  const sceneReady = scene && scene.children && scene.children.length > 0

  if (!sceneReady) {
    const reason = !scene
      ? 'Waiting for 3D scene to initialize...'
      : 'Scene is loading content...'

    console.log('🎥 FPV Camera waiting for scene:', {
      sceneExists: !!scene,
      sceneChildren: scene?.children?.length || 0,
      reason
    })

    return (
      <div className="camera-view-window loading" style={{
        position: 'fixed',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        background: 'rgba(0,0,0,0.9)',
        padding: '20px',
        borderRadius: '8px',
        color: '#fff',
        zIndex: 1000
      }}>
        <p>⏳ Initializing camera view...</p>
        <p style={{ fontSize: '0.9em', opacity: 0.7 }}>{reason}</p>
      </div>
    )
  }

  console.log('✅ FPV Camera scene ready, rendering CameraView')

  // Convert objects to format expected by CameraView
  const objects = editableObjects.map(obj => ({
    position: new THREE.Vector3(obj.position[0], obj.position[1], obj.position[2]),
    label: obj.type || 'object',
    radius: obj.radius || (obj.size ? Math.max(...obj.size) / 2 : 0.5),
    color: obj.color
  }))

  // Add scene config objects if present
  if (sceneConfig?.objects) {
    sceneConfig.objects.forEach((obj: any) => {
      objects.push({
        position: new THREE.Vector3(obj.position[0], obj.position[1], obj.position[2]),
        label: obj.name || obj.type || 'object',
        radius: obj.radius || obj.scale?.[0] || 0.5,
        color: obj.color
      })
    })
  }

  const handleDetectionsUpdate = (detections: any[]) => {
    console.log('CV Detections:', detections)
  }

  return (
    <Draggable
      position={position}
      onDrag={(e, data) => setPosition({ x: data.x, y: data.y })}
      onStart={() => setIsDragging(true)}
      onStop={() => setIsDragging(false)}
      handle=".window-header"
      bounds="parent"
    >
      <div className={`camera-view-window ${isMinimized ? 'minimized' : ''} ${isDragging ? 'dragging' : ''}`}>
        <div className="window-header">
          <h4>🎥 Computer Vision Camera</h4>
          <div className="window-controls">
            <button
              className="window-minimize"
              onClick={() => setIsMinimized(!isMinimized)}
              title={isMinimized ? 'Maximize' : 'Minimize'}
            >
              {isMinimized ? '▲' : '▼'}
            </button>
            <button className="window-close" onClick={onClose} title="Close camera view">
              ×
            </button>
          </div>
        </div>
        {!isMinimized && (
          <div className="camera-viewport">
            <CameraView
              scene={scene}
              robotPosition={new THREE.Vector3(robotPosition[0], robotPosition[1], robotPosition[2])}
              robotRotation={new THREE.Euler(robotRotation[0], robotRotation[1], robotRotation[2])}
              objects={objects}
              onDetectionsUpdate={handleDetectionsUpdate}
            />
          </div>
        )}
      </div>
    </Draggable>
  )
}
