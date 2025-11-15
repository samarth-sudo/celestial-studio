import { useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { PerspectiveCamera, OrbitControls } from '@react-three/drei'
import { Physics } from '@react-three/rapier'
import * as THREE from 'three'

interface CameraViewWindowProps {
  isVisible: boolean
  onClose: () => void
  robotPosition?: [number, number, number]
  robotRotation?: [number, number, number]
  sceneConfig?: any
  editableObjects?: any[]
}

// Component to sync camera with robot position
function RobotCamera({ position, rotation }: { position: [number, number, number], rotation: [number, number, number] }) {
  const cameraRef = useRef<THREE.PerspectiveCamera>(null)

  // Position camera at robot's eye level, looking forward
  const cameraPosition: [number, number, number] = [
    position[0],
    position[1] + 0.5, // Slightly above robot
    position[2]
  ]

  return (
    <PerspectiveCamera
      ref={cameraRef}
      makeDefault
      position={cameraPosition}
      rotation={rotation}
      fov={75}
    />
  )
}

export default function CameraViewWindow({
  isVisible,
  onClose,
  robotPosition = [0, 0.5, 0],
  robotRotation = [0, 0, 0],
  sceneConfig,
  editableObjects = []
}: CameraViewWindowProps) {
  if (!isVisible) return null

  return (
    <div className="camera-view-window">
      <div className="window-header">
        <h4>📹 Robot Camera (FPV)</h4>
        <button className="window-close" onClick={onClose} title="Close camera view">
          ×
        </button>
      </div>
      <div className="camera-viewport">
        <Canvas shadows>
          <color attach="background" args={['#0a0a0a']} />

          {/* Robot's perspective camera */}
          <RobotCamera position={robotPosition} rotation={robotRotation} />

          {/* Lighting */}
          <ambientLight intensity={0.4} />
          <directionalLight position={[5, 10, 5]} intensity={0.8} castShadow />
          <pointLight position={robotPosition} intensity={0.5} distance={10} />

          {/* Render scene objects in camera view */}
          {editableObjects.map((obj: any) => (
            <mesh
              key={obj.id}
              position={obj.position}
              castShadow
              receiveShadow
            >
              {obj.shape === 'sphere' ? (
                <sphereGeometry args={obj.size || [0.3, 16, 16]} />
              ) : obj.shape === 'cylinder' ? (
                <cylinderGeometry args={[obj.radius || 0.3, obj.radius || 0.3, obj.size?.[1] || 1, 16]} />
              ) : (
                <boxGeometry args={obj.size || [0.5, 0.5, 0.5]} />
              )}
              <meshStandardMaterial color={obj.color || '#888'} />
            </mesh>
          ))}

          {/* Ground plane */}
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
            <planeGeometry args={[50, 50]} />
            <meshStandardMaterial color="#1a1a1a" />
          </mesh>

          {/* Grid helper */}
          <gridHelper args={[20, 20, '#444', '#222']} position={[0, 0.01, 0]} />
        </Canvas>

        {/* FPV overlay indicator */}
        <div className="fpv-overlay">
          <div className="fpv-crosshair">+</div>
          <div className="fpv-info">
            <span>X: {robotPosition[0].toFixed(1)}</span>
            <span>Y: {robotPosition[1].toFixed(1)}</span>
            <span>Z: {robotPosition[2].toFixed(1)}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
