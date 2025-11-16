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
  if (!isVisible || !scene) return null

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
    <div className="camera-view-window">
      <div className="window-header">
        <h4>🎥 Computer Vision Camera</h4>
        <button className="window-close" onClick={onClose} title="Close camera view">
          ×
        </button>
      </div>
      <div className="camera-viewport">
        <CameraView
          scene={scene}
          robotPosition={new THREE.Vector3(robotPosition[0], robotPosition[1], robotPosition[2])}
          robotRotation={new THREE.Euler(robotRotation[0], robotRotation[1], robotRotation[2])}
          objects={objects}
          onDetectionsUpdate={handleDetectionsUpdate}
        />
      </div>
    </div>
  )
}
