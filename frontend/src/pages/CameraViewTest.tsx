import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import CameraView from '../components/CameraView'
import './CameraViewTest.css'

export default function CameraViewTest() {
  const sceneRef = useRef<THREE.Scene | null>(null)
  const [robotPosition, setRobotPosition] = useState(new THREE.Vector3(0, 1, 0))
  const [robotRotation, setRobotRotation] = useState(new THREE.Euler(0, 0, 0))
  const [objects, setObjects] = useState<Array<{
    position: THREE.Vector3
    label: string
    radius: number
    color?: string
  }>>([])

  // Initialize Three.js scene
  useEffect(() => {
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x222222)
    sceneRef.current = scene

    // Add lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
    directionalLight.position.set(10, 20, 10)
    scene.add(directionalLight)

    // Add ground plane
    const groundGeometry = new THREE.PlaneGeometry(50, 50)
    const groundMaterial = new THREE.MeshStandardMaterial({
      color: 0x333333,
      metalness: 0.1,
      roughness: 0.8
    })
    const ground = new THREE.Mesh(groundGeometry, groundMaterial)
    ground.rotation.x = -Math.PI / 2
    ground.receiveShadow = true
    scene.add(ground)

    // Add grid helper
    const gridHelper = new THREE.GridHelper(50, 50, 0x444444, 0x444444)
    scene.add(gridHelper)

    // Create test objects
    const testObjects = [
      // Red boxes (obstacles)
      { position: new THREE.Vector3(3, 0.5, 2), label: 'red_box_1', radius: 0.5, color: '#ff0000' },
      { position: new THREE.Vector3(-2, 0.5, 4), label: 'red_box_2', radius: 0.5, color: '#ff0000' },
      { position: new THREE.Vector3(5, 0.5, -3), label: 'red_box_3', radius: 0.5, color: '#ff0000' },

      // Blue cylinders (targets)
      { position: new THREE.Vector3(-4, 0.75, -2), label: 'blue_cylinder_1', radius: 0.4, color: '#0000ff' },
      { position: new THREE.Vector3(6, 0.75, 3), label: 'blue_cylinder_2', radius: 0.4, color: '#0000ff' },

      // Green spheres (waypoints)
      { position: new THREE.Vector3(0, 0.3, 5), label: 'green_sphere_1', radius: 0.3, color: '#00ff00' },
      { position: new THREE.Vector3(4, 0.3, 0), label: 'green_sphere_2', radius: 0.3, color: '#00ff00' },
      { position: new THREE.Vector3(-3, 0.3, -4), label: 'green_sphere_3', radius: 0.3, color: '#00ff00' },

      // Yellow cones (markers)
      { position: new THREE.Vector3(2, 0.5, -5), label: 'yellow_marker_1', radius: 0.3, color: '#ffff00' },
      { position: new THREE.Vector3(-5, 0.5, 1), label: 'yellow_marker_2', radius: 0.3, color: '#ffff00' },
    ]

    // Add meshes to scene
    testObjects.forEach(obj => {
      let geometry: THREE.BufferGeometry

      if (obj.label.includes('box')) {
        geometry = new THREE.BoxGeometry(obj.radius * 2, obj.radius * 2, obj.radius * 2)
      } else if (obj.label.includes('cylinder')) {
        geometry = new THREE.CylinderGeometry(obj.radius, obj.radius, obj.radius * 3, 16)
      } else if (obj.label.includes('sphere')) {
        geometry = new THREE.SphereGeometry(obj.radius, 16, 16)
      } else {
        geometry = new THREE.ConeGeometry(obj.radius, obj.radius * 3, 8)
      }

      const material = new THREE.MeshStandardMaterial({
        color: obj.color || '#888888',
        metalness: 0.3,
        roughness: 0.7
      })

      const mesh = new THREE.Mesh(geometry, material)
      mesh.position.copy(obj.position)
      mesh.castShadow = true
      mesh.receiveShadow = true
      scene.add(mesh)
    })

    // Add robot representation (simple cube)
    const robotGeometry = new THREE.BoxGeometry(0.6, 0.4, 0.8)
    const robotMaterial = new THREE.MeshStandardMaterial({
      color: 0x00aaff,
      metalness: 0.6,
      roughness: 0.4
    })
    const robotMesh = new THREE.Mesh(robotGeometry, robotMaterial)
    robotMesh.position.copy(robotPosition)
    robotMesh.castShadow = true
    scene.add(robotMesh)

    setObjects(testObjects)

    // Store robot mesh for animation
    ;(window as any).robotMesh = robotMesh

    return () => {
      scene.clear()
    }
  }, [])

  // Animate robot movement
  useEffect(() => {
    let animationFrame: number
    let time = 0

    const animate = () => {
      time += 0.016 // ~60fps

      // Move robot in a circle
      const radius = 3
      const x = Math.cos(time * 0.5) * radius
      const z = Math.sin(time * 0.5) * radius
      const y = 1

      const newPosition = new THREE.Vector3(x, y, z)
      setRobotPosition(newPosition)

      // Rotate robot to face forward in movement direction
      const angle = Math.atan2(Math.sin(time * 0.5), Math.cos(time * 0.5)) + Math.PI / 2
      setRobotRotation(new THREE.Euler(0, angle, 0))

      // Update robot mesh
      const robotMesh = (window as any).robotMesh as THREE.Mesh
      if (robotMesh) {
        robotMesh.position.copy(newPosition)
        robotMesh.rotation.y = angle
      }

      animationFrame = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      cancelAnimationFrame(animationFrame)
    }
  }, [])

  const handleDetectionsUpdate = (detections: any[]) => {
    console.log('Detections updated:', detections)
  }

  return (
    <div className="camera-view-test">
      <div className="test-header">
        <h1>🎥 Camera View Test</h1>
        <p>Testing real-time computer vision camera view with LLaVA integration</p>
        <div className="test-info">
          <div className="info-item">
            <span className="info-label">Robot Position:</span>
            <span className="info-value">
              ({robotPosition.x.toFixed(2)}, {robotPosition.y.toFixed(2)}, {robotPosition.z.toFixed(2)})
            </span>
          </div>
          <div className="info-item">
            <span className="info-label">Robot Rotation:</span>
            <span className="info-value">
              {(robotRotation.y * (180 / Math.PI)).toFixed(0)}°
            </span>
          </div>
          <div className="info-item">
            <span className="info-label">Objects in Scene:</span>
            <span className="info-value">{objects.length}</span>
          </div>
        </div>
      </div>

      <div className="test-content">
        {sceneRef.current && (
          <CameraView
            scene={sceneRef.current}
            robotPosition={robotPosition}
            robotRotation={robotRotation}
            objects={objects}
            onDetectionsUpdate={handleDetectionsUpdate}
          />
        )}
      </div>

      <div className="test-instructions">
        <h3>What to Look For:</h3>
        <ul>
          <li>✅ Camera view renders from robot's perspective</li>
          <li>✅ Bounding boxes appear around visible objects</li>
          <li>✅ Labels show object names and confidence scores</li>
          <li>✅ Distance measurements displayed for each object</li>
          <li>✅ Tracking trails follow moving objects</li>
          <li>✅ FPS counter shows ~60 FPS</li>
          <li>✅ Statistics panel updates in real-time</li>
          <li>✅ Vision analysis panel appears after ~5 seconds (LLaVA)</li>
        </ul>

        <h3>Expected Behavior:</h3>
        <ul>
          <li>Robot moves in circular path</li>
          <li>Objects enter and exit camera view</li>
          <li>Bounding boxes change color based on confidence</li>
          <li>Trails fade as objects move</li>
          <li>LLaVA analyzes scene every 5 seconds</li>
        </ul>

        <h3>Console Output:</h3>
        <p>Check browser console for detection logs and vision analysis results</p>
      </div>
    </div>
  )
}
