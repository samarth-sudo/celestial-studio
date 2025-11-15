import { useEffect, useRef, useState } from 'react'
import { useThree } from '@react-three/fiber'
import * as THREE from 'three'
import type { InteractionMode, Obstacle } from '../types/PathPlanning'

interface SceneInteractionProps {
  mode: InteractionMode
  onOriginSet?: (position: THREE.Vector3) => void
  onDestinationSet?: (position: THREE.Vector3) => void
  onObstaclePlaced?: (obstacle: Omit<Obstacle, 'id'>) => void
  onObstacleSelected?: (obstacleId: string | null) => void
  obstacles?: Obstacle[]
  currentObstacleType?: Obstacle['type']
}

export default function SceneInteraction({
  mode,
  onOriginSet,
  onDestinationSet,
  onObstaclePlaced,
  onObstacleSelected,
  obstacles = [],
  currentObstacleType = 'box'
}: SceneInteractionProps) {
  const { camera, gl, scene } = useThree()
  const [hoverPosition, setHoverPosition] = useState<THREE.Vector3 | null>(null)
  const raycaster = useRef(new THREE.Raycaster())
  const groundPlane = useRef(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0))

  useEffect(() => {
    const canvas = gl.domElement

    const handleMouseMove = (event: MouseEvent) => {
      if (mode === 'none') {
        setHoverPosition(null)
        return
      }

      // Calculate mouse position in normalized device coordinates
      const rect = canvas.getBoundingClientRect()
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
      )

      // Update raycaster
      raycaster.current.setFromCamera(mouse, camera)

      // Intersect with ground plane
      const intersectPoint = new THREE.Vector3()
      raycaster.current.ray.intersectPlane(groundPlane.current, intersectPoint)

      if (intersectPoint) {
        setHoverPosition(intersectPoint.clone())
      }
    }

    const handleClick = () => {
      if (mode === 'none' || !hoverPosition) return

      const clickPosition = hoverPosition.clone()

      switch (mode) {
        case 'place-origin':
          onOriginSet?.(clickPosition)
          break

        case 'place-destination':
          onDestinationSet?.(clickPosition)
          break

        case 'place-obstacle':
          const defaultSize = new THREE.Vector3(1, 1, 1)
          const defaultRadius = 0.5

          const newObstacle: Omit<Obstacle, 'id'> = {
            type: currentObstacleType,
            position: clickPosition,
            size: defaultSize,
            radius: currentObstacleType === 'cylinder' || currentObstacleType === 'sphere' ? defaultRadius : undefined,
            rotation: new THREE.Euler(0, 0, 0),
            color: '#ff6b6b'
          }

          onObstaclePlaced?.(newObstacle)
          break

        case 'delete-obstacle':
          // Raycast to check if we clicked on an obstacle
          const intersects = raycaster.current.intersectObjects(scene.children, true)
          for (const intersect of intersects) {
            // Find obstacle by matching mesh
            const obstacle = obstacles.find(obs => {
              // Check if intersected object is part of this obstacle
              return intersect.object.position.equals(obs.position)
            })
            if (obstacle) {
              onObstacleSelected?.(obstacle.id)
              break
            }
          }
          break

        default:
          break
      }
    }

    canvas.addEventListener('mousemove', handleMouseMove)
    canvas.addEventListener('click', handleClick)

    return () => {
      canvas.removeEventListener('mousemove', handleMouseMove)
      canvas.removeEventListener('click', handleClick)
    }
  }, [mode, camera, gl, scene, hoverPosition, obstacles, currentObstacleType, onOriginSet, onDestinationSet, onObstaclePlaced, onObstacleSelected])

  // Render hover preview
  if (!hoverPosition || mode === 'none') return null

  return (
    <group>
      {mode === 'place-origin' && (
        <mesh position={[hoverPosition.x, 0.1, hoverPosition.z]}>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshBasicMaterial color="#00ff00" transparent opacity={0.5} />
        </mesh>
      )}

      {mode === 'place-destination' && (
        <mesh position={[hoverPosition.x, 0.1, hoverPosition.z]}>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshBasicMaterial color="#ff0000" transparent opacity={0.5} />
        </mesh>
      )}

      {mode === 'place-obstacle' && (
        <>
          {currentObstacleType === 'box' && (
            <mesh position={[hoverPosition.x, 0.5, hoverPosition.z]}>
              <boxGeometry args={[1, 1, 1]} />
              <meshBasicMaterial color="#ff6b6b" transparent opacity={0.3} wireframe />
            </mesh>
          )}

          {currentObstacleType === 'cylinder' && (
            <mesh position={[hoverPosition.x, 0.5, hoverPosition.z]}>
              <cylinderGeometry args={[0.5, 0.5, 1, 16]} />
              <meshBasicMaterial color="#ff6b6b" transparent opacity={0.3} wireframe />
            </mesh>
          )}

          {currentObstacleType === 'sphere' && (
            <mesh position={[hoverPosition.x, 0.5, hoverPosition.z]}>
              <sphereGeometry args={[0.5, 16, 16]} />
              <meshBasicMaterial color="#ff6b6b" transparent opacity={0.3} wireframe />
            </mesh>
          )}

          {currentObstacleType === 'plant' && (
            <group position={[hoverPosition.x, 0, hoverPosition.z]}>
              <mesh position={[0, 0.3, 0]}>
                <cylinderGeometry args={[0.1, 0.15, 0.6, 8]} />
                <meshBasicMaterial color="#8B4513" transparent opacity={0.3} />
              </mesh>
              <mesh position={[0, 0.9, 0]}>
                <sphereGeometry args={[0.4, 12, 12]} />
                <meshBasicMaterial color="#228B22" transparent opacity={0.3} />
              </mesh>
            </group>
          )}

          {currentObstacleType === 'wall' && (
            <mesh position={[hoverPosition.x, 1, hoverPosition.z]}>
              <boxGeometry args={[2, 2, 0.2]} />
              <meshBasicMaterial color="#666666" transparent opacity={0.3} wireframe />
            </mesh>
          )}
        </>
      )}

      {/* Hover indicator circle */}
      <mesh position={[hoverPosition.x, 0.01, hoverPosition.z]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.3, 0.35, 32]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.5} />
      </mesh>
    </group>
  )
}
