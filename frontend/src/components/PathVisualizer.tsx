import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Line } from '@react-three/drei'
import * as THREE from 'three'
import type { ComputedPath, Obstacle } from '../types/PathPlanning'

interface PathVisualizerProps {
  origin: THREE.Vector3 | null
  destination: THREE.Vector3 | null
  path: ComputedPath | null
  obstacles: Obstacle[]
  showGrid?: boolean
}

export default function PathVisualizer({
  origin,
  destination,
  path,
  obstacles,
  showGrid = false
}: PathVisualizerProps) {
  const pathLineRef = useRef<any>(null)
  const gridRef = useRef<THREE.GridHelper>(null)

  // Animate path line
  useFrame(({ clock }) => {
    if (pathLineRef.current && path?.isValid) {
      const material = pathLineRef.current.material as THREE.LineBasicMaterial
      material.opacity = 0.7 + Math.sin(clock.elapsedTime * 2) * 0.3
    }
  })

  return (
    <group>
      {/* Origin marker (green sphere) */}
      {origin && (
        <mesh position={[origin.x, origin.y, origin.z]}>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshStandardMaterial
            color="#00ff00"
            emissive="#00ff00"
            emissiveIntensity={0.5}
          />
        </mesh>
      )}

      {/* Destination marker (red sphere) */}
      {destination && (
        <mesh position={[destination.x, destination.y, destination.z]}>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshStandardMaterial
            color="#ff0000"
            emissive="#ff0000"
            emissiveIntensity={0.5}
          />
        </mesh>
      )}

      {/* Computed path line */}
      {path && path.waypoints.length > 1 && (
        <>
          <Line
            ref={pathLineRef}
            points={path.waypoints.map(p => [p.x, p.y, p.z])}
            color={path.isValid ? "#00ff00" : "#ff0000"}
            lineWidth={3}
            transparent
            opacity={0.8}
          />

          {/* Waypoint markers */}
          {path.waypoints.map((waypoint, index) => (
            <mesh key={index} position={[waypoint.x, waypoint.y, waypoint.z]}>
              <sphereGeometry args={[0.1, 8, 8]} />
              <meshBasicMaterial color={path.isValid ? "#00ff00" : "#ff0000"} />
            </mesh>
          ))}
        </>
      )}

      {/* Obstacles */}
      {obstacles.map((obstacle) => {
        switch (obstacle.type) {
          case 'box':
            return (
              <mesh
                key={obstacle.id}
                position={[obstacle.position.x, obstacle.position.y, obstacle.position.z]}
                rotation={obstacle.rotation}
              >
                <boxGeometry args={[
                  obstacle.size.x,
                  obstacle.size.y,
                  obstacle.size.z
                ]} />
                <meshStandardMaterial
                  color={obstacle.color || "#ff6b6b"}
                  transparent
                  opacity={0.7}
                />
              </mesh>
            )

          case 'cylinder':
            return (
              <mesh
                key={obstacle.id}
                position={[obstacle.position.x, obstacle.position.y, obstacle.position.z]}
                rotation={obstacle.rotation}
              >
                <cylinderGeometry args={[
                  obstacle.radius || 0.5,
                  obstacle.radius || 0.5,
                  obstacle.size.y,
                  16
                ]} />
                <meshStandardMaterial
                  color={obstacle.color || "#ff6b6b"}
                  transparent
                  opacity={0.7}
                />
              </mesh>
            )

          case 'sphere':
            return (
              <mesh
                key={obstacle.id}
                position={[obstacle.position.x, obstacle.position.y, obstacle.position.z]}
              >
                <sphereGeometry args={[obstacle.radius || 0.5, 16, 16]} />
                <meshStandardMaterial
                  color={obstacle.color || "#ff6b6b"}
                  transparent
                  opacity={0.7}
                />
              </mesh>
            )

          case 'plant':
            // Simple plant representation: cylinder (trunk) + sphere (foliage)
            return (
              <group
                key={obstacle.id}
                position={[obstacle.position.x, obstacle.position.y, obstacle.position.z]}
              >
                <mesh position={[0, 0.3, 0]}>
                  <cylinderGeometry args={[0.1, 0.15, 0.6, 8]} />
                  <meshStandardMaterial color="#8B4513" />
                </mesh>
                <mesh position={[0, 0.9, 0]}>
                  <sphereGeometry args={[0.4, 12, 12]} />
                  <meshStandardMaterial color="#228B22" />
                </mesh>
              </group>
            )

          case 'wall':
            return (
              <mesh
                key={obstacle.id}
                position={[obstacle.position.x, obstacle.position.y, obstacle.position.z]}
                rotation={obstacle.rotation}
              >
                <boxGeometry args={[
                  obstacle.size.x,
                  obstacle.size.y,
                  0.2 // thin wall
                ]} />
                <meshStandardMaterial color="#666666" />
              </mesh>
            )

          case 'custom-area':
            // Custom polygon area - render as flat polygon
            if (obstacle.points && obstacle.points.length >= 3) {
              const shape = new THREE.Shape()
              shape.moveTo(obstacle.points[0].x, obstacle.points[0].z)
              for (let i = 1; i < obstacle.points.length; i++) {
                shape.lineTo(obstacle.points[i].x, obstacle.points[i].z)
              }
              shape.lineTo(obstacle.points[0].x, obstacle.points[0].z)

              return (
                <mesh
                  key={obstacle.id}
                  rotation={[-Math.PI / 2, 0, 0]}
                  position={[0, obstacle.position.y, 0]}
                >
                  <shapeGeometry args={[shape]} />
                  <meshStandardMaterial
                    color="#ff0000"
                    transparent
                    opacity={0.3}
                    side={THREE.DoubleSide}
                  />
                </mesh>
              )
            }
            return null

          default:
            return null
        }
      })}

      {/* Ground grid overlay (optional) */}
      {showGrid && (
        <gridHelper
          ref={gridRef}
          args={[20, 20, "#444444", "#222222"]}
          position={[0, 0.01, 0]}
        />
      )}
    </group>
  )
}
