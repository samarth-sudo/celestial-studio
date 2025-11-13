import { useRef, useState, useEffect } from 'react'
import { useFrame } from '@react-three/fiber'
import { RigidBody } from '@react-three/rapier'
import type * as THREE from 'three'
import { Line } from '@react-three/drei'

interface PathFollowingRobotProps {
  path: THREE.Vector3[]
  origin: THREE.Vector3
  destination: THREE.Vector3
}

export default function PathFollowingRobot({ path, origin, destination }: PathFollowingRobotProps) {
  const robotRef = useRef<any>(null)
  const [currentWaypointIndex, setCurrentWaypointIndex] = useState(0)
  const [isMoving, setIsMoving] = useState(false)
  const speedRef = useRef(2.0) // meters per second
  const [completedPath, setCompletedPath] = useState(false)

  // Start animation after a short delay
  useEffect(() => {
    const timer = setTimeout(() => setIsMoving(true), 500)
    return () => clearTimeout(timer)
  }, [])

  useFrame((_, delta) => {
    if (!isMoving || !robotRef.current || completedPath || path.length === 0) return

    const currentTarget = path[currentWaypointIndex]
    if (!currentTarget) return

    const currentPos = robotRef.current.translation()
    const targetPos = [currentTarget.x, currentTarget.y, currentTarget.z]

    // Calculate direction to target
    const dx = targetPos[0] - currentPos.x
    const dz = targetPos[2] - currentPos.z
    const distanceToTarget = Math.sqrt(dx * dx + dz * dz)

    // Check if we've reached the current waypoint
    if (distanceToTarget < 0.1) {
      if (currentWaypointIndex < path.length - 1) {
        setCurrentWaypointIndex(prev => prev + 1)
      } else {
        setCompletedPath(true)
        console.log('🎯 Robot reached destination!')
      }
      return
    }

    // Move towards target
    const moveDistance = speedRef.current * delta
    const normalizedDx = (dx / distanceToTarget) * moveDistance
    const normalizedDz = (dz / distanceToTarget) * moveDistance

    // Update position
    robotRef.current.setTranslation({
      x: currentPos.x + normalizedDx,
      y: 0.25, // Keep robot at ground level
      z: currentPos.z + normalizedDz
    }, true)

    // Rotate to face direction of movement
    const angle = Math.atan2(dx, dz)
    robotRef.current.setRotation({ w: Math.cos(angle / 2), x: 0, y: Math.sin(angle / 2), z: 0 }, true)
  })

  // Convert path waypoints to line points for visualization
  const pathPoints = path.map(p => [p.x, p.y + 0.05, p.z] as [number, number, number])

  return (
    <>
      {/* Visualize the path */}
      <Line
        points={pathPoints}
        color="#fbbf24"
        lineWidth={3}
        dashed={false}
      />

      {/* Origin marker */}
      <mesh position={[origin.x, 0.5, origin.z]}>
        <sphereGeometry args={[0.3, 16, 16]} />
        <meshStandardMaterial color="#10b981" emissive="#10b981" emissiveIntensity={0.5} />
      </mesh>

      {/* Destination marker */}
      <mesh position={[destination.x, 0.5, destination.z]}>
        <sphereGeometry args={[0.3, 16, 16]} />
        <meshStandardMaterial color="#ef4444" emissive="#ef4444" emissiveIntensity={0.5} />
      </mesh>

      {/* Mobile robot */}
      <RigidBody
        ref={robotRef}
        type="kinematicPosition"
        colliders="cuboid"
        position={[origin.x, 0.25, origin.z]}
      >
        {/* Robot body */}
        <mesh castShadow>
          <boxGeometry args={[0.6, 0.4, 0.8]} />
          <meshStandardMaterial color="#3b82f6" roughness={0.3} metalness={0.7} />
        </mesh>

        {/* Robot top indicator (forward direction) */}
        <mesh position={[0, 0.3, 0.3]} castShadow>
          <coneGeometry args={[0.15, 0.3, 4]} rotation={[0, 0, 0]} />
          <meshStandardMaterial color="#fbbf24" emissive="#fbbf24" emissiveIntensity={0.3} />
        </mesh>

        {/* Robot wheels */}
        <mesh position={[-0.35, -0.15, 0.3]} rotation={[0, 0, Math.PI / 2]} castShadow>
          <cylinderGeometry args={[0.15, 0.15, 0.1, 12]} />
          <meshStandardMaterial color="#1f2937" />
        </mesh>
        <mesh position={[0.35, -0.15, 0.3]} rotation={[0, 0, Math.PI / 2]} castShadow>
          <cylinderGeometry args={[0.15, 0.15, 0.1, 12]} />
          <meshStandardMaterial color="#1f2937" />
        </mesh>
        <mesh position={[-0.35, -0.15, -0.3]} rotation={[0, 0, Math.PI / 2]} castShadow>
          <cylinderGeometry args={[0.15, 0.15, 0.1, 12]} />
          <meshStandardMaterial color="#1f2937" />
        </mesh>
        <mesh position={[0.35, -0.15, -0.3]} rotation={[0, 0, Math.PI / 2]} castShadow>
          <cylinderGeometry args={[0.15, 0.15, 0.1, 12]} />
          <meshStandardMaterial color="#1f2937" />
        </mesh>

        {/* Status light */}
        <mesh position={[0, 0.35, 0]} >
          <sphereGeometry args={[0.08, 8, 8]} />
          <meshStandardMaterial
            color={completedPath ? "#10b981" : "#3b82f6"}
            emissive={completedPath ? "#10b981" : "#3b82f6"}
            emissiveIntensity={0.8}
          />
        </mesh>
      </RigidBody>

      {/* Progress indicator light beam */}
      {isMoving && !completedPath && (
        <pointLight
          position={[origin.x, 2, origin.z]}
          color="#3b82f6"
          intensity={1}
          distance={5}
        />
      )}
    </>
  )
}
