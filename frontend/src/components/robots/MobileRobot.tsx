import { useRef, useState, useEffect } from 'react'
import { useFrame } from '@react-three/fiber'
import { RigidBody, type RapierRigidBody } from '@react-three/rapier'
import * as THREE from 'three'
import type { ComputedPath } from '../../types/PathPlanning'

interface MobileRobotProps {
  onPositionUpdate?: (position: [number, number, number], rotation: [number, number, number]) => void
  onWaypointUpdate?: (currentWaypoint: number, totalWaypoints: number) => void
  path?: ComputedPath | null
  isPaused?: boolean
}

export default function MobileRobot({ onPositionUpdate, onWaypointUpdate, path, isPaused = false }: MobileRobotProps) {
  const bodyRef = useRef<RapierRigidBody>(null)

  // Demo waypoints (used when no path is provided)
  const demoWaypoints = [
    new THREE.Vector3(0, 0.5, 0),
    new THREE.Vector3(3, 0.5, 0),
    new THREE.Vector3(3, 0.5, 3),
    new THREE.Vector3(0, 0.5, 3),
    new THREE.Vector3(0, 0.5, 0),
  ]

  // Use computed path waypoints if available, otherwise use demo waypoints
  const [waypoints, setWaypoints] = useState<THREE.Vector3[]>(demoWaypoints)
  const [currentWaypoint, setCurrentWaypoint] = useState(0)
  const speed = 2

  // Update waypoints when path changes
  useEffect(() => {
    if (path && path.waypoints && path.waypoints.length > 0) {
      // Convert path waypoints to include the robot's Y position
      const pathWithHeight = path.waypoints.map(wp =>
        new THREE.Vector3(wp.x, 0.5, wp.z)
      )
      setWaypoints(pathWithHeight)
      setCurrentWaypoint(0) // Reset to start of path
    } else {
      setWaypoints(demoWaypoints)
      setCurrentWaypoint(0)
    }
  }, [path])

  // Report waypoint progress to parent
  useEffect(() => {
    if (onWaypointUpdate) {
      onWaypointUpdate(currentWaypoint, waypoints.length)
    }
  }, [currentWaypoint, waypoints.length, onWaypointUpdate])

  useFrame(() => {
    if (!bodyRef.current) return

    const position = bodyRef.current.translation()
    const rotation = bodyRef.current.rotation()

    // Update parent component with current position and rotation
    if (onPositionUpdate) {
      // Convert quaternion to Euler angles
      const euler = new THREE.Euler().setFromQuaternion(
        new THREE.Quaternion(rotation.x, rotation.y, rotation.z, rotation.w)
      )
      onPositionUpdate(
        [position.x, position.y, position.z],
        [euler.x, euler.y, euler.z]
      )
    }

    // Stop movement if paused
    if (isPaused) {
      bodyRef.current.setLinvel({ x: 0, y: 0, z: 0 }, true)
      return
    }

    const target = waypoints[currentWaypoint]

    // Calculate direction to target
    const direction = new THREE.Vector3(
      target.x - position.x,
      0,
      target.z - position.z
    )

    const distance = direction.length()

    if (distance < 0.2) {
      // Reached waypoint, move to next
      setCurrentWaypoint((prev) => (prev + 1) % waypoints.length)
    } else {
      // Move towards target
      direction.normalize()
      const velocity = direction.multiplyScalar(speed)

      bodyRef.current.setLinvel({ x: velocity.x, y: 0, z: velocity.z }, true)

      // Rotate to face direction
      const angle = Math.atan2(direction.x, direction.z)
      bodyRef.current.setRotation({ x: 0, y: angle, z: 0, w: 1 }, true)
    }
  })

  return (
    <RigidBody ref={bodyRef} position={[0, 0.5, 0]} colliders="cuboid">
      <group>
        {/* Robot Body */}
        <mesh castShadow receiveShadow>
          <boxGeometry args={[1.2, 0.6, 0.8]} />
          <meshStandardMaterial color="#3498db" />
        </mesh>

        {/* Wheels */}
        <mesh position={[-0.5, -0.3, 0.5]} rotation={[Math.PI / 2, 0, 0]} castShadow>
          <cylinderGeometry args={[0.15, 0.15, 0.1, 16]} />
          <meshStandardMaterial color="#2c3e50" />
        </mesh>
        <mesh position={[0.5, -0.3, 0.5]} rotation={[Math.PI / 2, 0, 0]} castShadow>
          <cylinderGeometry args={[0.15, 0.15, 0.1, 16]} />
          <meshStandardMaterial color="#2c3e50" />
        </mesh>
        <mesh position={[-0.5, -0.3, -0.5]} rotation={[Math.PI / 2, 0, 0]} castShadow>
          <cylinderGeometry args={[0.15, 0.15, 0.1, 16]} />
          <meshStandardMaterial color="#2c3e50" />
        </mesh>
        <mesh position={[0.5, -0.3, -0.5]} rotation={[Math.PI / 2, 0, 0]} castShadow>
          <cylinderGeometry args={[0.15, 0.15, 0.1, 16]} />
          <meshStandardMaterial color="#2c3e50" />
        </mesh>

        {/* Sensor/Top piece */}
        <mesh position={[0, 0.5, 0]} castShadow>
          <boxGeometry args={[0.4, 0.3, 0.4]} />
          <meshStandardMaterial color="#e74c3c" />
        </mesh>
      </group>
    </RigidBody>
  )
}
