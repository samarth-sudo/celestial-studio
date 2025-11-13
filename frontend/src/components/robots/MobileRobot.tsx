import { useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { RigidBody, type RapierRigidBody } from '@react-three/rapier'
import * as THREE from 'three'

export default function MobileRobot() {
  const bodyRef = useRef<RapierRigidBody>(null)
  const [waypoints] = useState([
    new THREE.Vector3(0, 0.5, 0),
    new THREE.Vector3(3, 0.5, 0),
    new THREE.Vector3(3, 0.5, 3),
    new THREE.Vector3(0, 0.5, 3),
    new THREE.Vector3(0, 0.5, 0),
  ])
  const [currentWaypoint, setCurrentWaypoint] = useState(0)
  const speed = 2

  useFrame(() => {
    if (!bodyRef.current) return

    const position = bodyRef.current.translation()
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
