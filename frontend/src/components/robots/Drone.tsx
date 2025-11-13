import { useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

export default function Drone() {
  const droneRef = useRef<THREE.Group>(null)
  const propeller1Ref = useRef<THREE.Mesh>(null)
  const propeller2Ref = useRef<THREE.Mesh>(null)
  const propeller3Ref = useRef<THREE.Mesh>(null)
  const propeller4Ref = useRef<THREE.Mesh>(null)

  const [time, setTime] = useState(0)

  useFrame((_, delta) => {
    setTime((t) => t + delta)

    // Rotate propellers (fast spin)
    const propellers = [propeller1Ref, propeller2Ref, propeller3Ref, propeller4Ref]
    propellers.forEach((propRef) => {
      if (propRef.current) {
        propRef.current.rotation.y += delta * 30 // Fast rotation
      }
    })

    // Drone flight path (figure-8 pattern)
    if (droneRef.current) {
      const t = time * 0.5
      const x = Math.sin(t) * 3
      const z = Math.sin(t * 2) * 1.5
      const y = 2 + Math.sin(t * 3) * 0.3 // Hovering with gentle bobbing

      droneRef.current.position.set(x, y, z)

      // Tilt based on movement
      const tiltX = Math.cos(t * 2) * 0.1
      const tiltZ = Math.cos(t) * 0.1
      droneRef.current.rotation.set(tiltX, time * 0.3, tiltZ)
    }
  })

  return (
    <group ref={droneRef} position={[0, 2, 0]}>
      {/* Central body */}
      <mesh castShadow>
        <boxGeometry args={[0.6, 0.2, 0.6]} />
        <meshStandardMaterial color="#2c3e50" />
      </mesh>

      {/* Arms */}
      <mesh position={[0, 0, 0]} castShadow>
        <boxGeometry args={[1.4, 0.05, 0.05]} />
        <meshStandardMaterial color="#34495e" />
      </mesh>
      <mesh position={[0, 0, 0]} rotation={[0, Math.PI / 2, 0]} castShadow>
        <boxGeometry args={[1.4, 0.05, 0.05]} />
        <meshStandardMaterial color="#34495e" />
      </mesh>

      {/* Motors at arm ends */}
      <mesh position={[0.7, 0, 0.7]} castShadow>
        <cylinderGeometry args={[0.08, 0.08, 0.15, 16]} />
        <meshStandardMaterial color="#e74c3c" />
      </mesh>
      <mesh position={[-0.7, 0, 0.7]} castShadow>
        <cylinderGeometry args={[0.08, 0.08, 0.15, 16]} />
        <meshStandardMaterial color="#e74c3c" />
      </mesh>
      <mesh position={[0.7, 0, -0.7]} castShadow>
        <cylinderGeometry args={[0.08, 0.08, 0.15, 16]} />
        <meshStandardMaterial color="#e74c3c" />
      </mesh>
      <mesh position={[-0.7, 0, -0.7]} castShadow>
        <cylinderGeometry args={[0.08, 0.08, 0.15, 16]} />
        <meshStandardMaterial color="#e74c3c" />
      </mesh>

      {/* Propellers */}
      <mesh ref={propeller1Ref} position={[0.7, 0.1, 0.7]} rotation={[0, 0, 0]}>
        <boxGeometry args={[0.4, 0.02, 0.1]} />
        <meshStandardMaterial color="#3498db" opacity={0.7} transparent />
      </mesh>
      <mesh ref={propeller2Ref} position={[-0.7, 0.1, 0.7]} rotation={[0, 0, 0]}>
        <boxGeometry args={[0.4, 0.02, 0.1]} />
        <meshStandardMaterial color="#3498db" opacity={0.7} transparent />
      </mesh>
      <mesh ref={propeller3Ref} position={[0.7, 0.1, -0.7]} rotation={[0, 0, 0]}>
        <boxGeometry args={[0.4, 0.02, 0.1]} />
        <meshStandardMaterial color="#3498db" opacity={0.7} transparent />
      </mesh>
      <mesh ref={propeller4Ref} position={[-0.7, 0.1, -0.7]} rotation={[0, 0, 0]}>
        <boxGeometry args={[0.4, 0.02, 0.1]} />
        <meshStandardMaterial color="#3498db" opacity={0.7} transparent />
      </mesh>

      {/* Camera gimbal */}
      <mesh position={[0, -0.15, 0]} castShadow>
        <sphereGeometry args={[0.12, 16, 16]} />
        <meshStandardMaterial color="#95a5a6" />
      </mesh>
    </group>
  )
}
