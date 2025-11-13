import { useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

export default function RoboticArm() {
  const baseRef = useRef<THREE.Group>(null)
  const joint1Ref = useRef<THREE.Group>(null)
  const joint2Ref = useRef<THREE.Group>(null)
  const joint3Ref = useRef<THREE.Group>(null)
  const gripperRef = useRef<THREE.Group>(null)

  const [time, setTime] = useState(0)

  useFrame((_, delta) => {
    setTime((t) => t + delta)

    // Animated pick-and-place sequence
    const cycle = (time % 8) / 8 // 8 second cycle

    if (joint1Ref.current) {
      // Base rotation
      joint1Ref.current.rotation.y = Math.sin(cycle * Math.PI * 2) * 0.5
    }

    if (joint2Ref.current) {
      // Shoulder joint
      joint2Ref.current.rotation.z = -Math.PI / 4 + Math.sin(cycle * Math.PI * 2) * 0.3
    }

    if (joint3Ref.current) {
      // Elbow joint
      joint3Ref.current.rotation.z = Math.PI / 3 + Math.cos(cycle * Math.PI * 2) * 0.4
    }

    if (gripperRef.current) {
      // Gripper open/close
      const gripperState = cycle < 0.3 || cycle > 0.7 ? 0.2 : 0.05
      gripperRef.current.scale.x = gripperState
    }
  })

  return (
    <group ref={baseRef} position={[0, 0, 0]}>
      {/* Base */}
      <mesh position={[0, 0.2, 0]} castShadow>
        <cylinderGeometry args={[0.4, 0.5, 0.4, 16]} />
        <meshStandardMaterial color="#7f8c8d" />
      </mesh>

      {/* Joint 1 (Base rotation) */}
      <group ref={joint1Ref} position={[0, 0.4, 0]}>
        <mesh castShadow>
          <boxGeometry args={[0.3, 0.3, 0.3]} />
          <meshStandardMaterial color="#34495e" />
        </mesh>

        {/* Link 1 */}
        <mesh position={[0, 0.5, 0]} castShadow>
          <boxGeometry args={[0.15, 1, 0.15]} />
          <meshStandardMaterial color="#e74c3c" />
        </mesh>

        {/* Joint 2 (Shoulder) */}
        <group ref={joint2Ref} position={[0, 1, 0]}>
          <mesh castShadow>
            <sphereGeometry args={[0.2, 16, 16]} />
            <meshStandardMaterial color="#34495e" />
          </mesh>

          {/* Link 2 */}
          <mesh position={[0, 0.4, 0]} castShadow>
            <boxGeometry args={[0.12, 0.8, 0.12]} />
            <meshStandardMaterial color="#3498db" />
          </mesh>

          {/* Joint 3 (Elbow) */}
          <group ref={joint3Ref} position={[0, 0.8, 0]}>
            <mesh castShadow>
              <sphereGeometry args={[0.15, 16, 16]} />
              <meshStandardMaterial color="#34495e" />
            </mesh>

            {/* Link 3 (forearm) */}
            <mesh position={[0, 0.3, 0]} castShadow>
              <boxGeometry args={[0.1, 0.6, 0.1]} />
              <meshStandardMaterial color="#2ecc71" />
            </mesh>

            {/* Gripper */}
            <group ref={gripperRef} position={[0, 0.6, 0]}>
              <mesh position={[-0.05, 0, 0]} castShadow>
                <boxGeometry args={[0.08, 0.2, 0.05]} />
                <meshStandardMaterial color="#95a5a6" />
              </mesh>
              <mesh position={[0.05, 0, 0]} castShadow>
                <boxGeometry args={[0.08, 0.2, 0.05]} />
                <meshStandardMaterial color="#95a5a6" />
              </mesh>
            </group>
          </group>
        </group>
      </group>

      {/* Target object (cube to pick) */}
      <mesh position={[1.5, 0.15, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.3, 0.3, 0.3]} />
        <meshStandardMaterial color="#f39c12" />
      </mesh>
    </group>
  )
}
