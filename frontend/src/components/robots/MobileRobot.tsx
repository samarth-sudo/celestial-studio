import { useRef, useState, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { RigidBody, type RapierRigidBody } from '@react-three/rapier'
import * as THREE from 'three'
import type { ComputedPath } from '../../types/PathPlanning'
import { getAlgorithmManager } from '../../services/AlgorithmManager'
import {
  FrameRateLimiter,
  clampVelocity,
  isValidVelocity,
  throttle
} from '../../utils/performanceUtils'

interface MobileRobotProps {
  onPositionUpdate?: (position: [number, number, number], rotation: [number, number, number]) => void
  onWaypointUpdate?: (currentWaypoint: number, totalWaypoints: number) => void
  onAlgorithmStatusUpdate?: (active: boolean, algorithmName?: string) => void
  path?: ComputedPath | null
  isPaused?: boolean
  obstacles?: Array<{ position: THREE.Vector3; radius: number }>
  robotId?: string
}

export default function MobileRobot({
  onPositionUpdate,
  onWaypointUpdate,
  onAlgorithmStatusUpdate,
  path,
  isPaused = false,
  obstacles = [],
  robotId = 'robot-1'
}: MobileRobotProps) {
  const bodyRef = useRef<RapierRigidBody>(null)
  const manager = getAlgorithmManager()

  // Performance: Frame rate limiter for algorithm execution (10 Hz instead of 60)
  const algorithmLimiter = useRef(new FrameRateLimiter(10))

  // Performance: Cache algorithm lookups to avoid querying manager 60 times/second
  const [cachedAlgorithms, setCachedAlgorithms] = useState<{
    obstacleAvoidance: any[]
    lastUpdate: number
  }>({ obstacleAvoidance: [], lastUpdate: 0 })

  // Demo waypoints (used when no path is provided)
  const demoWaypoints = useMemo(() => [
    new THREE.Vector3(0, 0.5, 0),
    new THREE.Vector3(3, 0.5, 0),
    new THREE.Vector3(3, 0.5, 3),
    new THREE.Vector3(0, 0.5, 3),
    new THREE.Vector3(0, 0.5, 0),
  ], [])

  // Use computed path waypoints if available, otherwise use demo waypoints
  const [waypoints, setWaypoints] = useState<THREE.Vector3[]>(demoWaypoints)
  const [currentWaypoint, setCurrentWaypoint] = useState(0)
  const speed = 2
  const MAX_VELOCITY = 5 // Safety: Maximum allowed velocity magnitude

  // Visual indicator: Track if algorithm is actively modifying velocity
  const [algorithmActive, setAlgorithmActive] = useState(false)

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

  // Performance: Update algorithm cache periodically (every 2 seconds instead of every frame)
  useEffect(() => {
    const updateCache = () => {
      const obstacleAvoidance = manager.getAlgorithmsByType(robotId, 'obstacle_avoidance')
      setCachedAlgorithms({
        obstacleAvoidance,
        lastUpdate: Date.now()
      })
    }

    // Initial update
    updateCache()

    // Update every 2 seconds
    const interval = setInterval(updateCache, 2000)
    return () => clearInterval(interval)
  }, [robotId, manager])

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
      // Calculate base velocity towards target
      direction.normalize()
      let velocity = direction.clone().multiplyScalar(speed)
      let algorithmModified = false

      // Performance: Only run obstacle avoidance at 10 Hz (not 60 Hz)
      // Safety: Use cached algorithms instead of querying every frame
      if (cachedAlgorithms.obstacleAvoidance.length > 0 &&
          obstacles.length > 0 &&
          algorithmLimiter.current.shouldExecute()) {
        try {
          // Use the first active obstacle avoidance algorithm
          const algo = cachedAlgorithms.obstacleAvoidance[0]

          // Create current velocity vector for algorithm
          const currentVel = { x: velocity.x, z: velocity.z }
          const currentPos = { x: position.x, z: position.z }
          const goal = { x: target.x, z: target.z }

          // Call obstacle avoidance algorithm
          // Try common function names
          const functionNames = ['calculateSafeVelocity', 'avoidObstacles', 'computeSafeVelocity']

          for (const funcName of functionNames) {
            try {
              const result = manager.executeAlgorithm(
                algo.id,
                funcName,
                currentPos,
                currentVel,
                obstacles,
                goal,
                speed
              )

              // Safety: Validate algorithm output before using it
              if (isValidVelocity(result)) {
                velocity = new THREE.Vector3(result.x, 0, result.z)

                // Safety: Clamp velocity to maximum allowed magnitude
                velocity = clampVelocity(velocity, MAX_VELOCITY)

                algorithmModified = true
                console.log(`🛡️ Obstacle avoidance applied: (${result.x.toFixed(2)}, ${result.z.toFixed(2)})`)
                break
              } else {
                console.warn(`⚠️ Invalid velocity from algorithm: ${JSON.stringify(result)}`)
              }
            } catch (error) {
              // Try next function name
              continue
            }
          }
        } catch (error: any) {
          console.warn(`⚠️ Obstacle avoidance failed, using direct path:`, error.message)
          algorithmModified = false
        }
      }

      // Update visual indicator state
      if (algorithmActive !== algorithmModified) {
        setAlgorithmActive(algorithmModified)

        // Notify parent component of algorithm status change
        if (onAlgorithmStatusUpdate) {
          const algoName = cachedAlgorithms.obstacleAvoidance[0]?.name
          onAlgorithmStatusUpdate(algorithmModified, algoName)
        }
      }

      // Safety: Always clamp velocity even if no algorithm was applied
      velocity = clampVelocity(velocity, MAX_VELOCITY)

      bodyRef.current.setLinvel({ x: velocity.x, y: 0, z: velocity.z }, true)

      // Rotate to face movement direction
      const angle = Math.atan2(velocity.x, velocity.z)
      bodyRef.current.setRotation({ x: 0, y: angle, z: 0, w: 1 }, true)
    }
  })

  return (
    <RigidBody ref={bodyRef} position={[0, 0.5, 0]} colliders="cuboid">
      <group>
        {/* Visual Indicator: Glowing outline when algorithm is active */}
        {algorithmActive && (
          <mesh scale={[1.3, 1.1, 1.1]}>
            <boxGeometry args={[1.2, 0.6, 0.8]} />
            <meshBasicMaterial
              color="#00ff88"
              transparent
              opacity={0.3}
              wireframe
            />
          </mesh>
        )}

        {/* Robot Body */}
        <mesh castShadow receiveShadow>
          <boxGeometry args={[1.2, 0.6, 0.8]} />
          <meshStandardMaterial
            color="#3498db"
            emissive={algorithmActive ? "#00ff88" : "#000000"}
            emissiveIntensity={algorithmActive ? 0.3 : 0}
          />
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
