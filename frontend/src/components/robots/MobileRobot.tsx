import { useRef, useState, useEffect, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { RigidBody, type RapierRigidBody } from '@react-three/rapier'
import * as THREE from 'three'
import type { ComputedPath } from '../../types/PathPlanning'
import type { Vector2D } from '../../types/AlgorithmInterface'
import { getAlgorithmManager } from '../../services/AlgorithmManager'
import {
  FrameRateLimiter,
  clampVelocity,
  isValidVelocity
} from '../../utils/performanceUtils'

interface MobileRobotProps {
  onPositionUpdate?: (position: [number, number, number], rotation: [number, number, number]) => void
  onWaypointUpdate?: (currentWaypoint: number, totalWaypoints: number) => void
  onAlgorithmStatusUpdate?: (active: boolean, algorithmName?: string) => void
  onError?: (error: string) => void  // NEW: Error callback
  path?: ComputedPath | null
  isPaused?: boolean
  obstacles?: Array<{ position: THREE.Vector3; radius: number }>
  robotId?: string
}

export default function MobileRobot({
  onPositionUpdate,
  onWaypointUpdate,
  onAlgorithmStatusUpdate,
  onError,
  path,
  isPaused = false,
  obstacles = [],
  robotId = 'robot-1'
}: MobileRobotProps) {
  const bodyRef = useRef<RapierRigidBody>(null)
  const manager = getAlgorithmManager()

  // Performance: Frame rate limiter
  const algorithmLimiter = useRef(new FrameRateLimiter(10))

  // Cache algorithm lookups
  const [cachedAlgorithms, setCachedAlgorithms] = useState<{
    obstacleAvoidance: any[]
    lastUpdate: number
  }>({ obstacleAvoidance: [], lastUpdate: 0 })

  const demoWaypoints = useMemo(() => [
    new THREE.Vector3(0, 0.5, 0),
    new THREE.Vector3(3, 0.5, 0),
    new THREE.Vector3(3, 0.5, 3),
    new THREE.Vector3(0, 0.5, 3),
    new THREE.Vector3(0, 0.5, 0),
  ], [])

  const [waypoints, setWaypoints] = useState<THREE.Vector3[]>(demoWaypoints)
  const [currentWaypoint, setCurrentWaypoint] = useState(0)
  const [algorithmActive, setAlgorithmActive] = useState(false)
  const [lastError, setLastError] = useState<string | null>(null)

  const speed = 2
  const MAX_VELOCITY = 5

  // Update waypoints when path changes
  useEffect(() => {
    if (path && path.waypoints && path.waypoints.length > 0) {
      const pathWithHeight = path.waypoints.map(wp =>
        new THREE.Vector3(wp.x, 0.5, wp.z)
      )
      setWaypoints(pathWithHeight)
      setCurrentWaypoint(0)
    } else {
      setWaypoints(demoWaypoints)
      setCurrentWaypoint(0)
    }
  }, [path, demoWaypoints])

  // Report waypoint progress
  useEffect(() => {
    if (onWaypointUpdate) {
      onWaypointUpdate(currentWaypoint, waypoints.length)
    }
  }, [currentWaypoint, waypoints.length, onWaypointUpdate])

  // Update algorithm cache periodically
  useEffect(() => {
    const updateCache = () => {
      const obstacleAvoidance = manager.getAlgorithmsByType(robotId, 'obstacle_avoidance')
      setCachedAlgorithms({
        obstacleAvoidance,
        lastUpdate: Date.now()
      })
    }

    updateCache()
    const interval = setInterval(updateCache, 2000)
    return () => clearInterval(interval)
  }, [robotId, manager])

  useFrame(() => {
    if (!bodyRef.current) return

    const position = bodyRef.current.translation()
    const rotation = bodyRef.current.rotation()

    // Update parent component
    if (onPositionUpdate) {
      const euler = new THREE.Euler().setFromQuaternion(
        new THREE.Quaternion(rotation.x, rotation.y, rotation.z, rotation.w)
      )
      onPositionUpdate(
        [position.x, position.y, position.z],
        [euler.x, euler.y, euler.z]
      )
    }

    // Stop if paused
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
      // Reached waypoint
      setCurrentWaypoint((prev) => (prev + 1) % waypoints.length)
    } else {
      // Calculate base velocity
      direction.normalize()
      let velocity = direction.clone().multiplyScalar(speed)
      let algorithmModified = false
      let algorithmName = ''

      // Execute obstacle avoidance algorithm (FIXED VERSION)
      if (cachedAlgorithms.obstacleAvoidance.length > 0 &&
          obstacles.length > 0 &&
          algorithmLimiter.current.shouldExecute()) {

        // FIXED: Get first algorithm from array (was incorrectly treating array as single object)
        const algo = cachedAlgorithms.obstacleAvoidance[0]
        if (!algo) return  // Guard clause
        algorithmName = algo.name

        try {
          // Use standardized interface
          const currentPos: Vector2D = { x: position.x, z: position.z }
          const currentVel: Vector2D = { x: velocity.x, z: velocity.z }
          const goal: Vector2D = { x: target.x, z: target.z }

          // Call algorithm with standardized function name
          const result = manager.executeObstacleAvoidance(
            algo.id,
            currentPos,
            currentVel,
            obstacles,
            goal,
            speed
          )

          // Validate result
          if (isValidVelocity(result)) {
            velocity = new THREE.Vector3(result.x, 0, result.z)
            velocity = clampVelocity(velocity, MAX_VELOCITY)
            algorithmModified = true

            // Clear previous errors
            if (lastError) {
              setLastError(null)
            }

            console.log(`🛡️ Obstacle avoidance: (${result.x.toFixed(2)}, ${result.z.toFixed(2)})`)
          } else {
            throw new Error(`Invalid velocity: ${JSON.stringify(result)}`)
          }

        } catch (error: any) {
          const errorMsg = `Algorithm '${algo.name}' failed: ${error.message}`
          console.error(`❌ ${errorMsg}`)

          // Report error to parent (only once per unique error)
          if (errorMsg !== lastError) {
            setLastError(errorMsg)
            if (onError) {
              onError(errorMsg)
            }
          }

          algorithmModified = false
        }
      }

      // Update visual indicator
      if (algorithmActive !== algorithmModified) {
        setAlgorithmActive(algorithmModified)

        if (onAlgorithmStatusUpdate) {
          onAlgorithmStatusUpdate(algorithmModified, algorithmName)
        }
      }

      // Safety: Clamp velocity
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

        {/* Error Indicator: Red glow when algorithm failed */}
        {lastError && (
          <mesh scale={[1.35, 1.15, 1.15]}>
            <boxGeometry args={[1.2, 0.6, 0.8]} />
            <meshBasicMaterial
              color="#ff0000"
              transparent
              opacity={0.2}
              wireframe
            />
          </mesh>
        )}

        {/* Robot Body */}
        <mesh castShadow receiveShadow>
          <boxGeometry args={[1.2, 0.6, 0.8]} />
          <meshStandardMaterial
            color={lastError ? "#e74c3c" : "#3498db"}
            emissive={algorithmActive ? "#00ff88" : lastError ? "#ff0000" : "#000000"}
            emissiveIntensity={algorithmActive || lastError ? 0.3 : 0}
          />
        </mesh>

        {/* Wheels */}
        {[
          [-0.5, -0.3, 0.5],
          [0.5, -0.3, 0.5],
          [-0.5, -0.3, -0.5],
          [0.5, -0.3, -0.5]
        ].map((pos, i) => (
          <mesh
            key={i}
            position={pos as [number, number, number]}
            rotation={[Math.PI / 2, 0, 0]}
            castShadow
          >
            <cylinderGeometry args={[0.15, 0.15, 0.1, 16]} />
            <meshStandardMaterial color="#2c3e50" />
          </mesh>
        ))}

        {/* Sensor/Top piece */}
        <mesh position={[0, 0.5, 0]} castShadow>
          <boxGeometry args={[0.4, 0.3, 0.4]} />
          <meshStandardMaterial color="#e74c3c" />
        </mesh>
      </group>
    </RigidBody>
  )
}
