import { useRef, useMemo, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { Mesh, Euler, Vector3 } from 'three'

interface URDFRobotProps {
  sceneConfig: any
  position?: [number, number, number]
  scale?: number
}

interface LinkConfig {
  name: string
  geometry: {
    type: string
    args: number[]
    filename?: string
    scale?: number[]
  }
  material: {
    color: string
    metalness: number
    roughness: number
  }
  position: number[]
  rotation: number[]
  castShadow: boolean
  receiveShadow: boolean
}

interface JointConfig {
  name: string
  type: string
  parent_link: string
  child_link: string
  position: number[]
  rotation: number[]
  axis: number[]
  limits: {
    lower: number
    upper: number
    effort: number
    velocity: number
  }
  current_angle: number
}

function RobotLink({
  link,
  jointAngle = 0,
  jointAxis = [1, 0, 0]
}: {
  link: LinkConfig
  jointAngle?: number
  jointAxis?: number[]
}) {
  const meshRef = useRef<Mesh>(null)

  // Apply joint rotation
  useMemo(() => {
    if (meshRef.current && jointAngle !== 0) {
      const axis = new Vector3(...jointAxis).normalize()
      meshRef.current.rotateOnAxis(axis, jointAngle)
    }
  }, [jointAngle, jointAxis])

  const renderGeometry = () => {
    const { type, args } = link.geometry

    switch (type) {
      case 'box':
        return <boxGeometry args={args as [number, number, number]} />

      case 'cylinder':
        return <cylinderGeometry args={args as [number, number, number, number]} />

      case 'sphere':
        return <sphereGeometry args={args as [number, number, number]} />

      case 'mesh':
        // TODO: Load external mesh files (.stl, .dae, .obj)
        // For now, use a placeholder box
        console.warn(`External mesh loading not yet implemented: ${link.geometry.filename}`)
        return <boxGeometry args={[0.1, 0.1, 0.1]} />

      default:
        return <boxGeometry args={[0.1, 0.1, 0.1]} />
    }
  }

  return (
    <mesh
      ref={meshRef}
      position={new Vector3(...link.position)}
      rotation={new Euler(...link.rotation)}
      castShadow={link.castShadow}
      receiveShadow={link.receiveShadow}
    >
      {renderGeometry()}
      <meshStandardMaterial
        color={link.material.color}
        metalness={link.material.metalness}
        roughness={link.material.roughness}
      />
    </mesh>
  )
}

export default function URDFRobot({
  sceneConfig,
  position = [0, 0, 0],
  scale = 1
}: URDFRobotProps) {
  const groupRef = useRef<THREE.Group>(null)
  const [jointAngles, setJointAngles] = useState<Record<string, number>>({})

  // Extract robot data
  const robot = sceneConfig?.robot
  const links: LinkConfig[] = robot?.links || []
  const kinematicTree = robot?.kinematic_tree || {}

  // Memoize joints array to prevent unnecessary recalculations
  const joints: JointConfig[] = useMemo(() => {
    return robot?.joints || []
  }, [robot?.joints])

  // Build link-to-joint mapping - MUST be before conditional return
  const linkJoints = useMemo(() => {
    const mapping: Record<string, JointConfig> = {}

    joints.forEach(joint => {
      mapping[joint.child_link] = joint
    })

    return mapping
  }, [joints])

  // Animate revolute joints (demo - oscillate joints) - MUST be before conditional return
  useFrame(({ clock }) => {
    if (!robot) return // Early exit inside hook is OK

    const time = clock.getElapsedTime()
    const newAngles: Record<string, number> = {}

    joints.forEach(joint => {
      if (joint.type === 'revolute' || joint.type === 'continuous') {
        const { lower, upper } = joint.limits

        if (joint.type === 'continuous') {
          // Continuous joint - full rotation
          newAngles[joint.name] = (time * 0.5) % (Math.PI * 2)
        } else {
          // Revolute joint - oscillate within limits
          const range = upper - lower
          const midpoint = (upper + lower) / 2
          newAngles[joint.name] = midpoint + Math.sin(time) * (range / 4)
        }
      }
    })

    setJointAngles(newAngles)
  })

  // Conditional return AFTER all hooks
  if (!robot) {
    console.error('Invalid scene config: missing robot data')
    return null
  }

  // Render kinematic tree recursively
  const renderKinematicTree = (linkName: string) => {
    const link = links.find(l => l.name === linkName)
    if (!link) return null

    const joint = linkJoints[linkName]
    const jointAngle = joint ? jointAngles[joint.name] || 0 : 0
    const jointAxis = joint ? joint.axis : [1, 0, 0]

    // Get children from kinematic tree
    const treeNode = kinematicTree[linkName]
    const children = treeNode?.children || {}

    return (
      <group key={linkName}>
        <RobotLink
          link={link}
          jointAngle={jointAngle}
          jointAxis={jointAxis}
        />

        {/* Render child links */}
        {Object.keys(children).map(childLinkName =>
          renderKinematicTree(childLinkName)
        )}
      </group>
    )
  }

  return (
    <group
      ref={groupRef}
      position={position}
      scale={scale}
    >
      {/* Find and render from base link */}
      {robot.base_link && renderKinematicTree(robot.base_link)}

      {/* Fallback: render all links if no kinematic tree */}
      {!robot.base_link && links.map(link => {
        const joint = linkJoints[link.name]
        const jointAngle = joint ? jointAngles[joint.name] || 0 : 0

        return (
          <RobotLink
            key={link.name}
            link={link}
            jointAngle={jointAngle}
            jointAxis={joint ? joint.axis : [1, 0, 0]}
          />
        )
      })}

      {/* Ground plane for reference */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.5, 0]} receiveShadow>
        <planeGeometry args={[10, 10]} />
        <meshStandardMaterial color="#1a1a1a" />
      </mesh>
    </group>
  )
}
