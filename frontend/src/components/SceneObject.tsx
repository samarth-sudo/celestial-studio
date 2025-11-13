import { RigidBody } from '@react-three/rapier'
import { Box, Cylinder, Sphere } from '@react-three/drei'

interface SceneObjectProps {
  object: {
    type: string
    id: string
    position: number[]
    size?: number[]
    color: string
    radius?: number
    height?: number
    physics?: {
      mass: number
      friction: number
      static?: boolean
      restitution?: number
    }
    interactive?: boolean
    opacity?: number
  }
}

export default function SceneObject({ object }: SceneObjectProps) {
  const position: [number, number, number] = [
    object.position[0],
    object.position[1],
    object.position[2]
  ]

  // Determine physics body type
  const bodyType = object.physics?.static ? 'fixed' : 'dynamic'

  // Render different object types
  if (object.type === 'box') {
    const size: [number, number, number] = object.size
      ? [object.size[0], object.size[1], object.size[2]]
      : [1, 1, 1]

    return (
      <RigidBody
        position={position}
        type={bodyType}
        friction={object.physics?.friction || 0.5}
        restitution={object.physics?.restitution || 0.3}
        mass={object.physics?.mass || 1}
      >
        <Box args={size} castShadow receiveShadow>
          <meshStandardMaterial
            color={object.color}
            transparent={!!object.opacity}
            opacity={object.opacity || 1}
          />
        </Box>
      </RigidBody>
    )
  }

  if (object.type === 'cylinder' || object.type === 'obstacle') {
    const radius = object.radius || (object.size ? object.size[0] : 0.5)
    const height = object.height || (object.size ? object.size[1] : 1)

    return (
      <RigidBody
        position={position}
        type={bodyType}
        friction={object.physics?.friction || 0.7}
        mass={object.physics?.mass || 10}
      >
        <Cylinder args={[radius, radius, height, 16]} castShadow receiveShadow>
          <meshStandardMaterial color={object.color} />
        </Cylinder>
      </RigidBody>
    )
  }

  if (object.type === 'sphere') {
    const radius = object.radius || 0.5

    return (
      <RigidBody
        position={position}
        type={bodyType}
        friction={object.physics?.friction || 0.5}
        restitution={object.physics?.restitution || 0.3}
        mass={object.physics?.mass || 1}
      >
        <Sphere args={[radius, 32, 32]} castShadow receiveShadow>
          <meshStandardMaterial
            color={object.color}
            transparent={!!object.opacity}
            opacity={object.opacity || 1}
          />
        </Sphere>
      </RigidBody>
    )
  }

  if (object.type === 'person') {
    const radius = object.radius || 0.3
    const height = object.height || 1.8

    return (
      <RigidBody
        position={position}
        type={bodyType}
        friction={object.physics?.friction || 0.6}
        mass={object.physics?.mass || 70}
      >
        <Cylinder args={[radius, radius, height, 16]} castShadow receiveShadow>
          <meshStandardMaterial color={object.color} />
        </Cylinder>
      </RigidBody>
    )
  }

  if (object.type === 'shelf') {
    const size: [number, number, number] = object.size
      ? [object.size[0], object.size[1], object.size[2]]
      : [2, 3, 0.5]

    return (
      <RigidBody
        position={position}
        type="fixed"
        friction={object.physics?.friction || 0.8}
      >
        <Box args={size} castShadow receiveShadow>
          <meshStandardMaterial color={object.color} />
        </Box>
      </RigidBody>
    )
  }

  if (object.type === 'pallet') {
    const size: [number, number, number] = object.size
      ? [object.size[0], object.size[1], object.size[2]]
      : [1.2, 0.15, 0.8]

    return (
      <RigidBody
        position={position}
        type="fixed"
        friction={object.physics?.friction || 0.7}
      >
        <Box args={size} castShadow receiveShadow>
          <meshStandardMaterial color={object.color} />
        </Box>
      </RigidBody>
    )
  }

  if (object.type === 'target_zone') {
    const size: [number, number, number] = object.size
      ? [object.size[0], object.size[1], object.size[2]]
      : [1.5, 0.1, 1.5]

    return (
      <mesh position={position}>
        <boxGeometry args={size} />
        <meshStandardMaterial
          color={object.color}
          transparent
          opacity={object.opacity || 0.5}
          emissive={object.color}
          emissiveIntensity={0.2}
        />
      </mesh>
    )
  }

  if (object.type === 'waypoint') {
    const radius = object.radius || 0.3

    return (
      <mesh position={position}>
        <cylinderGeometry args={[radius, radius, 0.1, 16]} />
        <meshStandardMaterial
          color={object.color}
          emissive={object.color}
          emissiveIntensity={0.5}
        />
      </mesh>
    )
  }

  // Default fallback - render as a box
  return (
    <RigidBody position={position} type={bodyType}>
      <Box args={[0.5, 0.5, 0.5]} castShadow receiveShadow>
        <meshStandardMaterial color={object.color} />
      </Box>
    </RigidBody>
  )
}
