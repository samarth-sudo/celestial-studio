import { RigidBody } from '@react-three/rapier'
import type { Obstacle } from '../types/PathPlanning'

interface SimulatedObstacleProps {
  obstacle: Obstacle
}

export default function SimulatedObstacle({ obstacle }: SimulatedObstacleProps) {
  const { position, size, type, color, radius, rotation } = obstacle

  const renderGeometry = () => {
    switch (type) {
      case 'box':
      case 'furniture':
      case 'wall':
        return (
          <boxGeometry args={[size.x, size.y, size.z]} />
        )
      case 'cylinder':
        return (
          <cylinderGeometry args={[radius || size.x / 2, radius || size.x / 2, size.y, 16]} />
        )
      case 'sphere':
        return (
          <sphereGeometry args={[radius || size.x / 2, 16, 16]} />
        )
      case 'plant':
        // Simple representation: cylinder for pot + sphere for leaves
        return (
          <>
            <cylinderGeometry args={[radius || size.x / 2, radius || size.x / 2, size.y * 0.4, 12]} />
          </>
        )
      default:
        return <boxGeometry args={[size.x, size.y, size.z]} />
    }
  }

  const getMaterial = () => {
    const obstacleColor = color || getDefaultColor(type)
    return (
      <meshStandardMaterial
        color={obstacleColor}
        roughness={0.7}
        metalness={0.1}
      />
    )
  }

  return (
    <RigidBody type="fixed" colliders="cuboid">
      <mesh
        position={[position.x, position.y + size.y / 2, position.z]}
        rotation={rotation ? [rotation.x, rotation.y, rotation.z] : undefined}
        castShadow
        receiveShadow
      >
        {renderGeometry()}
        {getMaterial()}
      </mesh>

      {/* Add a leaf sphere for plants */}
      {type === 'plant' && (
        <mesh
          position={[position.x, position.y + size.y * 0.7, position.z]}
          castShadow
        >
          <sphereGeometry args={[size.x * 0.6, 12, 12]} />
          <meshStandardMaterial color="#2d5016" roughness={0.8} />
        </mesh>
      )}
    </RigidBody>
  )
}

function getDefaultColor(type: string): string {
  const colorMap: Record<string, string> = {
    box: '#8b4513',
    cylinder: '#a9a9a9',
    sphere: '#ff6b6b',
    plant: '#3d5a3d',
    furniture: '#6d4c41',
    wall: '#7f8c8d',
    'custom-area': '#95a5a6'
  }
  return colorMap[type] || '#95a5a6'
}
