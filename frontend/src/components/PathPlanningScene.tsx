import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid } from '@react-three/drei'
import * as THREE from 'three'
import PathVisualizer from './PathVisualizer'
import SceneInteraction from './SceneInteraction'
import type { InteractionMode, Obstacle, ComputedPath } from '../types/PathPlanning'

interface PathPlanningSceneProps {
  origin: THREE.Vector3 | null
  destination: THREE.Vector3 | null
  obstacles: Obstacle[]
  computedPath: ComputedPath | null
  mode: InteractionMode
  currentObstacleType: Obstacle['type']
  onOriginSet: (position: THREE.Vector3) => void
  onDestinationSet: (position: THREE.Vector3) => void
  onObstaclePlaced: (obstacle: Omit<Obstacle, 'id'>) => void
  onObstacleSelected?: (obstacleId: string | null) => void
}

export default function PathPlanningScene({
  origin,
  destination,
  obstacles,
  computedPath,
  mode,
  currentObstacleType,
  onOriginSet,
  onDestinationSet,
  onObstaclePlaced,
  onObstacleSelected
}: PathPlanningSceneProps) {
  return (
    <Canvas
      camera={{ position: [15, 15, 15], fov: 50 }}
      style={{ background: '#0a0a0a' }}
    >
      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
      <directionalLight position={[-10, 10, -5]} intensity={0.5} />

      {/* Ground plane */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
        <planeGeometry args={[20, 20]} />
        <meshStandardMaterial color="#1a1a1a" />
      </mesh>

      {/* Grid helper */}
      <Grid
        args={[20, 20]}
        cellSize={1}
        cellThickness={0.5}
        cellColor="#333333"
        sectionSize={5}
        sectionThickness={1}
        sectionColor="#444444"
        fadeDistance={30}
        fadeStrength={1}
        followCamera={false}
      />

      {/* Path visualization */}
      <PathVisualizer
        origin={origin}
        destination={destination}
        path={computedPath}
        obstacles={obstacles}
        showGrid={false}
      />

      {/* Scene interaction */}
      <SceneInteraction
        mode={mode}
        onOriginSet={onOriginSet}
        onDestinationSet={onDestinationSet}
        onObstaclePlaced={onObstaclePlaced}
        onObstacleSelected={onObstacleSelected}
        obstacles={obstacles}
        currentObstacleType={currentObstacleType}
      />

      {/* Camera controls */}
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={5}
        maxDistance={50}
        maxPolarAngle={Math.PI / 2}
      />
    </Canvas>
  )
}
