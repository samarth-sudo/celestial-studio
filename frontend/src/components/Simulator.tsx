import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid, Environment } from '@react-three/drei'
import { Physics, RigidBody } from '@react-three/rapier'
import MobileRobot from './robots/MobileRobot'
import RoboticArm from './robots/RoboticArm'
import Drone from './robots/Drone'
import SceneObject from './SceneObject'
import './Simulator.css'

interface SimulatorProps {
  sceneConfig?: any  // Scene config from conversational chat
}

export default function Simulator({ sceneConfig }: SimulatorProps) {
  // Determine if we have any content to render
  const hasContent = sceneConfig

  return (
    <div className="simulator">
      {!hasContent ? (
        <div className="simulator-empty">
          <div className="empty-message">
            <h2>🎮 3D Simulator</h2>
            <p>Generate a robot or create a simulation to see it in action!</p>
          </div>
        </div>
      ) : (
        <Canvas
          camera={{ position: [5, 5, 5], fov: 50 }}
          shadows
        >
          <color attach="background" args={['#1a1a1a']} />

          {/* Lighting */}
          <ambientLight intensity={0.5} />
          <directionalLight
            position={[10, 10, 5]}
            intensity={1}
            castShadow
            shadow-mapSize={[1024, 1024]}
          />

          <Physics gravity={[0, -9.81, 0]}>
            {/* Ground */}
            <Grid
              args={sceneConfig?.environment?.floor?.size || [20, 20]}
              cellSize={0.5}
              cellThickness={0.5}
              cellColor={'#6f6f6f'}
              sectionSize={2}
              sectionThickness={1}
              sectionColor={'#9d4b4b'}
              fadeDistance={50}
              fadeStrength={1}
              position={[0, 0, 0]}
            />

            {/* Render environment walls if specified */}
            {sceneConfig?.environment?.walls && (() => {
              const floorSize = sceneConfig.environment?.floor?.size || [20, 20]
              const wallHeight = sceneConfig.environment?.wallHeight || 5
              const wallColor = sceneConfig.environment?.wallColor || '#A0A0A0'
              const wallThickness = 0.2

              return (
                <>
                  {/* Front wall (positive Z) */}
                  <RigidBody type="fixed" position={[0, wallHeight / 2, floorSize[1] / 2]}>
                    <mesh castShadow receiveShadow>
                      <boxGeometry args={[floorSize[0], wallHeight, wallThickness]} />
                      <meshStandardMaterial color={wallColor} />
                    </mesh>
                  </RigidBody>

                  {/* Back wall (negative Z) */}
                  <RigidBody type="fixed" position={[0, wallHeight / 2, -floorSize[1] / 2]}>
                    <mesh castShadow receiveShadow>
                      <boxGeometry args={[floorSize[0], wallHeight, wallThickness]} />
                      <meshStandardMaterial color={wallColor} />
                    </mesh>
                  </RigidBody>

                  {/* Left wall (negative X) */}
                  <RigidBody type="fixed" position={[-floorSize[0] / 2, wallHeight / 2, 0]}>
                    <mesh castShadow receiveShadow>
                      <boxGeometry args={[wallThickness, wallHeight, floorSize[1]]} />
                      <meshStandardMaterial color={wallColor} />
                    </mesh>
                  </RigidBody>

                  {/* Right wall (positive X) */}
                  <RigidBody type="fixed" position={[floorSize[0] / 2, wallHeight / 2, 0]}>
                    <mesh castShadow receiveShadow>
                      <boxGeometry args={[wallThickness, wallHeight, floorSize[1]]} />
                      <meshStandardMaterial color={wallColor} />
                    </mesh>
                  </RigidBody>
                </>
              )
            })()}

            {/* Render scene from conversational chat */}
            {sceneConfig && (
              <>
                {/* Render all scene objects */}
                {sceneConfig.objects?.map((obj: any, index: number) => (
                  <SceneObject key={obj.id || `obj-${index}`} object={obj} />
                ))}

                {/* Render waypoints if present */}
                {sceneConfig.waypoints?.map((waypoint: any) => (
                  <SceneObject key={waypoint.id} object={waypoint} />
                ))}

                {/* Render task markers if present */}
                {sceneConfig.task_markers?.map((marker: any) => (
                  <SceneObject key={marker.id} object={marker} />
                ))}

                {/* Render robot from scene config */}
                {sceneConfig.robot?.type === 'mobile_robot' && (
                  <MobileRobot />
                )}
                {sceneConfig.robot?.type === 'robotic_arm' && (
                  <RoboticArm />
                )}
                {sceneConfig.robot?.type === 'quadcopter' && (
                  <Drone />
                )}
              </>
            )}
          </Physics>

          {/* Camera controls */}
          <OrbitControls makeDefault />

          {/* Environment lighting */}
          <Environment preset="city" />
        </Canvas>
      )}
    </div>
  )
}
