import { useState, useRef, useEffect } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { OrbitControls, Grid, Environment } from '@react-three/drei'
import { Physics, RigidBody } from '@react-three/rapier'
import { Vector3, Raycaster, Plane, Vector2 } from 'three'
import * as THREE from 'three'
import MobileRobot from './robots/MobileRobot'
import RoboticArm from './robots/RoboticArm'
import Drone from './robots/Drone'
import SceneObject from './SceneObject'
import URDFRobot from './URDFRobot'
import IsaacSimulationPanel from './IsaacSimulationPanel'
import SceneEditor, { type EditorTool } from './SceneEditor'
import PathVisualizer from './PathVisualizer'
import CameraViewWindow from './CameraViewWindow'
import ObjectPalette from './ObjectPalette'
import PathExecutionPanel from './PathExecutionPanel'
import AlgorithmControls from './AlgorithmControls'
import { getAlgorithmManager } from '../services/AlgorithmManager'
import { AStarPathPlanner } from '../utils/pathPlanning'
import type { ComputedPath } from '../types/PathPlanning'
import './Simulator.css'

type SimulationMode = 'rapier' | 'isaac'

interface SimulatorProps {
  sceneConfig?: any
  onSceneChange?: (sceneConfig: any) => void
}

// ClickHandler component for proper 3D raycasting
interface ClickHandlerProps {
  activeTool: EditorTool
  onAddObject: (type: string, position: Vector3) => void
}

function ClickHandler({ activeTool, onAddObject }: ClickHandlerProps) {
  const { camera, raycaster, gl } = useThree()

  const handleClick = (event: any) => {
    // Only process clicks when a tool is active (not 'select')
    if (activeTool === 'select' || activeTool === 'delete') return

    // Stop event propagation to prevent OrbitControls from interfering
    event.stopPropagation()

    // Get canvas position and size
    const rect = gl.domElement.getBoundingClientRect()

    // Get normalized device coordinates (-1 to 1) relative to canvas
    const mouse = new Vector2()
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

    // Update raycaster with camera and mouse position
    raycaster.setFromCamera(mouse, camera)

    // Define ground plane at y = 0
    const groundPlane = new Plane(new Vector3(0, 1, 0), 0)

    // Find intersection with ground plane
    const intersectionPoint = new Vector3()
    raycaster.ray.intersectPlane(groundPlane, intersectionPoint)

    if (intersectionPoint) {
      // Determine object type based on active tool
      let objectType = ''
      if (activeTool === 'add_origin') objectType = 'origin'
      else if (activeTool === 'add_destination') objectType = 'destination'
      else if (activeTool === 'add_box') objectType = 'box'
      else if (activeTool === 'add_cylinder') objectType = 'cylinder'

      if (objectType) {
        onAddObject(objectType, intersectionPoint)
      }
    }
  }

  // Return an invisible plane that covers the entire scene for click detection
  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, 0, 0]}
      onClick={handleClick}
      visible={false}
    >
      <planeGeometry args={[100, 100]} />
      <meshBasicMaterial transparent opacity={0} />
    </mesh>
  )
}

export default function Simulator({ sceneConfig, onSceneChange }: SimulatorProps) {
  const [mode, setMode] = useState<SimulationMode>('rapier')
  const [showIsaacPanel, setShowIsaacPanel] = useState(false)
  const [activeTool, setActiveTool] = useState<EditorTool>('select')
  const [isPlaying, setIsPlaying] = useState(false)
  const [simulationSpeed, setSimulationSpeed] = useState(1)
  const [editableObjects, setEditableObjects] = useState<any[]>([])
  const [undoStack, setUndoStack] = useState<any[]>([])

  // Path planning state
  const [computedPath, setComputedPath] = useState<ComputedPath | null>(null)
  const [pathPlanner, setPathPlanner] = useState<AStarPathPlanner | null>(null)
  const [activeAlgorithmId, setActiveAlgorithmId] = useState<string>('builtin-astar')
  const [showObjectPanel, setShowObjectPanel] = useState(true)
  const [showPathStats, setShowPathStats] = useState(true)
  const [cameraMode, setCameraMode] = useState<'orbit' | 'fpv'>('orbit')
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null)

  // New window states
  const [showCameraView, setShowCameraView] = useState(false)
  const [showObjectPalette, setShowObjectPalette] = useState(false)
  const [showPathExecution, setShowPathExecution] = useState(false)
  const [showAlgorithmControls, setShowAlgorithmControls] = useState(false)

  // Robot position for FPV camera
  const [robotPosition, setRobotPosition] = useState<[number, number, number]>([0, 0.5, 0])
  const [robotRotation, setRobotRotation] = useState<[number, number, number]>([0, 0, 0])

  // Robot waypoint tracking for path execution
  const [currentWaypoint, setCurrentWaypoint] = useState(0)
  const [totalWaypoints, setTotalWaypoints] = useState(0)

  // Callback to update robot position from robot components
  const handleRobotPositionUpdate = (
    position: [number, number, number],
    rotation: [number, number, number]
  ) => {
    setRobotPosition(position)
    setRobotRotation(rotation)
  }

  // Callback to update waypoint progress from robot components
  const handleWaypointUpdate = (current: number, total: number) => {
    setCurrentWaypoint(current)
    setTotalWaypoints(total)
  }

  const manager = getAlgorithmManager()

  // Determine if we have any content to render
  const hasContent = sceneConfig || editableObjects.length > 0

  // Initialize A* path planner
  useEffect(() => {
    const planner = new AStarPathPlanner({
      gridSize: 0.5,
      worldBounds: {
        min: new THREE.Vector3(-10, 0, -10),
        max: new THREE.Vector3(10, 0, 10)
      },
      obstacles: []
    })
    setPathPlanner(planner)
  }, [])

  // Extract origin, destination, and obstacles from editableObjects
  const extractPathPlanningElements = () => {
    let origin: THREE.Vector3 | null = null
    let destination: THREE.Vector3 | null = null
    const obstacles: Array<{ position: THREE.Vector3; radius: number }> = []

    editableObjects.forEach((obj) => {
      if (obj.type === 'origin') {
        origin = new THREE.Vector3(obj.position[0], obj.position[1], obj.position[2])
      } else if (obj.type === 'destination') {
        destination = new THREE.Vector3(obj.position[0], obj.position[1], obj.position[2])
      } else if (obj.type === 'box' || obj.type === 'cylinder') {
        const radius = obj.type === 'box'
          ? Math.max(obj.size?.[0] || 0.5, obj.size?.[2] || 0.5) / 2
          : obj.radius || 0.3
        obstacles.push({
          position: new THREE.Vector3(obj.position[0], obj.position[1], obj.position[2]),
          radius: radius + 0.2 // Safety margin
        })
      }
    })

    return { origin, destination, obstacles }
  }

  // Auto-compute path when objects change
  useEffect(() => {
    if (!pathPlanner) return

    const { origin, destination, obstacles } = extractPathPlanningElements()

    if (!origin || !destination) {
      setComputedPath(null)
      return
    }

    try {
      // Update path planner with new obstacles
      pathPlanner.updateObstacles(obstacles)

      let waypoints: THREE.Vector3[] = []
      let algorithmName = 'A*'

      // Check which algorithm to use
      if (activeAlgorithmId === 'builtin-astar') {
        waypoints = pathPlanner.findPath(origin, destination)
        algorithmName = 'A* (Built-in)'
      } else {
        // Use custom generated algorithm
        try {
          const algorithm = manager.getAlgorithm(activeAlgorithmId)
          if (algorithm) {
            algorithmName = algorithm.name
            waypoints = manager.executePathPlanning(
              activeAlgorithmId,
              origin,
              destination,
              obstacles
            )
            console.log(`✅ Custom algorithm "${algorithmName}" generated path`)
          }
        } catch (error: any) {
          console.error(`❌ Custom algorithm failed, falling back to A*:`, error.message)
          waypoints = pathPlanner.findPath(origin, destination)
          algorithmName = 'A* (Fallback)'
        }
      }

      if (waypoints.length > 0) {
        // Calculate path length
        let length = 0
        for (let i = 1; i < waypoints.length; i++) {
          length += waypoints[i].distanceTo(waypoints[i - 1])
        }

        const path: ComputedPath = {
          waypoints,
          segments: waypoints.slice(0, -1).map((p, i) => ({
            start: p,
            end: waypoints[i + 1]
          })),
          length,
          estimatedTime: length / 1.0, // Assuming 1 m/s speed
          isValid: true,
          algorithmUsed: algorithmName
        }
        setComputedPath(path)
      } else {
        setComputedPath({
          waypoints: [],
          segments: [],
          length: 0,
          estimatedTime: 0,
          isValid: false,
          algorithmUsed: algorithmName
        })
      }
    } catch (error) {
      console.error('Path planning error:', error)
      setComputedPath(null)
    }
  }, [editableObjects, activeAlgorithmId, pathPlanner, manager])

  const handleModeSwitch = (newMode: SimulationMode) => {
    setMode(newMode)
    if (newMode === 'isaac') {
      setShowIsaacPanel(true)
    }
  }

  const addObject = (type: string, position: Vector3) => {
    // Save current state for undo
    setUndoStack(prev => [...prev, [...editableObjects]])

    // Determine Y position based on object type
    const yPosition = type === 'origin' || type === 'destination' ? 0.5 : 1

    const newObject: any = {
      id: `${type}_${Date.now()}`,
      type,
      position: [position.x, yPosition, position.z]
    }

    if (type === 'origin') {
      newObject.color = '#00FF00'
      newObject.shape = 'sphere'
      newObject.size = [0.3, 0.3, 0.3]
    } else if (type === 'destination') {
      newObject.color = '#FF0000'
      newObject.shape = 'sphere'
      newObject.size = [0.3, 0.3, 0.3]
    } else if (type === 'box') {
      newObject.color = '#FFA500'
      newObject.shape = 'box'
      newObject.size = [0.5, 0.5, 0.5]
    } else if (type === 'cylinder') {
      newObject.color = '#808080'
      newObject.shape = 'cylinder'
      newObject.radius = 0.3
      newObject.height = 1.0
    }

    setEditableObjects(prev => [...prev, newObject])

    // Auto-switch back to select tool after adding
    setActiveTool('select')
  }

  const handleClearScene = () => {
    if (confirm('Are you sure you want to clear all objects?')) {
      setUndoStack(prev => [...prev, [...editableObjects]])
      setEditableObjects([])
    }
  }

  const handleUndo = () => {
    if (undoStack.length > 0) {
      const previousState = undoStack[undoStack.length - 1]
      setEditableObjects(previousState)
      setUndoStack(prev => prev.slice(0, -1))
    }
  }

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying)
  }

  const handleStop = () => {
    setIsPlaying(false)
    // Reset objects to initial positions
    // (This would require storing initial state)
  }

  const handleSpeedChange = (speed: number) => {
    setSimulationSpeed(speed)
  }

  return (
    <div className="simulator">
      {/* Scene Editor Toolbar */}
      <SceneEditor
        activeTool={activeTool}
        onToolChange={setActiveTool}
        onClearScene={handleClearScene}
        onUndo={handleUndo}
        objectCount={editableObjects.length + (sceneConfig?.objects?.length || 0)}
      />

      {/* Object Window Panel */}
      {hasContent && showObjectPanel && (
        <div className="object-panel">
          <div className="panel-header">
            <h3>Scene Objects</h3>
            <button
              className="panel-close"
              onClick={() => setShowObjectPanel(false)}
              title="Close panel"
            >
              ×
            </button>
          </div>
          <div className="object-list">
            {editableObjects.length === 0 ? (
              <p className="empty-message">No objects placed</p>
            ) : (
              editableObjects.map((obj) => (
                <div
                  key={obj.id}
                  className={`object-item ${selectedObjectId === obj.id ? 'selected' : ''}`}
                  onClick={() => setSelectedObjectId(obj.id)}
                >
                  <div className="object-icon" style={{ backgroundColor: obj.color || '#888' }}>
                    {obj.type === 'origin' && '🟢'}
                    {obj.type === 'destination' && '🔴'}
                    {obj.type === 'box' && '📦'}
                    {obj.type === 'cylinder' && '🛢️'}
                  </div>
                  <div className="object-info">
                    <div className="object-type">{obj.type}</div>
                    <div className="object-position">
                      ({obj.position[0].toFixed(1)}, {obj.position[1].toFixed(1)}, {obj.position[2].toFixed(1)})
                    </div>
                  </div>
                  <button
                    className="object-delete"
                    onClick={(e) => {
                      e.stopPropagation()
                      setEditableObjects(prev => prev.filter(o => o.id !== obj.id))
                      if (selectedObjectId === obj.id) setSelectedObjectId(null)
                    }}
                    title="Delete object"
                  >
                    🗑️
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Path Stats Panel */}
      {hasContent && showPathStats && computedPath && computedPath.isValid && (
        <div className="path-stats-panel">
          <div className="panel-header">
            <h3>Path Information</h3>
            <button
              className="panel-close"
              onClick={() => setShowPathStats(false)}
              title="Close panel"
            >
              ×
            </button>
          </div>
          <div className="stats-content">
            <div className="stat-item">
              <span className="stat-label">Algorithm:</span>
              <span className="stat-value">{computedPath.algorithmUsed}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Path Length:</span>
              <span className="stat-value">{computedPath.length.toFixed(2)} m</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Est. Time:</span>
              <span className="stat-value">{computedPath.estimatedTime.toFixed(2)} s</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Waypoints:</span>
              <span className="stat-value">{computedPath.waypoints.length}</span>
            </div>
          </div>

          {/* Algorithm Selector */}
          <div className="algorithm-selector">
            <label>Path Algorithm:</label>
            <select
              value={activeAlgorithmId}
              onChange={(e) => setActiveAlgorithmId(e.target.value)}
            >
              <option value="builtin-astar">Built-in A*</option>
              {manager.getAllAlgorithms()
                .filter(a => a.type === 'path_planning')
                .map(algo => (
                  <option key={algo.id} value={algo.id}>
                    {algo.name}
                  </option>
                ))}
            </select>
            <button
              className="generate-algorithm-btn"
              onClick={() => setShowAlgorithmControls(!showAlgorithmControls)}
              title="Generate or manage algorithms"
            >
              {showAlgorithmControls ? '✖ Close' : '⚡ Generate'}
            </button>
          </div>
        </div>
      )}

      {/* Camera Controls Panel */}
      {hasContent && (
        <div className="camera-controls-panel">
          <button
            className={`camera-mode-btn ${!showCameraView ? 'active' : ''}`}
            onClick={() => setShowCameraView(false)}
            title="Main orbital camera view"
          >
            🔄 Orbit
          </button>
          <button
            className={`camera-mode-btn ${showCameraView ? 'active' : ''}`}
            onClick={() => setShowCameraView(!showCameraView)}
            title="Toggle robot FPV camera window"
          >
            👁️ FPV
          </button>
          <button
            className={`camera-mode-btn ${showObjectPalette ? 'active' : ''}`}
            onClick={() => setShowObjectPalette(!showObjectPalette)}
            title="Toggle object palette"
          >
            ➕ Objects
          </button>
          <button
            className={`camera-mode-btn ${showPathExecution ? 'active' : ''}`}
            onClick={() => setShowPathExecution(!showPathExecution)}
            title="Toggle path execution panel"
          >
            🎯 Path
          </button>
        </div>
      )}

      {/* Camera View Window (FPV) */}
      <CameraViewWindow
        isVisible={showCameraView}
        onClose={() => setShowCameraView(false)}
        robotPosition={robotPosition}
        robotRotation={robotRotation}
        sceneConfig={sceneConfig}
        editableObjects={editableObjects}
      />

      {/* Object Palette Window */}
      <ObjectPalette
        isVisible={showObjectPalette}
        onClose={() => setShowObjectPalette(false)}
        activeTool={activeTool}
        onToolSelect={setActiveTool}
      />

      {/* Path Execution Panel */}
      <PathExecutionPanel
        isVisible={showPathExecution}
        onClose={() => setShowPathExecution(false)}
        path={computedPath}
        isPlaying={isPlaying}
        onPlay={handlePlayPause}
        onPause={handlePlayPause}
        onStop={handleStop}
        currentWaypoint={currentWaypoint}
        totalWaypoints={totalWaypoints}
        distanceRemaining={computedPath?.length ? (computedPath.length * (totalWaypoints - currentWaypoint) / totalWaypoints) : 0}
        status={currentWaypoint >= totalWaypoints - 1 && totalWaypoints > 0 ? 'completed' : isPlaying ? 'moving' : 'idle'}
      />

      {/* Algorithm Controls Panel */}
      {showAlgorithmControls && sceneConfig && (
        <div className="algorithm-controls-overlay">
          <div className="algorithm-controls-panel">
            <div className="panel-header">
              <h3>Algorithm Generator</h3>
              <button className="close-btn" onClick={() => setShowAlgorithmControls(false)}>
                ✖
              </button>
            </div>
            <AlgorithmControls
              robotId="robot-1"
              robotType={sceneConfig.robot?.type === 'mobile_robot' ? 'mobile' : sceneConfig.robot?.type === 'robotic_arm' ? 'arm' : 'drone'}
              onAlgorithmApplied={(algorithm) => {
                console.log('Algorithm applied:', algorithm)
                // Reload algorithms to update the dropdown
                setActiveAlgorithmId(algorithm.id)
              }}
            />
          </div>
        </div>
      )}

      {/* Playback Controls */}
      {hasContent && (
        <div className="playback-controls">
          <button
            className={`playback-button ${isPlaying ? 'playing' : ''}`}
            onClick={handlePlayPause}
            title={isPlaying ? 'Pause simulation' : 'Play simulation'}
          >
            {isPlaying ? '⏸️' : '▶️'}
          </button>
          <button
            className="playback-button"
            onClick={handleStop}
            title="Stop and reset simulation"
          >
            ⏹️
          </button>
          <div className="speed-control">
            <label>Speed:</label>
            <input
              type="range"
              min="0.5"
              max="2"
              step="0.1"
              value={simulationSpeed}
              onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
            />
            <span>{simulationSpeed.toFixed(1)}x</span>
          </div>
        </div>
      )}

      {/* Mode Switcher - Only show when content exists */}
      {hasContent && (
        <div className="mode-switcher">
          <button
            className={`mode-button ${mode === 'rapier' ? 'active' : ''}`}
            onClick={() => handleModeSwitch('rapier')}
            title="Browser-based physics preview (fast)"
          >
            <span className="mode-icon">⚡</span>
            Rapier Preview
          </button>
          <button
            className={`mode-button ${mode === 'isaac' ? 'active' : ''}`}
            onClick={() => handleModeSwitch('isaac')}
            title="GPU-accelerated Isaac Lab simulation (accurate)"
          >
            <span className="mode-icon">🎯</span>
            Isaac Lab
          </button>
        </div>
      )}

      {!hasContent ? (
        <div className="simulator-empty">
          <div className="empty-message">
            <h2>🎮 3D Simulator</h2>
            <p>Use the scene editor to add objects, or generate a robot simulation!</p>
          </div>
        </div>
      ) : mode === 'isaac' && showIsaacPanel ? (
        <IsaacSimulationPanel
          sceneConfig={sceneConfig}
          onClose={() => setShowIsaacPanel(false)}
        />
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

            {/* Physical ground plane for collision */}
            <RigidBody type="fixed" position={[0, -0.25, 0]} colliders="cuboid" friction={0.7}>
              <mesh>
                <boxGeometry args={[100, 0.5, 100]} />
                <meshStandardMaterial transparent opacity={0} />
              </mesh>
            </RigidBody>

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
                  <MobileRobot
                    onPositionUpdate={handleRobotPositionUpdate}
                    onWaypointUpdate={handleWaypointUpdate}
                    path={computedPath}
                    isPaused={!isPlaying}
                  />
                )}
                {sceneConfig.robot?.type === 'robotic_arm' && (
                  <RoboticArm />
                )}
                {sceneConfig.robot?.type === 'quadcopter' && (
                  <Drone />
                )}
                {sceneConfig.robot?.type === 'urdf_custom' && (
                  <URDFRobot sceneConfig={sceneConfig} />
                )}
              </>
            )}

            {/* Render user-added editable objects */}
            {editableObjects.map((obj) => (
              <SceneObject key={obj.id} object={obj} />
            ))}

            {/* Path Visualizer - shows computed path */}
            <PathVisualizer
              origin={null}  // Don't render origin marker (already rendered via SceneObject)
              destination={null}  // Don't render destination marker (already rendered via SceneObject)
              path={computedPath}
              obstacles={[]}  // Don't render obstacles (already rendered via SceneObject)
              showGrid={false}
            />
          </Physics>

          {/* Click handler for adding objects */}
          <ClickHandler activeTool={activeTool} onAddObject={addObject} />

          {/* Camera controls */}
          <OrbitControls makeDefault />

          {/* Environment lighting */}
          <Environment preset="city" />
        </Canvas>
      )}
    </div>
  )
}
