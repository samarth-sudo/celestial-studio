import { useState, useEffect } from 'react'
import * as THREE from 'three'
import PathPlanningControls from './PathPlanningControls'
import PathPlanningScene from './PathPlanningScene'
import { AStarPathPlanner } from '../utils/pathPlanning'
import { SCENARIO_PRESETS, loadScenario } from '../utils/scenarioPresets'
import { getAlgorithmManager } from '../services/AlgorithmManager'
import type { InteractionMode, ObstacleType, Obstacle, ComputedPath } from '../types/PathPlanning'
import './PathTestSandbox.css'

interface PathTestSandboxProps {
  robotType?: 'mobile' | 'arm' | 'drone'
  selectedAlgorithmId?: string | null
  onSimulate?: (data: {
    obstacles: Obstacle[]
    path: ComputedPath
    origin: THREE.Vector3
    destination: THREE.Vector3
  }) => void
}

export default function PathTestSandbox({ robotType: _robotType, selectedAlgorithmId, onSimulate }: PathTestSandboxProps) {
  const [mode, setMode] = useState<InteractionMode>('none')
  const [origin, setOrigin] = useState<THREE.Vector3 | null>(null)
  const [destination, setDestination] = useState<THREE.Vector3 | null>(null)
  const [obstacles, setObstacles] = useState<Obstacle[]>([])
  const [computedPath, setComputedPath] = useState<ComputedPath | null>(null)
  const [currentObstacleType, setCurrentObstacleType] = useState<ObstacleType>('box')
  const [pathPlanner, setPathPlanner] = useState<AStarPathPlanner | null>(null)
  const [activeAlgorithmId, setActiveAlgorithmId] = useState<string>('builtin-astar')

  const manager = getAlgorithmManager()
  const pathPlanningAlgorithms = manager.getAllAlgorithms().filter(a => a.type === 'path_planning')

  // Auto-select algorithm when passed from App
  useEffect(() => {
    if (selectedAlgorithmId) {
      setActiveAlgorithmId(selectedAlgorithmId)
    }
  }, [selectedAlgorithmId])

  // Initialize path planner
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

  // Update path planner obstacles whenever obstacles change
  useEffect(() => {
    if (pathPlanner) {
      const obstacleData = obstacles.map(obs => ({
        position: obs.position,
        radius: obs.radius || Math.max(obs.size.x, obs.size.z) / 2
      }))
      pathPlanner.updateObstacles(obstacleData)
    }
  }, [obstacles, pathPlanner])

  // Recompute path when origin, destination, obstacles, or algorithm changes
  useEffect(() => {
    if (origin && destination) {
      try {
        let waypoints: THREE.Vector3[] = []
        let algorithmName = 'A*'

        // Check which algorithm to use
        if (activeAlgorithmId === 'builtin-astar') {
          // Use built-in A* pathfinding
          if (pathPlanner) {
            waypoints = pathPlanner.findPath(origin, destination)
          }
          algorithmName = 'A* (Built-in)'
        } else {
          // Use custom generated algorithm
          try {
            const algorithm = manager.getAlgorithm(activeAlgorithmId)
            if (algorithm) {
              algorithmName = algorithm.name

              // Prepare obstacles in the format expected by generated algorithms
              const obstacleData = obstacles.map(obs => ({
                position: obs.position,
                radius: obs.radius || Math.max(obs.size.x, obs.size.z) / 2
              }))

              // Execute custom path planning algorithm
              waypoints = manager.executePathPlanning(
                activeAlgorithmId,
                origin,
                destination,
                obstacleData
              )

              console.log(`✅ Custom algorithm "${algorithmName}" generated path with ${waypoints.length} waypoints`)
            } else {
              console.error(`Algorithm ${activeAlgorithmId} not found`)
              throw new Error('Algorithm not found')
            }
          } catch (error: any) {
            console.error(`❌ Custom algorithm failed, falling back to A*:`, error.message)
            // Fall back to built-in A* if custom algorithm fails
            if (pathPlanner) {
              waypoints = pathPlanner.findPath(origin, destination)
              algorithmName = 'A* (Fallback)'
            }
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
    } else {
      setComputedPath(null)
    }
  }, [origin, destination, obstacles, pathPlanner, activeAlgorithmId, manager])

  const handleOriginSet = (position: THREE.Vector3) => {
    setOrigin(position)
    setMode('none')
  }

  const handleDestinationSet = (position: THREE.Vector3) => {
    setDestination(position)
    setMode('none')
  }

  const handleObstaclePlaced = (obstacle: Omit<Obstacle, 'id'>) => {
    const newObstacle: Obstacle = {
      ...obstacle,
      id: `obstacle-${Date.now()}-${Math.random()}`
    }
    setObstacles(prev => [...prev, newObstacle])
    // Keep mode active for placing multiple obstacles
  }

  const handleClearAll = () => {
    setOrigin(null)
    setDestination(null)
    setObstacles([])
    setComputedPath(null)
    setMode('none')
  }

  const handleClearPath = () => {
    setComputedPath(null)
  }

  const handleLoadScenario = (scenarioId: string) => {
    const scenario = loadScenario(scenarioId)
    if (!scenario) {
      console.error(`Scenario ${scenarioId} not found`)
      return
    }

    console.log(`📦 Loading scenario: ${scenario.name}`)

    // Clear existing state
    setMode('none')
    setObstacles([])
    setComputedPath(null)

    // Load scenario obstacles
    const loadedObstacles: Obstacle[] = scenario.obstacles.map((obs, index) => ({
      ...obs,
      id: `scenario-obstacle-${index}`
    }))
    setObstacles(loadedObstacles)

    // Set origin and destination if provided (fix y-coordinate to 0 for 2D pathfinding)
    if (scenario.origin) {
      const fixedOrigin = new THREE.Vector3(scenario.origin.x, 0, scenario.origin.z)
      setOrigin(fixedOrigin)
      console.log(`✅ Origin set to (${fixedOrigin.x}, ${fixedOrigin.z})`)
    }
    if (scenario.destination) {
      const fixedDestination = new THREE.Vector3(scenario.destination.x, 0, scenario.destination.z)
      setDestination(fixedDestination)
      console.log(`✅ Destination set to (${fixedDestination.x}, ${fixedDestination.z})`)
    }

    console.log(`✅ Loaded ${loadedObstacles.length} obstacles`)
  }

  const handleSimulate = () => {
    if (!origin || !destination || !computedPath || !computedPath.isValid || !onSimulate) {
      return
    }

    onSimulate({
      obstacles,
      path: computedPath,
      origin,
      destination
    })

    console.log('🚀 Simulating path in main simulator...')
  }

  return (
    <div className="path-test-sandbox">
      <div className="sandbox-layout">
        {/* Left controls panel */}
        <div className="sandbox-controls">
          <div className="sandbox-header">
            <h3>🎯 Path Planning Sandbox</h3>
            <p>Test path planning algorithms with interactive obstacles</p>
          </div>

          <div className="algorithm-selector">
            <h4>🧠 Algorithm</h4>
            <select
              value={activeAlgorithmId}
              onChange={(e) => setActiveAlgorithmId(e.target.value)}
              className="algorithm-dropdown"
            >
              <option value="builtin-astar">Built-in A* (Recommended)</option>
              {pathPlanningAlgorithms.length > 0 && (
                <optgroup label="Generated Algorithms">
                  {pathPlanningAlgorithms.map(algo => (
                    <option key={algo.id} value={algo.id}>
                      {algo.name} - {algo.complexity}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
            {activeAlgorithmId !== 'builtin-astar' && (
              <p className="algorithm-note">
                ✨ Using generated algorithm. If you see "No path found", try the built-in A* instead.
              </p>
            )}
            {pathPlanningAlgorithms.length === 0 && (
              <p className="algorithm-note" style={{ color: '#888' }}>
                💡 Generate custom algorithms in the Algorithms tab
              </p>
            )}
          </div>

          <PathPlanningControls
            mode={mode}
            onModeChange={setMode}
            onClearAll={handleClearAll}
            onClearPath={handleClearPath}
            currentObstacleType={currentObstacleType}
            onObstacleTypeChange={setCurrentObstacleType}
            computedPath={computedPath}
            hasOrigin={origin !== null}
            hasDestination={destination !== null}
          />

          {computedPath && computedPath.isValid && (
            <div className="simulate-section">
              <button
                className="simulate-btn"
                onClick={handleSimulate}
                disabled={!onSimulate}
              >
                🚀 Simulate Path in 3D
              </button>
              <p className="simulate-hint">
                Transfer this scenario to the simulator and watch the robot navigate!
              </p>
            </div>
          )}

          <div className="scenarios">
            <h4>📦 Scenario Presets</h4>
            <div className="scenario-buttons">
              {SCENARIO_PRESETS.map(scenario => (
                <button
                  key={scenario.id}
                  className="scenario-btn"
                  onClick={() => handleLoadScenario(scenario.id)}
                  title={scenario.description}
                >
                  {scenario.name}
                </button>
              ))}
            </div>
          </div>

          <div className="instructions">
            <h4>📖 Instructions</h4>
            <ol>
              <li>Load a <strong>scenario preset</strong> or start from scratch</li>
              <li>Click <strong>Set Origin</strong> and click on 3D view (green marker)</li>
              <li>Click <strong>Set Destination</strong> and click to place end (red marker)</li>
              <li>Click <strong>Place Obstacle</strong>, choose type, click to add</li>
              <li>Path auto-calculates using A* algorithm in real-time</li>
            </ol>
          </div>

          <div className="stats">
            <h4>📊 Statistics</h4>
            <div className="stat-row">
              <span>Obstacles:</span>
              <span>{obstacles.length}</span>
            </div>
            <div className="stat-row">
              <span>Origin Set:</span>
              <span>{origin ? '✓' : '✗'}</span>
            </div>
            <div className="stat-row">
              <span>Destination Set:</span>
              <span>{destination ? '✓' : '✗'}</span>
            </div>
          </div>
        </div>

        {/* Right 3D viewer panel */}
        <div className="sandbox-viewer">
          <PathPlanningScene
            origin={origin}
            destination={destination}
            obstacles={obstacles}
            computedPath={computedPath}
            mode={mode}
            currentObstacleType={currentObstacleType}
            onOriginSet={handleOriginSet}
            onDestinationSet={handleDestinationSet}
            onObstaclePlaced={handleObstaclePlaced}
          />
        </div>
      </div>
    </div>
  )
}
