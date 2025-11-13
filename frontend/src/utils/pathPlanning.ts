/**
 * A* Path Planning Algorithm
 * Based on: "Robot obstacle avoidance optimization by A* and DWA fusion algorithm"
 * (PLOS One, April 2024)
 *
 * Implements grid-based A* pathfinding with Euclidean heuristic
 */

import * as THREE from 'three'

export interface GridCell {
  x: number
  z: number
  walkable: boolean
  g: number // Cost from start
  h: number // Heuristic to goal
  f: number // Total cost (g + h)
  parent: GridCell | null
}

export interface PathPlannerConfig {
  gridSize: number // Size of each grid cell
  worldBounds: { min: THREE.Vector3; max: THREE.Vector3 }
  obstacles: THREE.Vector3[] // Obstacle positions with radius
}

export class AStarPathPlanner {
  private grid: GridCell[][]
  private gridSize: number
  private worldBounds: { min: THREE.Vector3; max: THREE.Vector3 }
  private gridWidth: number
  private gridHeight: number

  constructor(config: PathPlannerConfig) {
    this.gridSize = config.gridSize
    this.worldBounds = config.worldBounds

    // Calculate grid dimensions
    const worldWidth = config.worldBounds.max.x - config.worldBounds.min.x
    const worldHeight = config.worldBounds.max.z - config.worldBounds.min.z

    this.gridWidth = Math.ceil(worldWidth / this.gridSize)
    this.gridHeight = Math.ceil(worldHeight / this.gridSize)

    // Initialize grid
    this.grid = this.createGrid()

    // Mark obstacle cells
    this.updateObstacles(config.obstacles)
  }

  private createGrid(): GridCell[][] {
    const grid: GridCell[][] = []

    for (let x = 0; x < this.gridWidth; x++) {
      grid[x] = []
      for (let z = 0; z < this.gridHeight; z++) {
        grid[x][z] = {
          x,
          z,
          walkable: true,
          g: Infinity,
          h: Infinity,
          f: Infinity,
          parent: null
        }
      }
    }

    return grid
  }

  /**
   * Update obstacle map based on world positions
   */
  public updateObstacles(obstacles: Array<{position: THREE.Vector3, radius: number}> | THREE.Vector3[]) {
    // Convert Vector3[] to proper format if needed
    const formattedObstacles = Array.isArray(obstacles) && obstacles.length > 0 && obstacles[0] instanceof THREE.Vector3
      ? obstacles.map(pos => ({ position: pos as THREE.Vector3, radius: 0.5 }))
      : obstacles as Array<{position: THREE.Vector3, radius: number}>
    // Reset all cells to walkable
    for (let x = 0; x < this.gridWidth; x++) {
      for (let z = 0; z < this.gridHeight; z++) {
        this.grid[x][z].walkable = true
      }
    }

    // Mark cells near obstacles as unwalkable
    formattedObstacles.forEach(obstacle => {
      const gridX = this.worldToGridX(obstacle.position.x)
      const gridZ = this.worldToGridZ(obstacle.position.z)
      const cellRadius = Math.ceil(obstacle.radius / this.gridSize) + 1 // Safety margin

      for (let dx = -cellRadius; dx <= cellRadius; dx++) {
        for (let dz = -cellRadius; dz <= cellRadius; dz++) {
          const x = gridX + dx
          const z = gridZ + dz

          if (x >= 0 && x < this.gridWidth && z >= 0 && z < this.gridHeight) {
            this.grid[x][z].walkable = false
          }
        }
      }
    })
  }

  /**
   * Find path from start to goal using A* algorithm
   */
  public findPath(start: THREE.Vector3, goal: THREE.Vector3): THREE.Vector3[] {
    const startCell = this.getCell(start)
    const goalCell = this.getCell(goal)

    if (!startCell || !goalCell) {
      console.warn('Start or goal outside grid bounds')
      return []
    }

    if (!goalCell.walkable) {
      console.warn('Goal is inside obstacle')
      return []
    }

    // Reset grid
    this.resetGrid()

    const openSet: GridCell[] = []
    const closedSet = new Set<GridCell>()

    startCell.g = 0
    startCell.h = this.heuristic(startCell, goalCell)
    startCell.f = startCell.h
    openSet.push(startCell)

    while (openSet.length > 0) {
      // Get cell with lowest f score
      openSet.sort((a, b) => a.f - b.f)
      const current = openSet.shift()!

      // Goal reached
      if (current.x === goalCell.x && current.z === goalCell.z) {
        return this.reconstructPath(current)
      }

      closedSet.add(current)

      // Check neighbors (4-connected grid)
      const neighbors = this.getNeighbors(current)

      for (const neighbor of neighbors) {
        if (!neighbor.walkable || closedSet.has(neighbor)) {
          continue
        }

        const tentativeG = current.g + this.distance(current, neighbor)

        if (tentativeG < neighbor.g) {
          neighbor.parent = current
          neighbor.g = tentativeG
          neighbor.h = this.heuristic(neighbor, goalCell)
          neighbor.f = neighbor.g + neighbor.h

          if (!openSet.includes(neighbor)) {
            openSet.push(neighbor)
          }
        }
      }
    }

    // No path found
    console.warn('No path found')
    return []
  }

  private getNeighbors(cell: GridCell): GridCell[] {
    const neighbors: GridCell[] = []
    const directions = [
      { dx: 0, dz: 1 },
      { dx: 1, dz: 0 },
      { dx: 0, dz: -1 },
      { dx: -1, dz: 0 },
      // Diagonal (8-connected)
      { dx: 1, dz: 1 },
      { dx: 1, dz: -1 },
      { dx: -1, dz: 1 },
      { dx: -1, dz: -1 }
    ]

    for (const dir of directions) {
      const x = cell.x + dir.dx
      const z = cell.z + dir.dz

      if (x >= 0 && x < this.gridWidth && z >= 0 && z < this.gridHeight) {
        neighbors.push(this.grid[x][z])
      }
    }

    return neighbors
  }

  private heuristic(a: GridCell, b: GridCell): number {
    // Euclidean distance (from paper)
    const dx = a.x - b.x
    const dz = a.z - b.z
    return Math.sqrt(dx * dx + dz * dz)
  }

  private distance(a: GridCell, b: GridCell): number {
    const dx = Math.abs(a.x - b.x)
    const dz = Math.abs(a.z - b.z)

    // Diagonal distance
    if (dx && dz) {
      return Math.SQRT2
    }
    return 1
  }

  private reconstructPath(goal: GridCell): THREE.Vector3[] {
    const path: THREE.Vector3[] = []
    let current: GridCell | null = goal

    while (current) {
      const worldPos = this.gridToWorld(current.x, current.z)
      path.unshift(worldPos)
      current = current.parent
    }

    return path
  }

  private resetGrid() {
    for (let x = 0; x < this.gridWidth; x++) {
      for (let z = 0; z < this.gridHeight; z++) {
        const cell = this.grid[x][z]
        cell.g = Infinity
        cell.h = Infinity
        cell.f = Infinity
        cell.parent = null
      }
    }
  }

  private getCell(worldPos: THREE.Vector3): GridCell | null {
    const x = this.worldToGridX(worldPos.x)
    const z = this.worldToGridZ(worldPos.z)

    if (x >= 0 && x < this.gridWidth && z >= 0 && z < this.gridHeight) {
      return this.grid[x][z]
    }

    return null
  }

  private worldToGridX(worldX: number): number {
    return Math.floor((worldX - this.worldBounds.min.x) / this.gridSize)
  }

  private worldToGridZ(worldZ: number): number {
    return Math.floor((worldZ - this.worldBounds.min.z) / this.gridSize)
  }

  private gridToWorld(gridX: number, gridZ: number): THREE.Vector3 {
    const worldX = this.worldBounds.min.x + (gridX + 0.5) * this.gridSize
    const worldZ = this.worldBounds.min.z + (gridZ + 0.5) * this.gridSize
    return new THREE.Vector3(worldX, 0.5, worldZ)
  }

  /**
   * Get grid for visualization
   */
  public getGrid(): GridCell[][] {
    return this.grid
  }
}
