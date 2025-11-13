import * as THREE from 'three'
import type { ScenarioPreset } from '../types/PathPlanning'

export const SCENARIO_PRESETS: ScenarioPreset[] = [
  {
    id: 'empty',
    name: 'Empty Space',
    description: 'Clean slate for testing',
    obstacles: [],
    origin: new THREE.Vector3(-8, 0.5, -8),
    destination: new THREE.Vector3(8, 0.5, 8)
  },
  {
    id: 'warehouse',
    name: 'Warehouse',
    description: 'Shelving units and storage areas',
    obstacles: [
      // Left shelving units
      { type: 'box', position: new THREE.Vector3(-6, 1, -4), size: new THREE.Vector3(3, 2, 1), color: '#8B4513' },
      { type: 'box', position: new THREE.Vector3(-6, 1, 0), size: new THREE.Vector3(3, 2, 1), color: '#8B4513' },
      { type: 'box', position: new THREE.Vector3(-6, 1, 4), size: new THREE.Vector3(3, 2, 1), color: '#8B4513' },

      // Right shelving units
      { type: 'box', position: new THREE.Vector3(6, 1, -4), size: new THREE.Vector3(3, 2, 1), color: '#8B4513' },
      { type: 'box', position: new THREE.Vector3(6, 1, 0), size: new THREE.Vector3(3, 2, 1), color: '#8B4513' },
      { type: 'box', position: new THREE.Vector3(6, 1, 4), size: new THREE.Vector3(3, 2, 1), color: '#8B4513' },

      // Center obstacle
      { type: 'box', position: new THREE.Vector3(0, 0.5, 0), size: new THREE.Vector3(2, 1, 2), color: '#CD853F' }
    ],
    origin: new THREE.Vector3(-8, 0.5, -6),
    destination: new THREE.Vector3(8, 0.5, 6)
  },
  {
    id: 'maze',
    name: 'Maze Challenge',
    description: 'Complex path with walls',
    obstacles: [
      // Top horizontal walls
      { type: 'wall', position: new THREE.Vector3(-4, 1, 6), size: new THREE.Vector3(4, 2, 0.2), rotation: new THREE.Euler(0, 0, 0) },
      { type: 'wall', position: new THREE.Vector3(4, 1, 6), size: new THREE.Vector3(4, 2, 0.2), rotation: new THREE.Euler(0, 0, 0) },

      // Vertical walls
      { type: 'wall', position: new THREE.Vector3(-6, 1, 0), size: new THREE.Vector3(0.2, 2, 8), rotation: new THREE.Euler(0, 0, 0) },
      { type: 'wall', position: new THREE.Vector3(6, 1, 0), size: new THREE.Vector3(0.2, 2, 8), rotation: new THREE.Euler(0, 0, 0) },

      // Middle obstacles
      { type: 'wall', position: new THREE.Vector3(-2, 1, 2), size: new THREE.Vector3(0.2, 2, 6), rotation: new THREE.Euler(0, 0, 0) },
      { type: 'wall', position: new THREE.Vector3(2, 1, -2), size: new THREE.Vector3(0.2, 2, 6), rotation: new THREE.Euler(0, 0, 0) },

      // Bottom horizontal wall
      { type: 'wall', position: new THREE.Vector3(0, 1, -6), size: new THREE.Vector3(6, 2, 0.2), rotation: new THREE.Euler(0, 0, 0) }
    ],
    origin: new THREE.Vector3(-8, 0.5, 8),
    destination: new THREE.Vector3(8, 0.5, -8)
  },
  {
    id: 'office',
    name: 'Office Space',
    description: 'Desks, plants, and furniture',
    obstacles: [
      // Desks
      { type: 'box', position: new THREE.Vector3(-4, 0.4, -4), size: new THREE.Vector3(2, 0.8, 1), color: '#654321' },
      { type: 'box', position: new THREE.Vector3(-4, 0.4, 2), size: new THREE.Vector3(2, 0.8, 1), color: '#654321' },
      { type: 'box', position: new THREE.Vector3(4, 0.4, -4), size: new THREE.Vector3(2, 0.8, 1), color: '#654321' },
      { type: 'box', position: new THREE.Vector3(4, 0.4, 2), size: new THREE.Vector3(2, 0.8, 1), color: '#654321' },

      // Plants
      { type: 'plant', position: new THREE.Vector3(-6, 0, -6), size: new THREE.Vector3(0.5, 1.2, 0.5) },
      { type: 'plant', position: new THREE.Vector3(6, 0, -6), size: new THREE.Vector3(0.5, 1.2, 0.5) },
      { type: 'plant', position: new THREE.Vector3(-6, 0, 6), size: new THREE.Vector3(0.5, 1.2, 0.5) },
      { type: 'plant', position: new THREE.Vector3(6, 0, 6), size: new THREE.Vector3(0.5, 1.2, 0.5) },

      // Meeting table (center)
      { type: 'cylinder', position: new THREE.Vector3(0, 0.4, 0), size: new THREE.Vector3(0, 0.8, 0), radius: 1.5, color: '#8B4513' }
    ],
    origin: new THREE.Vector3(-7, 0.5, -7),
    destination: new THREE.Vector3(7, 0.5, 7)
  },
  {
    id: 'cluttered',
    name: 'Cluttered Room',
    description: 'Random obstacles everywhere',
    obstacles: [
      { type: 'box', position: new THREE.Vector3(-5, 0.5, -5), size: new THREE.Vector3(1, 1, 1), color: '#ff6b6b' },
      { type: 'cylinder', position: new THREE.Vector3(-2, 0.5, -6), size: new THREE.Vector3(0, 1, 0), radius: 0.6, color: '#4ecdc4' },
      { type: 'sphere', position: new THREE.Vector3(3, 0.5, -4), size: new THREE.Vector3(0, 0, 0), radius: 0.8, color: '#ffe66d' },
      { type: 'box', position: new THREE.Vector3(6, 0.6, -2), size: new THREE.Vector3(1.2, 1.2, 1.2), color: '#95e1d3' },
      { type: 'cylinder', position: new THREE.Vector3(-6, 0.5, 2), size: new THREE.Vector3(0, 1, 0), radius: 0.5, color: '#f38181' },
      { type: 'box', position: new THREE.Vector3(-3, 0.4, 4), size: new THREE.Vector3(0.8, 0.8, 0.8), color: '#aa96da' },
      { type: 'sphere', position: new THREE.Vector3(1, 0.5, 5), size: new THREE.Vector3(0, 0, 0), radius: 0.7, color: '#fcbad3' },
      { type: 'cylinder', position: new THREE.Vector3(5, 0.5, 5), size: new THREE.Vector3(0, 1, 0), radius: 0.6, color: '#a8d8ea' },
      { type: 'box', position: new THREE.Vector3(-1, 0.5, -2), size: new THREE.Vector3(1, 1, 1), color: '#ffaaa5' },
      { type: 'sphere', position: new THREE.Vector3(2, 0.5, 2), size: new THREE.Vector3(0, 0, 0), radius: 0.5, color: '#ff8b94' }
    ],
    origin: new THREE.Vector3(-8, 0.5, -8),
    destination: new THREE.Vector3(8, 0.5, 8)
  },
  {
    id: 'corridor',
    name: 'Narrow Corridor',
    description: 'Test tight spaces navigation',
    obstacles: [
      // Left wall
      { type: 'wall', position: new THREE.Vector3(-3, 1, 0), size: new THREE.Vector3(0.2, 2, 16), rotation: new THREE.Euler(0, 0, 0) },
      // Right wall
      { type: 'wall', position: new THREE.Vector3(3, 1, 0), size: new THREE.Vector3(0.2, 2, 16), rotation: new THREE.Euler(0, 0, 0) },

      // Obstacles in corridor
      { type: 'box', position: new THREE.Vector3(-1, 0.5, -4), size: new THREE.Vector3(1, 1, 1), color: '#ff6b6b' },
      { type: 'cylinder', position: new THREE.Vector3(1, 0.5, 0), size: new THREE.Vector3(0, 1, 0), radius: 0.5, color: '#4ecdc4' },
      { type: 'box', position: new THREE.Vector3(-1, 0.5, 4), size: new THREE.Vector3(1, 1, 1), color: '#ffe66d' }
    ],
    origin: new THREE.Vector3(0, 0.5, -7),
    destination: new THREE.Vector3(0, 0.5, 7)
  },
  {
    id: 'garden',
    name: 'Garden Path',
    description: 'Navigate through plants and trees',
    obstacles: [
      // Plants scattered around
      { type: 'plant', position: new THREE.Vector3(-5, 0, -5), size: new THREE.Vector3(0.6, 1.4, 0.6) },
      { type: 'plant', position: new THREE.Vector3(-3, 0, -6), size: new THREE.Vector3(0.5, 1.2, 0.5) },
      { type: 'plant', position: new THREE.Vector3(-6, 0, -2), size: new THREE.Vector3(0.7, 1.5, 0.7) },
      { type: 'plant', position: new THREE.Vector3(-4, 0, 1), size: new THREE.Vector3(0.5, 1.3, 0.5) },
      { type: 'plant', position: new THREE.Vector3(-6, 0, 4), size: new THREE.Vector3(0.6, 1.4, 0.6) },
      { type: 'plant', position: new THREE.Vector3(-2, 0, 5), size: new THREE.Vector3(0.5, 1.2, 0.5) },

      { type: 'plant', position: new THREE.Vector3(5, 0, -5), size: new THREE.Vector3(0.6, 1.4, 0.6) },
      { type: 'plant', position: new THREE.Vector3(3, 0, -6), size: new THREE.Vector3(0.5, 1.2, 0.5) },
      { type: 'plant', position: new THREE.Vector3(6, 0, -2), size: new THREE.Vector3(0.7, 1.5, 0.7) },
      { type: 'plant', position: new THREE.Vector3(4, 0, 1), size: new THREE.Vector3(0.5, 1.3, 0.5) },
      { type: 'plant', position: new THREE.Vector3(6, 0, 4), size: new THREE.Vector3(0.6, 1.4, 0.6) },
      { type: 'plant', position: new THREE.Vector3(2, 0, 5), size: new THREE.Vector3(0.5, 1.2, 0.5) },

      // Some rocks (spheres)
      { type: 'sphere', position: new THREE.Vector3(-2, 0.3, -3), size: new THREE.Vector3(0, 0, 0), radius: 0.3, color: '#808080' },
      { type: 'sphere', position: new THREE.Vector3(2, 0.3, -3), size: new THREE.Vector3(0, 0, 0), radius: 0.3, color: '#808080' },
      { type: 'sphere', position: new THREE.Vector3(-2, 0.3, 3), size: new THREE.Vector3(0, 0, 0), radius: 0.3, color: '#808080' },
      { type: 'sphere', position: new THREE.Vector3(2, 0.3, 3), size: new THREE.Vector3(0, 0, 0), radius: 0.3, color: '#808080' }
    ],
    origin: new THREE.Vector3(-7, 0.5, -7),
    destination: new THREE.Vector3(7, 0.5, 7)
  }
]

export function loadScenario(scenarioId: string): ScenarioPreset | null {
  return SCENARIO_PRESETS.find(s => s.id === scenarioId) || null
}

export function getScenarioList() {
  return SCENARIO_PRESETS.map(s => ({
    id: s.id,
    name: s.name,
    description: s.description
  }))
}
