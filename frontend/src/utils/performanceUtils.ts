/**
 * Performance Optimization Utilities
 *
 * Provides throttling, debouncing, and object pooling for Three.js
 * to reduce garbage collection and improve frame rates.
 */

import * as THREE from 'three'

/**
 * Throttle a function to execute at most once per interval
 *
 * @param func Function to throttle
 * @param interval Minimum time between executions (ms)
 * @returns Throttled function
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  interval: number
): (...args: Parameters<T>) => void {
  let lastCall = 0
  let timeoutId: NodeJS.Timeout | null = null

  return function throttled(...args: Parameters<T>) {
    const now = Date.now()
    const timeSinceLastCall = now - lastCall

    if (timeSinceLastCall >= interval) {
      lastCall = now
      func(...args)
    } else if (!timeoutId) {
      // Schedule the next call
      timeoutId = setTimeout(() => {
        lastCall = Date.now()
        timeoutId = null
        func(...args)
      }, interval - timeSinceLastCall)
    }
  }
}

/**
 * Debounce a function to execute only after it stops being called
 *
 * @param func Function to debounce
 * @param delay Delay in milliseconds
 * @returns Debounced function
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: NodeJS.Timeout | null = null

  return function debounced(...args: Parameters<T>) {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }

    timeoutId = setTimeout(() => {
      func(...args)
      timeoutId = null
    }, delay)
  }
}

/**
 * Object Pool for Three.js Vector3 objects
 * Reduces garbage collection by reusing objects
 */
export class Vector3Pool {
  private pool: THREE.Vector3[] = []
  private readonly maxSize: number

  constructor(initialSize: number = 10, maxSize: number = 100) {
    this.maxSize = maxSize
    // Pre-allocate initial pool
    for (let i = 0; i < initialSize; i++) {
      this.pool.push(new THREE.Vector3())
    }
  }

  /**
   * Get a Vector3 from the pool
   */
  get(x: number = 0, y: number = 0, z: number = 0): THREE.Vector3 {
    const vector = this.pool.pop() || new THREE.Vector3()
    return vector.set(x, y, z)
  }

  /**
   * Return a Vector3 to the pool for reuse
   */
  release(vector: THREE.Vector3): void {
    if (this.pool.length < this.maxSize) {
      vector.set(0, 0, 0) // Reset to origin
      this.pool.push(vector)
    }
  }

  /**
   * Get current pool size
   */
  size(): number {
    return this.pool.length
  }

  /**
   * Clear the entire pool
   */
  clear(): void {
    this.pool = []
  }
}

/**
 * Object Pool for Three.js Euler objects
 */
export class EulerPool {
  private pool: THREE.Euler[] = []
  private readonly maxSize: number

  constructor(initialSize: number = 10, maxSize: number = 100) {
    this.maxSize = maxSize
    for (let i = 0; i < initialSize; i++) {
      this.pool.push(new THREE.Euler())
    }
  }

  get(x: number = 0, y: number = 0, z: number = 0, order: string = 'XYZ'): THREE.Euler {
    const euler = this.pool.pop() || new THREE.Euler()
    return euler.set(x, y, z, order as THREE.EulerOrder)
  }

  release(euler: THREE.Euler): void {
    if (this.pool.length < this.maxSize) {
      euler.set(0, 0, 0)
      this.pool.push(euler)
    }
  }

  size(): number {
    return this.pool.length
  }

  clear(): void {
    this.pool = []
  }
}

/**
 * Frame rate limiter for algorithm execution
 * Ensures algorithms don't run more than specified Hz
 */
export class FrameRateLimiter {
  private lastExecutionTime: number = 0
  private readonly intervalMs: number

  /**
   * @param hz Target execution frequency (Hz)
   */
  constructor(hz: number) {
    this.intervalMs = 1000 / hz
  }

  /**
   * Check if enough time has passed to execute again
   */
  shouldExecute(): boolean {
    const now = Date.now()
    const elapsed = now - this.lastExecutionTime

    if (elapsed >= this.intervalMs) {
      this.lastExecutionTime = now
      return true
    }

    return false
  }

  /**
   * Reset the limiter
   */
  reset(): void {
    this.lastExecutionTime = 0
  }

  /**
   * Get time until next execution (ms)
   */
  timeUntilNext(): number {
    const elapsed = Date.now() - this.lastExecutionTime
    return Math.max(0, this.intervalMs - elapsed)
  }
}

/**
 * Clamp a value between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

/**
 * Validate and clamp a velocity vector
 *
 * @param velocity Velocity vector to validate
 * @param maxMagnitude Maximum allowed magnitude
 * @returns Clamped velocity vector
 */
export function clampVelocity(
  velocity: THREE.Vector3,
  maxMagnitude: number
): THREE.Vector3 {
  const magnitude = velocity.length()

  if (magnitude > maxMagnitude) {
    // Scale down to max magnitude
    velocity.normalize().multiplyScalar(maxMagnitude)
  }

  return velocity
}

/**
 * Check if a value is a valid number
 */
export function isValidNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value) && isFinite(value)
}

/**
 * Validate algorithm result is a safe velocity object
 */
export function isValidVelocity(result: unknown): result is { x: number; z: number } {
  if (!result || typeof result !== 'object') return false

  const vel = result as any
  return (
    'x' in vel &&
    'z' in vel &&
    isValidNumber(vel.x) &&
    isValidNumber(vel.z)
  )
}

/**
 * Performance metrics tracker
 */
export class PerformanceMetrics {
  private frameTimes: number[] = []
  private readonly maxSamples: number = 60

  recordFrame(deltaTime: number): void {
    this.frameTimes.push(deltaTime)
    if (this.frameTimes.length > this.maxSamples) {
      this.frameTimes.shift()
    }
  }

  getFPS(): number {
    if (this.frameTimes.length === 0) return 0
    const avgDelta = this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length
    return avgDelta > 0 ? 1000 / avgDelta : 0
  }

  getAvgFrameTime(): number {
    if (this.frameTimes.length === 0) return 0
    return this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length
  }

  reset(): void {
    this.frameTimes = []
  }
}
