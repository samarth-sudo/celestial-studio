/**
 * TOON Format Service for Celestial Studio
 *
 * Provides token-efficient serialization for LLM interactions
 * Reduces token usage by ~40% compared to JSON
 */

import { serialize, deserialize } from '@toon-format/toon'

export interface BenchmarkData {
  algorithm_id: string
  algorithm_name: string
  execution_time_ms: number
  success_rate: number
  path_length?: number
  path_smoothness?: number
  collision_count: number
  optimality_score?: number
}

export interface SceneConfig {
  environment: string
  robot_type: string
  task: string
  objects: string[]
  bounds: {
    minX: number
    maxX: number
    minZ: number
    maxZ: number
  }
}

export class ToonService {
  /**
   * Serialize benchmark results to TOON format
   * Optimized for uniform arrays of benchmark data
   */
  serializeBenchmarkResults(results: BenchmarkData[]): string {
    try {
      // TOON is highly efficient for uniform tabular data
      const toonString = serialize(results, {
        format: 'toon',
        indent: 2
      })

      console.log('📊 TOON Serialization:')
      console.log(`   Original JSON: ${JSON.stringify(results).length} chars`)
      console.log(`   TOON format: ${toonString.length} chars`)
      console.log(`   Savings: ${((1 - toonString.length / JSON.stringify(results).length) * 100).toFixed(1)}%`)

      return toonString
    } catch (error) {
      console.warn('TOON serialization failed, falling back to JSON:', error)
      return JSON.stringify(results, null, 2)
    }
  }

  /**
   * Serialize scene configuration to TOON format
   */
  serializeSceneConfig(config: SceneConfig): string {
    try {
      return serialize(config, {
        format: 'toon',
        indent: 2
      })
    } catch (error) {
      console.warn('TOON serialization failed for scene config:', error)
      return JSON.stringify(config, null, 2)
    }
  }

  /**
   * Serialize algorithm comparison data for LLM analysis
   */
  serializeComparisonData(comparison: any): string {
    try {
      // Extract the rankings array which is uniform and benefits most from TOON
      const rankings = comparison.rankings || []

      if (rankings.length > 0) {
        const toonRankings = serialize(rankings, { format: 'toon', indent: 2 })

        // Combine TOON rankings with JSON metadata
        return `# Algorithm Comparison Data\n\n## Rankings (TOON Format)\n${toonRankings}\n\n## Recommendation\n${JSON.stringify(comparison.recommendation, null, 2)}`
      }

      return JSON.stringify(comparison, null, 2)
    } catch (error) {
      console.warn('TOON serialization failed for comparison:', error)
      return JSON.stringify(comparison, null, 2)
    }
  }

  /**
   * Deserialize TOON format back to objects
   */
  deserialize<T>(toonString: string): T {
    try {
      return deserialize(toonString) as T
    } catch (error) {
      console.warn('TOON deserialization failed, trying JSON:', error)
      return JSON.parse(toonString) as T
    }
  }

  /**
   * Calculate token savings estimate
   * Uses rough approximation of 1 token ≈ 4 characters
   */
  estimateTokenSavings(originalJson: string, toonString: string): {
    jsonTokens: number
    toonTokens: number
    savings: number
    savingsPercent: number
  } {
    const jsonTokens = Math.ceil(originalJson.length / 4)
    const toonTokens = Math.ceil(toonString.length / 4)
    const savings = jsonTokens - toonTokens
    const savingsPercent = (savings / jsonTokens) * 100

    return {
      jsonTokens,
      toonTokens,
      savings,
      savingsPercent
    }
  }

  /**
   * Prepare data for LLM prompt with TOON format
   * Returns formatted string ready to include in LLM context
   */
  prepareForLLM(data: any, dataType: 'benchmark' | 'scene' | 'comparison'): string {
    let toonData: string

    switch (dataType) {
      case 'benchmark':
        toonData = this.serializeBenchmarkResults(data)
        break
      case 'scene':
        toonData = this.serializeSceneConfig(data)
        break
      case 'comparison':
        toonData = this.serializeComparisonData(data)
        break
      default:
        toonData = JSON.stringify(data, null, 2)
    }

    return `
Data Format: TOON (Token-Oriented Object Notation)
Note: This is a compact format optimized for LLM parsing.
Each row represents one record with consistent field structure.

${toonData}
`
  }
}

// Singleton instance
let toonServiceInstance: ToonService | null = null

/**
 * Get or create ToonService singleton
 */
export function getToonService(): ToonService {
  if (!toonServiceInstance) {
    toonServiceInstance = new ToonService()
  }
  return toonServiceInstance
}
