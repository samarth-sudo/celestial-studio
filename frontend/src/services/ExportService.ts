/**
 * Export Service for Celestial Studio
 *
 * Handles package generation and download for simulations.
 * Supports multiple export formats: React, ROS, Python, and Algorithms-only.
 */

import axios from 'axios'

export type ExportFormat = 'react' | 'ros' | 'python' | 'algorithms'

export interface ExportRequest {
  export_format: ExportFormat
  algorithms: any[]
  scene_config: any
  robots: any[]
  project_name?: string
}

export interface ExportResponse {
  status: 'success' | 'error'
  filename?: string
  export_format?: string
  project_name?: string
  download_url?: string
  error?: string
}

export interface ExportFormatInfo {
  id: ExportFormat
  name: string
  description: string
  icon: string
  includes: string[]
}

/**
 * ExportService - Package generation and download
 */
export class ExportService {
  private readonly API_URL = 'http://localhost:8000'
  private currentExport: ExportResponse | null = null

  /**
   * Get available export formats with descriptions
   */
  getExportFormats(): ExportFormatInfo[] {
    return [
      {
        id: 'react',
        name: 'React/TypeScript Project',
        description: 'Complete standalone web app with Three.js + Rapier physics',
        icon: '⚛️',
        includes: [
          'Full React project setup',
          'Algorithm implementations',
          '3D scene with physics',
          'Robot components',
          'package.json & build config'
        ]
      },
      {
        id: 'ros',
        name: 'ROS Package',
        description: 'Robot Operating System workspace for real robot deployment',
        icon: '🤖',
        includes: [
          'ROS workspace structure',
          'Python ROS nodes',
          'Launch files',
          'URDF robot descriptions',
          'Configuration files'
        ]
      },
      {
        id: 'python',
        name: 'Python Scripts',
        description: 'Standalone Python implementations with examples',
        icon: '🐍',
        includes: [
          'Algorithm classes',
          'Configuration files',
          'Example scripts',
          'Utility functions',
          'requirements.txt'
        ]
      },
      {
        id: 'algorithms',
        name: 'Algorithm Files Only',
        description: 'Just the algorithm code without project scaffolding',
        icon: '📝',
        includes: [
          'TypeScript algorithm files',
          'Algorithm metadata',
          'README documentation'
        ]
      }
    ]
  }

  /**
   * Generate and download package
   */
  async exportPackage(request: ExportRequest): Promise<ExportResponse> {
    try {
      console.log(`📦 Generating ${request.export_format} package...`)

      // Generate package
      const response = await axios.post<ExportResponse>(
        `${this.API_URL}/api/export/package`,
        request,
        {
          timeout: 60000, // 60 second timeout for package generation
          headers: {
            'Content-Type': 'application/json'
          }
        }
      )

      if (response.data.status === 'success' && response.data.filename) {
        this.currentExport = response.data
        console.log(`✅ Package generated: ${response.data.filename}`)

        // Automatically trigger download
        await this.downloadPackage(response.data.filename)

        return response.data
      } else {
        throw new Error('Package generation failed')
      }
    } catch (error: any) {
      console.error('❌ Export error:', error)

      const errorMessage = error.response?.data?.detail || error.message || 'Unknown error'

      return {
        status: 'error',
        error: errorMessage
      }
    }
  }

  /**
   * Download generated package file
   */
  async downloadPackage(filename: string): Promise<void> {
    try {
      console.log(`⬇️ Downloading package: ${filename}`)

      // Create download URL
      const downloadUrl = `${this.API_URL}/api/export/download/${filename}`

      // Fetch the file as a blob
      const response = await axios.get(downloadUrl, {
        responseType: 'blob',
        timeout: 30000
      })

      // Create blob URL and trigger download
      const blob = new Blob([response.data], { type: 'application/zip' })
      const url = window.URL.createObjectURL(blob)

      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()

      // Cleanup
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)

      console.log(`✅ Download complete: ${filename}`)
    } catch (error: any) {
      console.error('❌ Download error:', error)
      throw new Error(error.response?.data?.detail || 'Failed to download package')
    }
  }

  /**
   * Get the last export response
   */
  getLastExport(): ExportResponse | null {
    return this.currentExport
  }

  /**
   * Clear export history
   */
  clearExport(): void {
    this.currentExport = null
  }
}

// Singleton instance
let exportServiceInstance: ExportService | null = null

/**
 * Get or create ExportService singleton
 */
export function getExportService(): ExportService {
  if (!exportServiceInstance) {
    exportServiceInstance = new ExportService()
  }
  return exportServiceInstance
}
