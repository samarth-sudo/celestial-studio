import { useState } from 'react'
import { getExportService, ExportFormat, ExportFormatInfo } from '../services/ExportService'
import type { Algorithm } from '../services/AlgorithmManager'
import type { SceneConfig, RobotConfig } from '../types'
import './DownloadPanel.css'

interface DownloadPanelProps {
  algorithms: Algorithm[]
  sceneConfig: SceneConfig | null
  robots: RobotConfig[]
  projectName?: string
}

export default function DownloadPanel({ algorithms, sceneConfig, robots, projectName }: DownloadPanelProps) {
  const [selectedFormat, setSelectedFormat] = useState<ExportFormat>('react')
  const [isExporting, setIsExporting] = useState(false)
  const [exportStatus, setExportStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [statusMessage, setStatusMessage] = useState('')

  const exportService = getExportService()
  const exportFormats = exportService.getExportFormats()

  const handleExport = async () => {
    if (isExporting) return

    // Validate that we have content to export
    if (!algorithms || algorithms.length === 0) {
      setExportStatus('error')
      setStatusMessage('No algorithms to export. Generate some algorithms first!')
      setTimeout(() => setExportStatus('idle'), 3000)
      return
    }

    setIsExporting(true)
    setExportStatus('idle')
    setStatusMessage('')

    try {
      const result = await exportService.exportPackage({
        export_format: selectedFormat,
        algorithms: algorithms,
        scene_config: sceneConfig || {},
        robots: robots || [],
        project_name: projectName || 'celestial_simulation'
      })

      if (result.status === 'success') {
        setExportStatus('success')
        setStatusMessage(`Package exported successfully! Download started.`)
        setTimeout(() => setExportStatus('idle'), 5000)
      } else {
        setExportStatus('error')
        setStatusMessage(result.error || 'Export failed. Please try again.')
        setTimeout(() => setExportStatus('idle'), 5000)
      }
    } catch (error: unknown) {
      console.error('Export error:', error)
      setExportStatus('error')
      setStatusMessage(error instanceof Error ? error.message : 'Export failed. Please try again.')
      setTimeout(() => setExportStatus('idle'), 5000)
    } finally {
      setIsExporting(false)
    }
  }

  const getSelectedFormatInfo = (): ExportFormatInfo | undefined => {
    return exportFormats.find(f => f.id === selectedFormat)
  }

  return (
    <div className="download-panel">
      <div className="download-header">
        <h2>📦 Export Simulation Package</h2>
        <p>Download your simulation in various formats for different use cases</p>
      </div>

      <div className="format-selector">
        <h3>Select Export Format</h3>
        <div className="format-grid">
          {exportFormats.map((format) => (
            <button
              key={format.id}
              className={`format-card ${selectedFormat === format.id ? 'selected' : ''}`}
              onClick={() => setSelectedFormat(format.id)}
              disabled={isExporting}
            >
              <div className="format-icon">{format.icon}</div>
              <div className="format-name">{format.name}</div>
              <div className="format-description">{format.description}</div>
            </button>
          ))}
        </div>
      </div>

      {getSelectedFormatInfo() && (
        <div className="format-details">
          <h3>What's Included</h3>
          <ul className="includes-list">
            {getSelectedFormatInfo()!.includes.map((item, index) => (
              <li key={index}>
                <span className="check-icon">✓</span>
                {item}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="export-actions">
        <button
          className="export-btn"
          onClick={handleExport}
          disabled={isExporting || !algorithms || algorithms.length === 0}
        >
          {isExporting ? (
            <>
              <span className="spinner"></span>
              Generating Package...
            </>
          ) : (
            <>
              <span className="download-icon">⬇️</span>
              Export as {getSelectedFormatInfo()?.name}
            </>
          )}
        </button>

        {exportStatus !== 'idle' && (
          <div className={`status-message ${exportStatus}`}>
            {exportStatus === 'success' && <span className="status-icon">✅</span>}
            {exportStatus === 'error' && <span className="status-icon">❌</span>}
            <span>{statusMessage}</span>
          </div>
        )}
      </div>

      {algorithms && algorithms.length === 0 && (
        <div className="empty-state">
          <div className="empty-icon">📝</div>
          <p>No algorithms generated yet</p>
          <p className="empty-hint">Generate some algorithms first, then export your simulation</p>
        </div>
      )}

      <div className="export-info">
        <h3>Export Information</h3>
        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">Algorithms:</span>
            <span className="info-value">{algorithms?.length || 0}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Robots:</span>
            <span className="info-value">{robots?.length || 0}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Project Name:</span>
            <span className="info-value">{projectName || 'celestial_simulation'}</span>
          </div>
        </div>
      </div>
    </div>
  )
}
