import { useState } from 'react'
import axios from 'axios'
import './URDFUploader.css'

interface URDFUploaderProps {
  onRobotLoaded: (sceneConfig: any) => void
}

export default function URDFUploader({ onRobotLoaded }: URDFUploaderProps) {
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [robotInfo, setRobotInfo] = useState<any | null>(null)

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.name.endsWith('.urdf')) {
      setError('Please upload a .urdf file')
      return
    }

    setUploading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('http://localhost:8000/api/urdf/parse', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      console.log('✅ URDF parsed successfully:', response.data)

      setRobotInfo(response.data.urdf_data)
      onRobotLoaded(response.data.scene_config)

    } catch (error: any) {
      console.error('❌ URDF upload failed:', error)
      setError(error.response?.data?.detail || 'Failed to parse URDF file')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="urdf-uploader">
      <h3>📁 Import Robot (URDF)</h3>
      <p className="description">Upload a URDF file to import a custom robot</p>

      <div className="upload-area">
        <input
          type="file"
          accept=".urdf"
          onChange={handleFileUpload}
          disabled={uploading}
          id="urdf-file-input"
        />
        <label htmlFor="urdf-file-input" className="upload-button">
          {uploading ? '⏳ Parsing...' : '📤 Choose URDF File'}
        </label>
      </div>

      {error && (
        <div className="error-message">
          ❌ {error}
        </div>
      )}

      {robotInfo && (
        <div className="robot-info">
          <h4>✅ Robot Loaded: {robotInfo.name}</h4>
          <div className="info-grid">
            <div className="info-item">
              <span className="label">Links:</span>
              <span className="value">{robotInfo.links?.length || 0}</span>
            </div>
            <div className="info-item">
              <span className="label">Joints:</span>
              <span className="value">{robotInfo.joints?.length || 0}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
