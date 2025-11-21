import { useState, useEffect } from 'react'
import { config } from '../config'
import './AlgorithmLibrary.css'

interface AlgorithmTemplate {
  id: string
  name: string
  type: string
  description: string
  complexity: string
  parameters: any[]
  code_template: string
}

interface AlgorithmLibraryProps {
  isOpen: boolean
  onClose: () => void
  onUseTemplate: (template: AlgorithmTemplate) => void
  onCustomize: (template: AlgorithmTemplate) => void
}

export default function AlgorithmLibrary({
  isOpen,
  onClose,
  onUseTemplate,
  onCustomize
}: AlgorithmLibraryProps) {
  const [templates, setTemplates] = useState<AlgorithmTemplate[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedCategory, setSelectedCategory] = useState<string>('all')

  useEffect(() => {
    if (isOpen) {
      fetchTemplates()
    }
  }, [isOpen])

  const fetchTemplates = async () => {
    setLoading(true)
    try {
      const response = await fetch(`${config.backendUrl}/api/algorithms`)
      const data = await response.json()
      setTemplates(data.templates || [])
    } catch (error) {
      console.error('Failed to fetch algorithm templates:', error)
    } finally {
      setLoading(false)
    }
  }

  const categories = [
    { id: 'all', label: 'All Algorithms' },
    { id: 'path_planning', label: 'Path Planning' },
    { id: 'obstacle_avoidance', label: 'Obstacle Avoidance' },
    { id: 'inverse_kinematics', label: 'Inverse Kinematics' },
    { id: 'computer_vision', label: 'Computer Vision' },
    { id: 'motion_control', label: 'Motion Control' }
  ]

  const filteredTemplates = selectedCategory === 'all'
    ? templates
    : templates.filter(t => t.type === selectedCategory)

  if (!isOpen) return null

  return (
    <div className="algorithm-library-overlay" onClick={onClose}>
      <div className="algorithm-library-modal" onClick={(e) => e.stopPropagation()}>
        <div className="library-header">
          <h2>Algorithm Library</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>

        <div className="library-categories">
          {categories.map(cat => (
            <button
              key={cat.id}
              className={`category-button ${selectedCategory === cat.id ? 'active' : ''}`}
              onClick={() => setSelectedCategory(cat.id)}
            >
              {cat.label}
            </button>
          ))}
        </div>

        <div className="library-content">
          {loading ? (
            <div className="library-loading">Loading templates...</div>
          ) : filteredTemplates.length === 0 ? (
            <div className="library-empty">No templates found in this category</div>
          ) : (
            <div className="templates-grid">
              {filteredTemplates.map(template => (
                <div key={template.id} className="template-card">
                  <div className="template-header">
                    <h3>{template.name}</h3>
                    <span className={`template-badge ${template.type}`}>
                      {template.type.replace('_', ' ')}
                    </span>
                  </div>

                  <p className="template-description">{template.description}</p>

                  <div className="template-metadata">
                    <div className="metadata-item">
                      <span className="metadata-label">Complexity:</span>
                      <span className="metadata-value">{template.complexity}</span>
                    </div>
                    <div className="metadata-item">
                      <span className="metadata-label">Parameters:</span>
                      <span className="metadata-value">{template.parameters.length}</span>
                    </div>
                  </div>

                  <div className="template-actions">
                    <button
                      className="template-button use"
                      onClick={() => onUseTemplate(template)}
                    >
                      Use Template
                    </button>
                    <button
                      className="template-button customize"
                      onClick={() => onCustomize(template)}
                    >
                      Customize with AI
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
