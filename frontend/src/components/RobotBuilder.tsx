import { useState } from 'react'
import axios from 'axios'
import { config } from '../config'
import './RobotBuilder.css'

interface Link {
  id: string
  name: string
  visual: {
    geometry: {
      type: 'box' | 'cylinder' | 'sphere'
      size?: number[]
      radius?: number
      length?: number
    }
    color?: number[]
    origin?: {
      xyz: number[]
      rpy: number[]
    }
  }
}

interface Joint {
  id: string
  name: string
  type: 'fixed' | 'revolute' | 'continuous' | 'prismatic'
  parent: string
  child: string
  origin?: {
    xyz: number[]
    rpy: number[]
  }
  axis?: number[]
  limits?: {
    lower: number
    upper: number
    effort: number
    velocity: number
  }
}

interface RobotBuilderProps {
  onRobotGenerated: (sceneConfig: any) => void
}

export default function RobotBuilder({ onRobotGenerated }: RobotBuilderProps) {
  const [robotName, setRobotName] = useState('custom_robot')
  const [links, setLinks] = useState<Link[]>([
    {
      id: 'link-0',
      name: 'base_link',
      visual: {
        geometry: { type: 'box', size: [0.3, 0.3, 0.1] },
        color: [0.5, 0.5, 0.5, 1.0]
      }
    }
  ])
  const [joints, setJoints] = useState<Joint[]>([])
  const [selectedLinkId, setSelectedLinkId] = useState<string | null>('link-0')
  const [selectedJointId, setSelectedJointId] = useState<string | null>(null)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showPresets, setShowPresets] = useState(false)

  // Add new link
  const addLink = () => {
    const newLink: Link = {
      id: `link-${links.length}`,
      name: `link${links.length}`,
      visual: {
        geometry: { type: 'box', size: [0.1, 0.1, 0.1] },
        color: [0.7, 0.7, 0.7, 1.0]
      }
    }
    setLinks([...links, newLink])
    setSelectedLinkId(newLink.id)
  }

  // Add new joint
  const addJoint = () => {
    if (links.length < 2) {
      setError('Need at least 2 links to create a joint')
      return
    }

    const newJoint: Joint = {
      id: `joint-${joints.length}`,
      name: `joint${joints.length}`,
      type: 'revolute',
      parent: links[0].name,
      child: links[1].name,
      origin: { xyz: [0, 0, 0.05], rpy: [0, 0, 0] },
      axis: [0, 0, 1],
      limits: { lower: -1.57, upper: 1.57, effort: 100, velocity: 1.0 }
    }
    setJoints([...joints, newJoint])
    setSelectedJointId(newJoint.id)
  }

  // Update link
  const updateLink = (linkId: string, updates: Partial<Link>) => {
    setLinks(links.map(link =>
      link.id === linkId ? { ...link, ...updates } : link
    ))
  }

  // Update joint
  const updateJoint = (jointId: string, updates: Partial<Joint>) => {
    setJoints(joints.map(joint =>
      joint.id === jointId ? { ...joint, ...updates } : joint
    ))
  }

  // Delete link
  const deleteLink = (linkId: string) => {
    const linkName = links.find(l => l.id === linkId)?.name
    // Remove joints that reference this link
    setJoints(joints.filter(j => j.parent !== linkName && j.child !== linkName))
    setLinks(links.filter(l => l.id !== linkId))
    if (selectedLinkId === linkId) setSelectedLinkId(null)
  }

  // Delete joint
  const deleteJoint = (jointId: string) => {
    setJoints(joints.filter(j => j.id !== jointId))
    if (selectedJointId === jointId) setSelectedJointId(null)
  }

  // Load preset
  const loadPreset = async (presetId: string) => {
    try {
      const response = await axios.get(`${config.backendUrl}/api/urdf/presets/${presetId}`)
      onRobotGenerated(response.data.scene_config)
      setShowPresets(false)
    } catch (error) {
      console.error('Failed to load preset:', error)
      setError('Failed to load preset')
    }
  }

  // Generate robot
  const generateRobot = async () => {
    setGenerating(true)
    setError(null)

    try {
      const response = await axios.post(`${config.backendUrl}/api/urdf/generate`, {
        robot_name: robotName,
        links: links.map(link => ({
          name: link.name,
          visual: link.visual
        })),
        joints: joints.map(joint => ({
          name: joint.name,
          type: joint.type,
          parent: joint.parent,
          child: joint.child,
          origin: joint.origin,
          axis: joint.axis,
          limits: joint.limits
        }))
      })

      console.log('✅ Robot generated:', response.data)
      onRobotGenerated(response.data.scene_config)

    } catch (error: any) {
      console.error('❌ Generation failed:', error)
      setError(error.response?.data?.detail || 'Failed to generate robot')
    } finally {
      setGenerating(false)
    }
  }

  const selectedLink = links.find(l => l.id === selectedLinkId)
  const selectedJoint = joints.find(j => j.id === selectedJointId)

  return (
    <div className="robot-builder">
      <div className="builder-header">
        <h3>🔧 Robot Builder</h3>
        <p>Design custom robots visually</p>
      </div>

      {/* Preset Templates */}
      <div className="preset-section">
        <button
          className="preset-toggle"
          onClick={() => setShowPresets(!showPresets)}
        >
          📚 {showPresets ? 'Hide' : 'Show'} Presets
        </button>

        {showPresets && (
          <div className="preset-grid">
            <button className="preset-card" onClick={() => loadPreset('simple_arm')}>
              <div className="preset-icon">🦾</div>
              <div className="preset-name">Simple Arm</div>
            </button>
            <button className="preset-card" onClick={() => loadPreset('mobile_base')}>
              <div className="preset-icon">🤖</div>
              <div className="preset-name">Mobile Base</div>
            </button>
          </div>
        )}
      </div>

      {/* Robot Name */}
      <div className="robot-name-section">
        <label>Robot Name:</label>
        <input
          type="text"
          value={robotName}
          onChange={(e) => setRobotName(e.target.value)}
          className="robot-name-input"
        />
      </div>

      {/* Structure Overview */}
      <div className="structure-section">
        <div className="structure-header">
          <h4>Links ({links.length})</h4>
          <button className="add-btn" onClick={addLink}>+ Add Link</button>
        </div>

        <div className="links-list">
          {links.map(link => (
            <div
              key={link.id}
              className={`link-item ${selectedLinkId === link.id ? 'selected' : ''}`}
              onClick={() => setSelectedLinkId(link.id)}
            >
              <span className="link-icon">🔗</span>
              <span className="link-name">{link.name}</span>
              <span className="link-type">{link.visual.geometry.type}</span>
              {links.length > 1 && (
                <button
                  className="delete-btn"
                  onClick={(e) => {
                    e.stopPropagation()
                    deleteLink(link.id)
                  }}
                >
                  ×
                </button>
              )}
            </div>
          ))}
        </div>

        <div className="structure-header">
          <h4>Joints ({joints.length})</h4>
          <button className="add-btn" onClick={addJoint}>+ Add Joint</button>
        </div>

        <div className="joints-list">
          {joints.map(joint => (
            <div
              key={joint.id}
              className={`joint-item ${selectedJointId === joint.id ? 'selected' : ''}`}
              onClick={() => {
                setSelectedJointId(joint.id)
                setSelectedLinkId(null)
              }}
            >
              <span className="joint-icon">⚙️</span>
              <span className="joint-name">{joint.name}</span>
              <span className="joint-type">{joint.type}</span>
              <button
                className="delete-btn"
                onClick={(e) => {
                  e.stopPropagation()
                  deleteJoint(joint.id)
                }}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Link Editor */}
      {selectedLink && (
        <div className="editor-section">
          <h4>Edit Link: {selectedLink.name}</h4>

          <div className="form-group">
            <label>Name:</label>
            <input
              type="text"
              value={selectedLink.name}
              onChange={(e) => updateLink(selectedLink.id, { name: e.target.value })}
            />
          </div>

          <div className="form-group">
            <label>Geometry Type:</label>
            <select
              value={selectedLink.visual.geometry.type}
              onChange={(e) => {
                const type = e.target.value as 'box' | 'cylinder' | 'sphere'
                const newGeometry = type === 'box'
                  ? { type, size: [0.1, 0.1, 0.1] }
                  : type === 'cylinder'
                  ? { type, radius: 0.05, length: 0.2 }
                  : { type, radius: 0.05 }

                updateLink(selectedLink.id, {
                  visual: { ...selectedLink.visual, geometry: newGeometry }
                })
              }}
            >
              <option value="box">Box</option>
              <option value="cylinder">Cylinder</option>
              <option value="sphere">Sphere</option>
            </select>
          </div>

          {selectedLink.visual.geometry.type === 'box' && (
            <div className="form-group">
              <label>Size [X, Y, Z]:</label>
              <div className="vector-input">
                {selectedLink.visual.geometry.size?.map((val, i) => (
                  <input
                    key={i}
                    type="number"
                    step="0.01"
                    value={val}
                    onChange={(e) => {
                      const newSize = [...(selectedLink.visual.geometry.size || [0.1, 0.1, 0.1])]
                      newSize[i] = parseFloat(e.target.value) || 0
                      updateLink(selectedLink.id, {
                        visual: {
                          ...selectedLink.visual,
                          geometry: { ...selectedLink.visual.geometry, size: newSize }
                        }
                      })
                    }}
                  />
                ))}
              </div>
            </div>
          )}

          {selectedLink.visual.geometry.type === 'cylinder' && (
            <>
              <div className="form-group">
                <label>Radius:</label>
                <input
                  type="number"
                  step="0.01"
                  value={selectedLink.visual.geometry.radius || 0.05}
                  onChange={(e) => {
                    updateLink(selectedLink.id, {
                      visual: {
                        ...selectedLink.visual,
                        geometry: { ...selectedLink.visual.geometry, radius: parseFloat(e.target.value) || 0.05 }
                      }
                    })
                  }}
                />
              </div>
              <div className="form-group">
                <label>Length:</label>
                <input
                  type="number"
                  step="0.01"
                  value={selectedLink.visual.geometry.length || 0.2}
                  onChange={(e) => {
                    updateLink(selectedLink.id, {
                      visual: {
                        ...selectedLink.visual,
                        geometry: { ...selectedLink.visual.geometry, length: parseFloat(e.target.value) || 0.2 }
                      }
                    })
                  }}
                />
              </div>
            </>
          )}

          {selectedLink.visual.geometry.type === 'sphere' && (
            <div className="form-group">
              <label>Radius:</label>
              <input
                type="number"
                step="0.01"
                value={selectedLink.visual.geometry.radius || 0.05}
                onChange={(e) => {
                  updateLink(selectedLink.id, {
                    visual: {
                      ...selectedLink.visual,
                      geometry: { ...selectedLink.visual.geometry, radius: parseFloat(e.target.value) || 0.05 }
                    }
                  })
                }}
              />
            </div>
          )}

          <div className="form-group">
            <label>Color (RGB):</label>
            <div className="color-input">
              <input
                type="color"
                value={`#${selectedLink.visual.color?.slice(0, 3).map(c =>
                  Math.round(c * 255).toString(16).padStart(2, '0')
                ).join('')}`}
                onChange={(e) => {
                  const hex = e.target.value
                  const r = parseInt(hex.slice(1, 3), 16) / 255
                  const g = parseInt(hex.slice(3, 5), 16) / 255
                  const b = parseInt(hex.slice(5, 7), 16) / 255
                  updateLink(selectedLink.id, {
                    visual: { ...selectedLink.visual, color: [r, g, b, 1.0] }
                  })
                }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Joint Editor */}
      {selectedJoint && (
        <div className="editor-section">
          <h4>Edit Joint: {selectedJoint.name}</h4>

          <div className="form-group">
            <label>Name:</label>
            <input
              type="text"
              value={selectedJoint.name}
              onChange={(e) => updateJoint(selectedJoint.id, { name: e.target.value })}
            />
          </div>

          <div className="form-group">
            <label>Type:</label>
            <select
              value={selectedJoint.type}
              onChange={(e) => updateJoint(selectedJoint.id, { type: e.target.value as any })}
            >
              <option value="fixed">Fixed</option>
              <option value="revolute">Revolute</option>
              <option value="continuous">Continuous</option>
              <option value="prismatic">Prismatic</option>
            </select>
          </div>

          <div className="form-group">
            <label>Parent Link:</label>
            <select
              value={selectedJoint.parent}
              onChange={(e) => updateJoint(selectedJoint.id, { parent: e.target.value })}
            >
              {links.map(link => (
                <option key={link.name} value={link.name}>{link.name}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Child Link:</label>
            <select
              value={selectedJoint.child}
              onChange={(e) => updateJoint(selectedJoint.id, { child: e.target.value })}
            >
              {links.map(link => (
                <option key={link.name} value={link.name}>{link.name}</option>
              ))}
            </select>
          </div>

          {selectedJoint.type !== 'fixed' && (
            <>
              <div className="form-group">
                <label>Axis [X, Y, Z]:</label>
                <div className="vector-input">
                  {selectedJoint.axis?.map((val, i) => (
                    <input
                      key={i}
                      type="number"
                      step="0.1"
                      value={val}
                      onChange={(e) => {
                        const newAxis = [...(selectedJoint.axis || [0, 0, 1])]
                        newAxis[i] = parseFloat(e.target.value) || 0
                        updateJoint(selectedJoint.id, { axis: newAxis })
                      }}
                    />
                  ))}
                </div>
              </div>

              {selectedJoint.type !== 'continuous' && (
                <>
                  <div className="form-group">
                    <label>Lower Limit:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={selectedJoint.limits?.lower || -1.57}
                      onChange={(e) => {
                        updateJoint(selectedJoint.id, {
                          limits: { ...(selectedJoint.limits || { upper: 1.57, effort: 100, velocity: 1.0 }), lower: parseFloat(e.target.value) }
                        })
                      }}
                    />
                  </div>
                  <div className="form-group">
                    <label>Upper Limit:</label>
                    <input
                      type="number"
                      step="0.1"
                      value={selectedJoint.limits?.upper || 1.57}
                      onChange={(e) => {
                        updateJoint(selectedJoint.id, {
                          limits: { ...(selectedJoint.limits || { lower: -1.57, effort: 100, velocity: 1.0 }), upper: parseFloat(e.target.value) }
                        })
                      }}
                    />
                  </div>
                </>
              )}
            </>
          )}
        </div>
      )}

      {/* Generate Button */}
      <div className="generate-section">
        {error && (
          <div className="error-message">
            ❌ {error}
          </div>
        )}

        <button
          className="generate-btn"
          onClick={generateRobot}
          disabled={generating || links.length === 0}
        >
          {generating ? '⏳ Generating...' : '🚀 Generate Robot'}
        </button>
      </div>
    </div>
  )
}
