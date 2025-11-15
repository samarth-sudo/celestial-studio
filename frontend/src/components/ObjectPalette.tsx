import type { EditorTool } from './SceneEditor'

interface ObjectPaletteProps {
  isVisible: boolean
  onClose: () => void
  activeTool: EditorTool
  onToolSelect: (tool: EditorTool) => void
}

interface ObjectType {
  tool: EditorTool
  name: string
  icon: string
  description: string
  category: 'markers' | 'obstacles'
  color: string
}

const AVAILABLE_OBJECTS: ObjectType[] = [
  {
    tool: 'add_origin',
    name: 'Origin',
    icon: '🟢',
    description: 'Start/origin point',
    category: 'markers',
    color: '#00ff00'
  },
  {
    tool: 'add_destination',
    name: 'Destination',
    icon: '🔴',
    description: 'Goal/destination point',
    category: 'markers',
    color: '#ff0000'
  },
  {
    tool: 'add_box',
    name: 'Box',
    icon: '📦',
    description: 'Box obstacle',
    category: 'obstacles',
    color: '#ffa500'
  },
  {
    tool: 'add_cylinder',
    name: 'Cylinder',
    icon: '🛢️',
    description: 'Cylinder obstacle',
    category: 'obstacles',
    color: '#808080'
  }
]

export default function ObjectPalette({
  isVisible,
  onClose,
  activeTool,
  onToolSelect
}: ObjectPaletteProps) {
  if (!isVisible) return null

  const markerObjects = AVAILABLE_OBJECTS.filter(obj => obj.category === 'markers')
  const obstacleObjects = AVAILABLE_OBJECTS.filter(obj => obj.category === 'obstacles')

  return (
    <div className="object-palette-window">
      <div className="window-header">
        <h4>➕ Add Objects</h4>
        <button className="window-close" onClick={onClose} title="Close palette">
          ×
        </button>
      </div>

      <div className="palette-content">
        <div className="palette-section">
          <h5 className="section-title">Path Markers</h5>
          <div className="object-grid">
            {markerObjects.map((obj) => (
              <button
                key={obj.tool}
                className={`object-card ${activeTool === obj.tool ? 'active' : ''}`}
                onClick={() => onToolSelect(obj.tool)}
                title={obj.description}
              >
                <div className="object-preview" style={{ backgroundColor: obj.color }}>
                  <span className="object-icon">{obj.icon}</span>
                </div>
                <span className="object-name">{obj.name}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="palette-section">
          <h5 className="section-title">Obstacles</h5>
          <div className="object-grid">
            {obstacleObjects.map((obj) => (
              <button
                key={obj.tool}
                className={`object-card ${activeTool === obj.tool ? 'active' : ''}`}
                onClick={() => onToolSelect(obj.tool)}
                title={obj.description}
              >
                <div className="object-preview" style={{ backgroundColor: obj.color }}>
                  <span className="object-icon">{obj.icon}</span>
                </div>
                <span className="object-name">{obj.name}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="palette-instructions">
          <p>💡 Select an object, then click on the scene to place it</p>
        </div>
      </div>
    </div>
  )
}
