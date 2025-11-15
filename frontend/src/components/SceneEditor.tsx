import { useState } from 'react'
import './SceneEditor.css'

export type EditorTool = 'select' | 'add_origin' | 'add_destination' | 'add_box' | 'add_cylinder' | 'delete'

interface SceneEditorProps {
  activeTool: EditorTool
  onToolChange: (tool: EditorTool) => void
  onClearScene: () => void
  onUndo: () => void
  objectCount: number
}

export default function SceneEditor({
  activeTool,
  onToolChange,
  onClearScene,
  onUndo,
  objectCount
}: SceneEditorProps) {
  return (
    <div className="scene-editor">
      <div className="editor-header">
        <h3>Scene Editor</h3>
        <span className="object-count">{objectCount} objects</span>
      </div>

      <div className="editor-toolbar">
        <button
          className={`tool-button ${activeTool === 'select' ? 'active' : ''}`}
          onClick={() => onToolChange('select')}
          title="Select and move objects"
        >
          <span className="tool-icon">🖱️</span>
          <span className="tool-label">Select</span>
        </button>

        <button
          className={`tool-button ${activeTool === 'add_origin' ? 'active' : ''}`}
          onClick={() => onToolChange('add_origin')}
          title="Add origin/start marker"
        >
          <span className="tool-icon">🟢</span>
          <span className="tool-label">Origin</span>
        </button>

        <button
          className={`tool-button ${activeTool === 'add_destination' ? 'active' : ''}`}
          onClick={() => onToolChange('add_destination')}
          title="Add destination/goal marker"
        >
          <span className="tool-icon">🔴</span>
          <span className="tool-label">Goal</span>
        </button>

        <button
          className={`tool-button ${activeTool === 'add_box' ? 'active' : ''}`}
          onClick={() => onToolChange('add_box')}
          title="Add box obstacle"
        >
          <span className="tool-icon">📦</span>
          <span className="tool-label">Box</span>
        </button>

        <button
          className={`tool-button ${activeTool === 'add_cylinder' ? 'active' : ''}`}
          onClick={() => onToolChange('add_cylinder')}
          title="Add cylinder obstacle"
        >
          <span className="tool-icon">🛢️</span>
          <span className="tool-label">Cylinder</span>
        </button>

        <button
          className={`tool-button ${activeTool === 'delete' ? 'active' : ''}`}
          onClick={() => onToolChange('delete')}
          title="Delete objects"
        >
          <span className="tool-icon">🗑️</span>
          <span className="tool-label">Delete</span>
        </button>
      </div>

      <div className="editor-actions">
        <button
          className="action-button undo-button"
          onClick={onUndo}
          title="Undo last action"
        >
          ↶ Undo
        </button>

        <button
          className="action-button clear-button"
          onClick={onClearScene}
          title="Clear all objects"
        >
          🗑️ Clear All
        </button>
      </div>

      <div className="editor-help">
        <p>
          {activeTool === 'select' && '👆 Click to select, drag to move objects'}
          {activeTool === 'add_origin' && '👆 Click on the scene to add origin marker'}
          {activeTool === 'add_destination' && '👆 Click on the scene to add goal marker'}
          {activeTool === 'add_box' && '👆 Click on the scene to add a box'}
          {activeTool === 'add_cylinder' && '👆 Click on the scene to add a cylinder'}
          {activeTool === 'delete' && '👆 Click on an object to delete it'}
        </p>
      </div>
    </div>
  )
}
