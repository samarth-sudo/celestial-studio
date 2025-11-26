/**
 * Camera Selector Component
 * Allows switching between different Genesis camera views
 */

import React, { useEffect, useState } from 'react';
import './CameraSelector.css';

interface CameraSelectorProps {
  /** WebSocket connection for sending camera commands */
  websocket: WebSocket | null;
  /** Currently active camera */
  activeCamera?: string;
  /** Available cameras */
  cameras?: string[];
  /** Show as compact buttons or full panel */
  compact?: boolean;
  /** Callback when camera changes */
  onCameraChange?: (camera: string) => void;
}

interface CameraInfo {
  name: string;
  label: string;
  icon: string;
  description: string;
}

const CAMERA_INFO: Record<string, CameraInfo> = {
  main: {
    name: 'main',
    label: 'Third-Person',
    icon: '🎥',
    description: 'Standard third-person view of the scene'
  },
  fpv: {
    name: 'fpv',
    label: 'First-Person',
    icon: '👁️',
    description: 'First-person view from robot perspective'
  },
  top: {
    name: 'top',
    label: 'Top-Down',
    icon: '🔼',
    description: 'Bird\'s eye view from above'
  },
  debug: {
    name: 'debug',
    label: 'Debug',
    icon: '🔧',
    description: 'Debug camera for close-up inspection'
  }
};

const CameraSelector: React.FC<CameraSelectorProps> = ({
  websocket,
  activeCamera = 'main',
  cameras = [],
  compact = false,
  onCameraChange
}) => {
  const [selectedCamera, setSelectedCamera] = useState(activeCamera);
  const [availableCameras, setAvailableCameras] = useState<string[]>(cameras);

  useEffect(() => {
    setSelectedCamera(activeCamera);
  }, [activeCamera]);

  useEffect(() => {
    if (cameras.length > 0) {
      setAvailableCameras(cameras);
    }
  }, [cameras]);

  // Request camera list on mount
  useEffect(() => {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      websocket.send(JSON.stringify({ type: 'get_cameras' }));
    }
  }, [websocket]);

  const switchCamera = (cameraName: string) => {
    if (!websocket || websocket.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected');
      return;
    }

    // Send camera switch command
    websocket.send(JSON.stringify({
      type: 'switch_camera',
      camera: cameraName
    }));

    setSelectedCamera(cameraName);

    if (onCameraChange) {
      onCameraChange(cameraName);
    }

    console.log(`📷 Switching to ${cameraName} camera`);
  };

  if (compact) {
    return (
      <div className="camera-selector-compact">
        {availableCameras.map((camera) => {
          const info = CAMERA_INFO[camera] || {
            name: camera,
            label: camera,
            icon: '📷',
            description: camera
          };

          return (
            <button
              key={camera}
              className={`camera-btn ${selectedCamera === camera ? 'active' : ''}`}
              onClick={() => switchCamera(camera)}
              title={info.description}
              disabled={!websocket}
            >
              <span className="camera-icon">{info.icon}</span>
              <span className="camera-label">{info.label}</span>
            </button>
          );
        })}
      </div>
    );
  }

  return (
    <div className="camera-selector">
      <div className="camera-selector-header">
        <h3>📷 Camera View</h3>
        <span className="camera-count">{availableCameras.length} cameras</span>
      </div>

      <div className="camera-grid">
        {availableCameras.map((camera) => {
          const info = CAMERA_INFO[camera] || {
            name: camera,
            label: camera,
            icon: '📷',
            description: camera
          };

          return (
            <button
              key={camera}
              className={`camera-card ${selectedCamera === camera ? 'active' : ''}`}
              onClick={() => switchCamera(camera)}
              disabled={!websocket}
            >
              <div className="camera-icon-large">{info.icon}</div>
              <div className="camera-name">{info.label}</div>
              <div className="camera-description">{info.description}</div>
              {selectedCamera === camera && (
                <div className="active-indicator">✓ Active</div>
              )}
            </button>
          );
        })}
      </div>

      {!websocket && (
        <div className="connection-warning">
          ⚠️ Not connected to Genesis stream
        </div>
      )}
    </div>
  );
};

export default CameraSelector;
