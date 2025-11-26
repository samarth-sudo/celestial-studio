/**
 * Genesis Viewer Component
 * Displays video stream from Genesis physics engine running on backend
 */

import React, { useEffect, useRef, useState } from 'react';
import './GenesisViewer.css';

interface GenesisViewerProps {
  /** WebSocket connection status */
  connected?: boolean;
  /** Callback when connection state changes */
  onConnectionChange?: (connected: boolean) => void;
  /** Backend URL */
  backendUrl?: string;
  /** Auto-start simulation on mount */
  autoStart?: boolean;
  /** Stream quality preset */
  quality?: 'draft' | 'medium' | 'high' | 'ultra';
  /** Show debug overlay */
  showDebug?: boolean;
}

interface SimulationState {
  timestamp: number;
  step: number;
  fps: number;
  robots: Record<string, any>;
  obstacles: Record<string, any>;
}

interface StreamStats {
  runtime_seconds: number;
  frames_delivered: number;
  bytes_delivered: number;
  avg_fps: number;
  avg_bitrate_mbps: number;
}

const GenesisViewer: React.FC<GenesisViewerProps> = ({
  connected: externalConnected,
  onConnectionChange,
  backendUrl = 'http://localhost:8000',
  autoStart = true,
  quality = 'medium',
  showDebug = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const [connected, setConnected] = useState(false);
  const [simState, setSimState] = useState<SimulationState | null>(null);
  const [streamStats, setStreamStats] = useState<StreamStats | null>(null);
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);

  const frameCountRef = useRef(0);
  const lastFrameTimeRef = useRef(Date.now());
  const fpsIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize Genesis simulation
  useEffect(() => {
    const initGenesisSimulation = async () => {
      try {
        console.log('🚀 Initializing Genesis simulation...');

        // Check if Genesis is available
        const statusRes = await fetch(`${backendUrl}/api/genesis/status`);
        const status = await statusRes.json();

        if (!status.available) {
          console.error('❌ Genesis not available:', status.message);
          return;
        }

        // Initialize simulation
        const initRes = await fetch(`${backendUrl}/api/genesis/init`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            backend: 'auto',
            fps: 60,
            render_width: 1920,
            render_height: 1080,
            stream_quality: quality,
          }),
        });

        const initResult = await initRes.json();
        console.log('✅ Genesis initialized:', initResult);

        // Add a test robot
        await fetch(`${backendUrl}/api/genesis/robot/add`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            robot_id: 'robot1',
            robot_type: 'mobile',
            position: [0, 0, 0.5],
          }),
        });

        // Add some obstacles
        await fetch(`${backendUrl}/api/genesis/obstacle/add`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            obstacle_id: 'obstacle1',
            position: [2, 2, 0.25],
            size: [0.5, 0.5, 0.5],
          }),
        });

        // Build scene
        await fetch(`${backendUrl}/api/genesis/scene/build`, {
          method: 'POST',
        });

        console.log('✅ Scene built');

        // Start simulation if autoStart
        if (autoStart) {
          await fetch(`${backendUrl}/api/genesis/control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'start' }),
          });

          console.log('✅ Simulation started');
        }

        // Connect WebSocket
        connectWebSocket();
      } catch (error) {
        console.error('❌ Genesis initialization error:', error);
      }
    };

    initGenesisSimulation();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [backendUrl, quality, autoStart]);

  // WebSocket connection
  const connectWebSocket = () => {
    const wsUrl = backendUrl.replace('http://', 'ws://').replace('https://', 'wss://');
    // UPDATED: Use new Genesis video streaming endpoint
    const ws = new WebSocket(`${wsUrl}/ws/genesis/video`);

    ws.onopen = () => {
      console.log('🔌 Genesis video stream connected');
      setConnected(true);
      if (onConnectionChange) {
        onConnectionChange(true);
      }

      // Request camera list
      ws.send(JSON.stringify({ type: 'get_cameras' }));
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);

        if (message.type === 'frame') {
          // Render frame (frames are pushed automatically, no request needed)
          renderFrame(message.image);

          // Update simulation state from frame metadata
          if (message.state) {
            setSimState({
              timestamp: message.timestamp,
              step: message.state.step,
              fps: message.state.fps,
              robots: message.state.robots,
              obstacles: {}
            });
          }
        } else if (message.type === 'cameras_list') {
          // Camera list received
          console.log('📷 Available cameras:', message.cameras);
        } else if (message.type === 'state_update') {
          setSimState(message.data);
        } else if (message.type === 'initial_state') {
          setSimState(message.data);
        }
      } catch (error) {
        console.error('WebSocket message error:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('🔌 WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('🔌 WebSocket disconnected');
      setConnected(false);
      if (onConnectionChange) {
        onConnectionChange(false);
      }

      // Attempt reconnection after 2 seconds
      setTimeout(() => {
        console.log('🔄 Attempting to reconnect...');
        connectWebSocket();
      }, 2000);
    };

    wsRef.current = ws;
  };

  // Render frame to canvas
  const renderFrame = (base64Data: string) => {
    const canvas = canvasRef.current;
    const img = imgRef.current;

    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Update FPS counter
    frameCountRef.current++;
    const now = Date.now();
    const elapsed = now - lastFrameTimeRef.current;

    if (elapsed >= 1000) {
      const currentFps = (frameCountRef.current / elapsed) * 1000;
      setFps(Math.round(currentFps));
      frameCountRef.current = 0;
      lastFrameTimeRef.current = now;
    }

    // Load and draw image
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      // Calculate latency (simplified)
      const latencyMs = Date.now() - now;
      setLatency(latencyMs);
    };

    img.src = `data:image/jpeg;base64,${base64Data}`;
  };

  // NOTE: Frame requests removed - new streaming endpoint pushes frames automatically at 30 FPS

  // Fetch stream stats periodically
  useEffect(() => {
    if (!connected) return;

    const fetchStats = async () => {
      try {
        const res = await fetch(`${backendUrl}/api/genesis/stream/stats`);
        const data = await res.json();
        if (data.status === 'running') {
          setStreamStats(data.stats);
        }
      } catch (error) {
        console.error('Failed to fetch stats:', error);
      }
    };

    const intervalId = setInterval(fetchStats, 2000);
    fetchStats(); // Initial fetch

    return () => {
      clearInterval(intervalId);
    };
  }, [connected, backendUrl]);

  // Control simulation
  const sendControl = (action: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'control',
        action: action,
      }));
    }
  };

  return (
    <div className="genesis-viewer">
      {/* Video Canvas */}
      <div className="video-container">
        <canvas ref={canvasRef} className="video-canvas" />
        <img ref={imgRef} style={{ display: 'none' }} alt="" />

        {/* Connection Status Overlay */}
        {!connected && (
          <div className="connection-overlay">
            <div className="spinner" />
            <p>Connecting to Genesis backend...</p>
          </div>
        )}

        {/* Debug Overlay */}
        {showDebug && connected && (
          <div className="debug-overlay">
            <div className="debug-section">
              <h4>🎮 Simulation</h4>
              <p>Step: {simState?.step || 0}</p>
              <p>Sim FPS: {simState?.fps.toFixed(1) || 0}</p>
              <p>Robots: {Object.keys(simState?.robots || {}).length}</p>
              <p>Obstacles: {Object.keys(simState?.obstacles || {}).length}</p>
            </div>

            <div className="debug-section">
              <h4>📹 Stream</h4>
              <p>Video FPS: {fps}</p>
              <p>Latency: {latency}ms</p>
              {streamStats && (
                <>
                  <p>Frames: {streamStats.frames_delivered}</p>
                  <p>Bitrate: {streamStats.avg_bitrate_mbps.toFixed(2)} Mbps</p>
                </>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="controls">
        <button
          onClick={() => sendControl('start')}
          className="control-btn"
          disabled={!connected}
        >
          ▶️ Start
        </button>
        <button
          onClick={() => sendControl('stop')}
          className="control-btn"
          disabled={!connected}
        >
          ⏸️ Pause
        </button>
        <button
          onClick={() => sendControl('reset')}
          className="control-btn"
          disabled={!connected}
        >
          🔄 Reset
        </button>
      </div>

      {/* Status Bar */}
      <div className="status-bar">
        <span className={`status-indicator ${connected ? 'connected' : 'disconnected'}`}>
          {connected ? '🟢 Connected' : '🔴 Disconnected'}
        </span>
        <span className="fps-counter">{fps} FPS</span>
        {simState && (
          <span className="step-counter">Step: {simState.step}</span>
        )}
      </div>
    </div>
  );
};

export default GenesisViewer;
