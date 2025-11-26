/**
 * Teleoperation Test Page
 *
 * Tests keyboard-based robot control with:
 * - WebSocket connection to Genesis backend
 * - Real-time keyboard input
 * - Trajectory recording
 * - Visual feedback
 */

import { useEffect, useRef, useState } from 'react';
import { ControlOverlay } from '../components/ControlOverlay';
import { KeyboardController } from '../services/KeyboardController';
import type { TeleopUpdate } from '../services/KeyboardController';
import './TeleopTest.css';

type RobotType = 'mobile' | 'arm' | 'drone';

export default function TeleopTest() {
  const [robotType, setRobotType] = useState<RobotType>('mobile');
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSteps, setRecordingSteps] = useState(0);
  const [recordingTime, setRecordingTime] = useState(0);
  const [currentAction, setCurrentAction] = useState<number[]>([]);
  const [currentState, setCurrentState] = useState<any>(null);
  const [logs, setLogs] = useState<string[]>([]);

  const controllerRef = useRef<KeyboardController | null>(null);
  const recordingStartTimeRef = useRef<number>(0);
  const recordingIntervalRef = useRef<number | null>(null);

  // Add log entry
  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [`[${timestamp}] ${message}`, ...prev.slice(0, 49)]);
  };

  // Initialize keyboard controller
  useEffect(() => {
    const wsUrl = 'ws://localhost:8000/api/control/teleop';
    const controller = new KeyboardController(wsUrl, 'robot-1', robotType);
    controllerRef.current = controller;

    // Set up callbacks
    controller.onConnect(() => {
      setIsConnected(true);
      addLog(`✅ Connected to teleoperation server (${robotType})`);
    });

    controller.onDisconnect(() => {
      setIsConnected(false);
      setIsRecording(false);
      addLog('🔌 Disconnected from server');
    });

    controller.onUpdate((update: TeleopUpdate) => {
      setCurrentAction(update.result.action);
      setCurrentState(update.result.state);

      if (update.result.recording) {
        setRecordingSteps(update.result.steps_recorded || 0);
      }
    });

    controller.onRecordingStart((taskName: string) => {
      setIsRecording(true);
      setRecordingSteps(0);
      recordingStartTimeRef.current = Date.now();
      addLog(`🔴 Recording started: ${taskName}`);

      // Start recording timer
      recordingIntervalRef.current = window.setInterval(() => {
        const elapsed = (Date.now() - recordingStartTimeRef.current) / 1000;
        setRecordingTime(elapsed);
      }, 100);
    });

    controller.onRecordingStop((result: any) => {
      setIsRecording(false);
      setRecordingTime(0);
      addLog(`✅ Recording saved: ${result.filename}`);
      addLog(`   Steps: ${result.num_steps}, Duration: ${result.duration.toFixed(2)}s`);

      // Clear recording timer
      if (recordingIntervalRef.current !== null) {
        clearInterval(recordingIntervalRef.current);
        recordingIntervalRef.current = null;
      }
    });

    addLog('🎮 Keyboard controller initialized');

    // Cleanup
    return () => {
      if (recordingIntervalRef.current !== null) {
        clearInterval(recordingIntervalRef.current);
      }
      controller.destroy();
    };
  }, [robotType]);

  const handleStartRecording = (taskName: string) => {
    if (controllerRef.current) {
      controllerRef.current.startRecording(taskName);
    }
  };

  const handleStopRecording = () => {
    if (controllerRef.current) {
      controllerRef.current.stopRecording();
    }
  };

  const handleReset = () => {
    if (controllerRef.current) {
      controllerRef.current.resetRobot();
      addLog('🔄 Robot reset requested');
    }
  };

  const handleRobotTypeChange = (newType: RobotType) => {
    addLog(`🔄 Switching to ${newType} robot...`);
    setRobotType(newType);
  };

  return (
    <div className="teleop-test">
      {/* Header */}
      <div className="test-header">
        <h1>🎮 Teleoperation Test</h1>
        <p>Testing Genesis keyboard control with WebSocket + trajectory recording</p>

        {/* Robot Type Selector */}
        <div className="robot-selector">
          <label>Robot Type:</label>
          <div className="robot-buttons">
            <button
              className={`robot-btn ${robotType === 'mobile' ? 'active' : ''}`}
              onClick={() => handleRobotTypeChange('mobile')}
              disabled={isConnected}
            >
              🚗 Mobile
            </button>
            <button
              className={`robot-btn ${robotType === 'arm' ? 'active' : ''}`}
              onClick={() => handleRobotTypeChange('arm')}
              disabled={isConnected}
            >
              🦾 Arm
            </button>
            <button
              className={`robot-btn ${robotType === 'drone' ? 'active' : ''}`}
              onClick={() => handleRobotTypeChange('drone')}
              disabled={isConnected}
            >
              🚁 Drone
            </button>
          </div>
          {isConnected && (
            <p className="selector-note">⚠️ Disconnect to change robot type</p>
          )}
        </div>

        {/* Connection Status */}
        <div className={`status-banner ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? '🟢 Connected to Genesis' : '🔴 Disconnected'}
        </div>
      </div>

      {/* Main Content */}
      <div className="test-content">
        {/* Left Panel: State Visualization */}
        <div className="state-panel">
          <h3>📊 Robot State</h3>

          <div className="state-section">
            <h4>Current Action:</h4>
            <div className="action-display">
              {currentAction.length > 0 ? (
                <pre>{JSON.stringify(currentAction.map(v => v.toFixed(3)), null, 2)}</pre>
              ) : (
                <p className="no-data">No action data</p>
              )}
            </div>
          </div>

          <div className="state-section">
            <h4>Current State:</h4>
            <div className="state-display">
              {currentState ? (
                <pre>{JSON.stringify(currentState, null, 2)}</pre>
              ) : (
                <p className="no-data">No state data</p>
              )}
            </div>
          </div>

          {/* Keyboard State Indicator */}
          <div className="state-section">
            <h4>Keyboard Input:</h4>
            <div className="keyboard-indicator">
              {isConnected ? (
                <p>✅ Active - Use keyboard controls</p>
              ) : (
                <p>⏸️ Inactive - Connect to server</p>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel: Activity Log */}
        <div className="log-panel">
          <h3>📜 Activity Log</h3>
          <div className="log-content">
            {logs.length > 0 ? (
              logs.map((log, idx) => (
                <div key={idx} className="log-entry">
                  {log}
                </div>
              ))
            ) : (
              <p className="no-data">No activity yet</p>
            )}
          </div>
          <button
            className="clear-logs-btn"
            onClick={() => setLogs([])}
          >
            Clear Logs
          </button>
        </div>
      </div>

      {/* Instructions */}
      <div className="instructions-panel">
        <h3>📖 Test Instructions</h3>
        <div className="instructions-grid">
          <div className="instruction-card">
            <h4>1️⃣ Connect</h4>
            <p>Ensure backend is running at <code>localhost:8000</code></p>
            <p>Wait for green "Connected" status</p>
          </div>
          <div className="instruction-card">
            <h4>2️⃣ Control</h4>
            <p>Use keyboard to control the robot</p>
            <p>See control hints in the overlay (right side)</p>
          </div>
          <div className="instruction-card">
            <h4>3️⃣ Record</h4>
            <p>Click "Record Demo" to start trajectory recording</p>
            <p>Perform your demonstration</p>
            <p>Click "Stop Recording" to save CSV file</p>
          </div>
          <div className="instruction-card">
            <h4>4️⃣ Verify</h4>
            <p>Check <code>backend/trajectories/</code> for CSV files</p>
            <p>Verify action/state data in left panel</p>
            <p>Monitor activity log for errors</p>
          </div>
        </div>
      </div>

      {/* Control Overlay (floating) */}
      <ControlOverlay
        robotType={robotType}
        isRecording={isRecording}
        recordingSteps={recordingSteps}
        recordingTime={recordingTime}
        isConnected={isConnected}
        onStartRecording={handleStartRecording}
        onStopRecording={handleStopRecording}
        onReset={handleReset}
      />
    </div>
  );
}
